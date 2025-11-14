import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer
from . import basic as basic
from ..utils import utils
from ..configs.utils import update_from
from . import loss
from .loss import MatchCriterion
from .basic import torch_class_label_to_segment_label, time_mask


class CLIPTextEncoder(nn.Module):
    """
    CLIP text encoder for action descriptions
    Trainable during open-vocabulary FACT training
    """
    def __init__(self, clip_model_name="openai/clip-vit-b-32", trainable=True):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        self.text_dim = self.clip.config.projection_dim  # 512 for ViT-B/32
        
        # Make trainable (per user requirement 2.b)
        for param in self.clip.text_model.parameters():
            param.requires_grad = trainable
        
        # Also make text projection trainable
        for param in self.clip.text_projection.parameters():
            param.requires_grad = trainable
    
    def encode_text(self, text_descriptions):
        """
        Args:
            text_descriptions: List of strings
        Returns:
            text_embeddings: (num_texts, 512) - projected to CLIP embedding space
        """
        if isinstance(text_descriptions, list):
            inputs = self.tokenizer(
                text_descriptions, 
                padding=True, 
                truncation=True,
                max_length=77,  # CLIP max length
                return_tensors="pt"
            ).to(self.clip.device)
        else:
            inputs = text_descriptions
        
        # Get text features (goes through text_model + text_projection)
        text_outputs = self.clip.get_text_features(**inputs)
        return text_outputs  # (N, 512) - already in CLIP embedding space


class VisualFeatureProjection(nn.Module):
    """
    Projects pre-computed visual features (I3D, ResNet, etc.) 
    to CLIP-compatible embedding space
    
    This is the key component that enables using existing features
    without re-extraction!
    """
    def __init__(self, input_dim, clip_dim=512, hidden_dim=1024, dropout=0.1):
        super().__init__()
        
        # Multi-layer projection for better alignment
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, clip_dim),
            nn.LayerNorm(clip_dim)
        )
        
        # Initialize to preserve information
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, visual_features):
        """
        Args:
            visual_features: (T, B, input_dim) - e.g., (T, 1, 2048) for I3D
        Returns:
            projected_features: (T, B, 512) - CLIP-compatible
        """
        return self.projection(visual_features)


class FACT_OpenVocab(nn.Module):
    """
    Open-vocabulary FACT using feature projection approach
    
    Architecture:
    1. I3D features (2048) → Visual Projection → CLIP space (512)
    2. Action text → CLIP Text Encoder → CLIP space (512)
    3. Align visual and text embeddings via contrastive loss
    4. Maintain FACT's temporal modeling architecture
    """
    
    def __init__(self, cfg, visual_input_dim, action_descriptions):
        super().__init__()
        self.cfg = cfg
        
        # CLIP text encoder (trainable)
        self.clip_text = CLIPTextEncoder(
            clip_model_name=cfg.CLIP.model_name,
            trainable=cfg.CLIP.text_trainable
        )
        clip_dim = self.clip_text.text_dim  # 512
        
        # Visual feature projection: I3D (2048) → CLIP space (512)
        self.visual_projection = VisualFeatureProjection(
            input_dim=visual_input_dim,  # 2048 for I3D
            clip_dim=clip_dim,            # 512 for CLIP
            hidden_dim=cfg.CLIP.projection_hidden_dim,
            dropout=cfg.CLIP.projection_dropout
        )
        
        # Additional projection to FACT's internal dimensions
        base_cfg = cfg.Bi
        self.visual_to_fact = nn.Linear(clip_dim, base_cfg.hid_dim)
        self.text_to_fact = nn.Linear(clip_dim, base_cfg.a_dim)
        
        # FACT components (same as original)
        self.frame_pe = basic.PositionalEncoding(
            base_cfg.hid_dim, max_len=10000, empty=(not cfg.FACT.fpos)
        )
        self.channel_masking_dropout = nn.Dropout2d(p=cfg.FACT.cmr)
        
        # Action queries (when transcript not available)
        if not cfg.FACT.trans:
            self.action_query = nn.Parameter(
                torch.randn([cfg.FACT.ntoken, 1, base_cfg.a_dim])
            )
        else:
            self.action_pe = basic.PositionalEncoding(base_cfg.a_dim, max_len=1000)
        
        # Build FACT blocks (modified for open-vocabulary)
        block_list = []
        for i, t in enumerate(cfg.FACT.block):
            if t == 'i':
                block = InputBlock_OV(cfg, base_cfg.hid_dim)
            elif t == 'u':
                update_from(cfg.Bu, base_cfg, inplace=True)
                base_cfg = cfg.Bu
                block = UpdateBlock_OV(cfg)
            elif t == 'U':
                update_from(cfg.BU, base_cfg, inplace=True)
                base_cfg = cfg.BU
                block = UpdateBlockTDU_OV(cfg)
            block_list.append(block)
        
        self.block_list = nn.ModuleList(block_list)
        
        # Temperature for similarity scaling (learnable)
        self.temperature = nn.Parameter(torch.ones([]) * cfg.CLIP.temp)
        
        # Cache for pre-computed text embeddings (training efficiency)
        self.register_buffer('text_embeddings_clip', None)  # In CLIP space
        self.register_buffer('text_embeddings_fact', None)  # In FACT space
        self.action_descriptions = action_descriptions
        
        self.mcriterion = None
    
    def register_text_embeddings(self, text_embeddings_clip):
        """
        Cache pre-computed text embeddings for training efficiency
        
        Args:
            text_embeddings_clip: (C, 512) - text embeddings in CLIP space
        """
        self.text_embeddings_clip = text_embeddings_clip
        # Also project to FACT space
        self.text_embeddings_fact = self.text_to_fact(text_embeddings_clip)
    
    def get_text_embeddings(self, dynamic_descriptions=None):
        """
        Get text embeddings in both CLIP and FACT spaces
        
        Returns:
            (text_emb_clip, text_emb_fact): Both (C, dim) tensors
        """
        # Training: use cached
        if self.training and self.text_embeddings_clip is not None:
            return self.text_embeddings_clip, self.text_embeddings_fact
        
        # Zero-shot inference: encode dynamically
        if dynamic_descriptions is not None:
            text_emb_clip = self.clip_text.encode_text(dynamic_descriptions)
        else:
            text_emb_clip = self.clip_text.encode_text(self.action_descriptions)
        
        text_emb_fact = self.text_to_fact(text_emb_clip)
        return text_emb_clip, text_emb_fact
    
    def _forward_one_video(self, seq, transcript=None, text_embeddings=None):
        """
        Args:
            seq: (T, visual_dim) - pre-computed I3D features (2048-dim)
            transcript: (S,) - segment labels (optional)
            text_embeddings: tuple of (clip_emb, fact_emb) (optional)
        """
        # 1. Project I3D features to CLIP space, then to FACT space
        seq = seq.unsqueeze(1) if seq.dim() == 2 else seq  # (T, 1, 2048)
        
        visual_clip = self.visual_projection(seq)  # (T, 1, 512) - CLIP space
        frame_feature = self.visual_to_fact(visual_clip)  # (T, 1, hid_dim) - FACT space
        frame_pe = self.frame_pe(frame_feature)
        
        # Apply augmentations
        if self.cfg.FACT.cmr:
            frame_feature = frame_feature.permute([1, 2, 0])
            frame_feature = self.channel_masking_dropout(frame_feature)
            frame_feature = frame_feature.permute([2, 0, 1])
        
        if self.cfg.TM.use and self.training:
            frame_feature = time_mask(
                frame_feature, self.cfg.TM.t, self.cfg.TM.m, 
                self.cfg.TM.p, replace_with_zero=True
            )
        
        # 2. Get text embeddings
        if text_embeddings is None:
            text_emb_clip, text_emb_fact = self.get_text_embeddings()
        else:
            text_emb_clip, text_emb_fact = text_embeddings
        
        # 3. Prepare action features
        if not self.cfg.FACT.trans:
            action_pe = self.action_query
            action_feature = torch.zeros_like(action_pe)
        else:
            action_pe = self.action_pe(transcript)
            action_feature = text_emb_fact[transcript].unsqueeze(1)
            action_feature = action_feature + action_pe
            action_pe = torch.zeros_like(action_pe)
        
        # 4. Forward through FACT blocks
        block_output = []
        for block in self.block_list:
            frame_feature, action_feature = block(
                frame_feature, action_feature, frame_pe, action_pe,
                visual_clip_features=visual_clip,  # For similarity computation
                text_embeddings_clip=text_emb_clip,
                temperature=self.temperature
            )
            block_output.append([frame_feature, action_feature])
        
        return block_output
    
    def _loss_one_video(self, label):
        mcriterion = self.mcriterion
        mcriterion.set_label(label)
        
        # Get last block predictions
        last_block = self.block_list[-1]
        action_sim = last_block.action_sim  # (M, 1, C)
        
        # Match action tokens to GT segments
        action_prob = torch.softmax(action_sim / self.temperature, dim=-1)
        match = mcriterion.match(action_prob, last_block.a2f_attn)
        
        # Compute loss for each block
        loss_list = []
        for block in self.block_list:
            loss = block.compute_loss(mcriterion, match)
            loss_list.append(loss)
        
        self.loss_list = loss_list
        return sum(loss_list) / len(loss_list)
    
    def forward(self, seq_list, label_list, compute_loss=False):
        save_list = []
        final_loss = []
        
        for seq, label in zip(seq_list, label_list):
            seq = seq.unsqueeze(1) if seq.dim() == 2 else seq
            trans = torch_class_label_to_segment_label(label)[0]
            self._forward_one_video(seq, trans)
            
            pred = self.block_list[-1].eval(trans)
            save_data = {'pred': utils.to_numpy(pred)}
            save_list.append(save_data)
            
            if compute_loss:
                loss = self._loss_one_video(label)
                final_loss.append(loss)
                save_data['loss'] = {'loss': loss.item()}
        
        if compute_loss:
            return sum(final_loss) / len(final_loss), save_list
        return save_list
    
    def save_model(self, fname):
        torch.save(self.state_dict(), fname)


####################################################################
# Blocks

class Block_OV(nn.Module):
    """
    Base block for open-vocabulary FACT
    
    Key change: Compute similarity in CLIP embedding space,
    not in FACT's internal feature space
    """
    
    def __init__(self):
        super().__init__()
    
    def __str__(self):
        lines = f"{type(self).__name__}(\n  f:{self.frame_branch},\n  a:{self.action_branch},\n  a2f:{self.a2f_layer if hasattr(self, 'a2f_layer') else None},\n  f2a:{self.f2a_layer if hasattr(self, 'f2a_layer') else None}\n)"
        return lines
    
    def __repr__(self):
        return str(self)
    
    def compute_similarity(self, visual_clip, text_emb_clip, temperature):
        """
        Compute cosine similarity in CLIP embedding space
        
        Args:
            visual_clip: (T, B, 512) - visual features in CLIP space
            text_emb_clip: (C, 512) - text embeddings in CLIP space
            temperature: scalar
        Returns:
            similarity: (T, B, C)
        """
        # L2 normalize (CLIP embeddings are already normalized, but be explicit)
        visual_norm = F.normalize(visual_clip, dim=-1)  # (T, B, 512)
        text_norm = F.normalize(text_emb_clip, dim=-1)  # (C, 512)
        
        # Cosine similarity via dot product
        similarity = torch.einsum('tbh,ch->tbc', visual_norm, text_norm)
        similarity = similarity / temperature
        
        return similarity
    
    def create_fbranch(self, cfg, in_dim=None, f_inmap=False):
        if in_dim is None:
            in_dim = cfg.f_dim
        
        if cfg.f == 'm':  # use MSTCN
            frame_branch = basic.MSTCN(in_dim, cfg.f_dim, cfg.hid_dim, cfg.f_layers, 
                                dropout=cfg.dropout, ln=cfg.f_ln, ngroup=cfg.f_ngp, in_map=f_inmap)
        elif cfg.f == 'm2':  # use MSTCN++
            frame_branch = basic.MSTCN2(in_dim, cfg.f_dim, cfg.hid_dim, cfg.f_layers, 
                                dropout=cfg.dropout, ln=cfg.f_ln, ngroup=cfg.f_ngp, in_map=f_inmap)
        
        return frame_branch
    
    def create_abranch(self, cfg):
        if cfg.a == 'sa':  # self-attention layers, for update blocks
            l = basic.SALayer(cfg.a_dim, cfg.a_nhead, dim_feedforward=cfg.a_ffdim, dropout=cfg.dropout, attn_dropout=cfg.dropout)
            action_branch = basic.SADecoder(cfg.a_dim, cfg.a_dim, cfg.hid_dim, l, cfg.a_layers, in_map=False)
        elif cfg.a == 'sca':  # self+cross-attention layers, for input blocks
            layer = basic.SCALayer(cfg.a_dim, cfg.hid_dim, cfg.a_nhead, cfg.a_ffdim, dropout=cfg.dropout, attn_dropout=cfg.dropout)
            norm = torch.nn.LayerNorm(cfg.a_dim)
            action_branch = basic.SCADecoder(cfg.a_dim, cfg.a_dim, cfg.hid_dim, layer, cfg.a_layers, norm=norm, in_map=False)
        elif cfg.a in ['gru', 'gru_om']:  # GRU
            assert self.cfg.FACT.trans
            out_map = (cfg.a == 'gru_om')
            action_branch = basic.ActionUpdate_GRU(cfg.a_dim, cfg.a_dim, cfg.hid_dim, cfg.a_layers, dropout=cfg.dropout, out_map=out_map)
        else:
            raise ValueError(cfg.a)
        
        return action_branch
    
    def create_cross_attention(self, cfg, outdim, kq_pos=True):
        # one layer of cross-attention for cross-branch communication
        layer = basic.X2Y_map(cfg.hid_dim, cfg.hid_dim, outdim, 
            head_dim=cfg.hid_dim,
            dropout=cfg.dropout, kq_pos=kq_pos)
        
        return layer
    
    @staticmethod
    def _eval(action_sim, a2f_attn, frame_sim, weight):
        # Softmax over all classes for frame branch
        fbranch_prob_full = torch.softmax(frame_sim.squeeze(1), dim=-1)
        # Remove null class for final prediction
        fbranch_prob = fbranch_prob_full[:, :-1]
        
        action_sim = action_sim.squeeze(1)
        a2f_attn = a2f_attn.squeeze(0)
        qtk_cpred = action_sim.argmax(1) 
        null_cid = action_sim.shape[-1] - 1
        action_loc = torch.where(qtk_cpred!=null_cid)[0]
        
        if len(action_loc) == 0:
            return fbranch_prob.argmax(1)
        
        qtk_prob = torch.softmax(action_sim[:, :-1], dim=1)  # remove logit of null classes
        action_pred = a2f_attn[:, action_loc].argmax(-1)
        action_pred = action_loc[action_pred]
        abranch_prob = qtk_prob[action_pred]
        
        prob = (1-weight) * abranch_prob + weight * fbranch_prob
        return prob.argmax(1)
    
    @staticmethod
    def _eval_w_transcript(transcript, a2f_attn, frame_sim, weight):
        fbranch_prob = torch.softmax(frame_sim.squeeze(1), dim=-1)
        fbranch_prob = fbranch_prob[:, transcript] 
        
        N = len(transcript)
        a2f_attn = a2f_attn[0, :, :N]  # 1, f, a -> f, s'
        abranch_prob = torch.softmax(a2f_attn, dim=-1)  # f, s'
        
        prob = (1-weight) * abranch_prob + weight * fbranch_prob
        pred = prob.argmax(1)  # f
        pred = transcript[pred]
        return pred
    
    def eval(self, transcript=None):
        if not self.cfg.FACT.trans:
            return self._eval(self.action_sim, self.a2f_attn, self.frame_sim, self.cfg.FACT.mwt)
        else:
            return self._eval_w_transcript(transcript, self.a2f_attn, self.frame_sim, self.cfg.FACT.mwt)


class InputBlock_OV(Block_OV):
    """Input block adapted for open-vocabulary"""
    
    def __init__(self, cfg, in_dim):
        super().__init__()
        self.cfg = cfg
        cfg_block = cfg.Bi
        
        self.frame_branch = self.create_fbranch(cfg_block, in_dim, f_inmap=True)
        self.action_branch = self.create_abranch(cfg_block)
        
        # Additional projection: FACT features → CLIP space for similarity
        self.frame_to_clip = nn.Linear(cfg_block.hid_dim, 512)
        self.action_to_clip = nn.Linear(cfg_block.a_dim, 512)
    
    def forward(self, frame_feature, action_feature, frame_pos, action_pos,
                visual_clip_features=None, text_embeddings_clip=None, temperature=1.0):
        # Frame branch (in FACT space)
        frame_feature = self.frame_branch(frame_feature)  # (T, 1, hid_dim)
        
        # Action branch (in FACT space)
        action_feature = self.action_branch(
            action_feature, frame_feature, 
            pos=frame_pos, query_pos=action_pos
        )
        
        # Project back to CLIP space for similarity computation
        frame_clip = self.frame_to_clip(frame_feature)  # (T, 1, 512)
        action_clip = self.action_to_clip(action_feature)  # (M, 1, 512)
        
        # Compute similarities in CLIP space
        self.frame_sim = self.compute_similarity(
            frame_clip, text_embeddings_clip, temperature
        )  # (T, 1, C)
        self.action_sim = self.compute_similarity(
            action_clip, text_embeddings_clip, temperature
        )  # (M, 1, C)
        
        # Store for evaluation (needed by eval method)
        self.a2f_attn = None  # Will be set by parent if needed
        
        return frame_feature, action_feature
    
    def compute_loss(self, criterion, match=None):
        """Use contrastive losses"""
        frame_loss = criterion.frame_loss(self.frame_sim.squeeze(1))
        atk_loss = criterion.action_token_loss(match, self.action_sim.squeeze(1))
        
        # Smooth loss on similarity scores
        smooth_loss = loss.smooth_loss(
            self.frame_sim.transpose(0, 1), is_logit=True
        )
        
        return frame_loss + atk_loss + self.cfg.Loss.sw * smooth_loss


class UpdateBlock_OV(Block_OV):
    """Update block adapted for open-vocabulary"""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        cfg_block = cfg.Bu
        
        # fbranch
        self.frame_branch = self.create_fbranch(cfg_block)
        
        # f2a: query is action
        self.f2a_layer = self.create_cross_attention(cfg_block, cfg_block.a_dim)
        
        # abranch
        self.action_branch = self.create_abranch(cfg_block)
        
        # a2f: query is frame
        self.a2f_layer = self.create_cross_attention(cfg_block, cfg_block.f_dim)
        
        # Additional projection: FACT features → CLIP space for similarity
        self.frame_to_clip = nn.Linear(cfg_block.hid_dim, 512)
        self.action_to_clip = nn.Linear(cfg_block.a_dim, 512)
    
    def forward(self, frame_feature, action_feature, frame_pos, action_pos,
                visual_clip_features=None, text_embeddings_clip=None, temperature=1.0):
        # a->f
        action_feature = self.f2a_layer(frame_feature, action_feature, X_pos=frame_pos, Y_pos=action_pos)
        
        # a branch
        action_feature = self.action_branch(action_feature, action_pos)
        
        # f->a
        frame_feature = self.a2f_layer(action_feature, frame_feature, X_pos=action_pos, Y_pos=frame_pos)
        
        # f branch
        frame_feature = self.frame_branch(frame_feature)
        
        # Project to CLIP space for similarity computation
        frame_clip = self.frame_to_clip(frame_feature)
        action_clip = self.action_to_clip(action_feature)
        
        # Compute similarities
        self.frame_sim = self.compute_similarity(frame_clip, text_embeddings_clip, temperature)
        self.action_sim = self.compute_similarity(action_clip, text_embeddings_clip, temperature)
        
        # Save attention for loss and evaluation
        self.f2a_attn = self.f2a_layer.attn[0]
        self.a2f_attn = self.a2f_layer.attn[0]
        self.f2a_attn_logit = self.f2a_layer.attn_logit[0].unsqueeze(0)
        self.a2f_attn_logit = self.a2f_layer.attn_logit[0].unsqueeze(0)
        
        return frame_feature, action_feature
    
    def compute_loss(self, criterion, match=None):
        frame_loss = criterion.frame_loss(self.frame_sim.squeeze(1))
        atk_loss = criterion.action_token_loss(match, self.action_sim.squeeze(1))
        f2a_loss = criterion.cross_attn_loss(match, torch.transpose(self.f2a_attn_logit, 1, 2), dim=1)
        a2f_loss = criterion.cross_attn_loss(match, self.a2f_attn_logit, dim=2)
        
        # temporal smoothing loss
        al = loss.smooth_loss(self.a2f_attn_logit, is_logit=True)
        fl = loss.smooth_loss(torch.transpose(self.f2a_attn_logit, 1, 2), is_logit=True)
        frame_sim_trans = torch.transpose(self.frame_sim, 0, 1)  # f, 1, c -> 1, f, c
        l = loss.smooth_loss(frame_sim_trans, is_logit=True)
        smooth_loss = al + fl + l
        
        return atk_loss + f2a_loss + a2f_loss + frame_loss + self.cfg.Loss.sw * smooth_loss


class UpdateBlockTDU_OV(Block_OV):
    """
    Update Block with Temporal Downsampling and Upsampling (Open Vocabulary)
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        cfg_block = cfg.BU
        
        # fbranch
        self.frame_branch = self.create_fbranch(cfg_block)
        
        # layers for temporal downsample and upsample
        self.seg_update = nn.GRU(cfg_block.hid_dim, cfg_block.hid_dim//2, cfg_block.s_layers, bidirectional=True)
        self.seg_combine = nn.Linear(cfg_block.hid_dim, cfg_block.hid_dim)
        
        # f2a: query is action
        self.f2a_layer = self.create_cross_attention(cfg_block, cfg_block.a_dim)
        
        # abranch
        self.action_branch = self.create_abranch(cfg_block)
        
        # a2f: query is frame
        self.a2f_layer = self.create_cross_attention(cfg_block, cfg_block.f_dim)
        
        # layers for temporal downsample and upsample
        self.sf_merge = nn.Sequential(nn.Linear((cfg_block.hid_dim+cfg_block.f_dim), cfg_block.f_dim), nn.ReLU())
        
        # Additional projection: FACT features → CLIP space for similarity
        self.frame_to_clip = nn.Linear(cfg_block.hid_dim, 512)
        self.action_to_clip = nn.Linear(cfg_block.a_dim, 512)
        self.seg_to_clip = nn.Linear(cfg_block.hid_dim, 512)
    
    def temporal_downsample(self, frame_feature, frame_sim):
        # get action segments based on predictions
        _, pred = frame_sim[:, 0].max(dim=-1)
        pred = utils.to_numpy(pred)
        segs = utils.parse_label(pred)
        
        tdu = basic.TemporalDownsampleUpsample(segs)
        tdu.to(frame_feature.device)
        
        # downsample frames to segments
        seg_feature = tdu.feature_frame2seg(frame_feature)
        
        # refine segment features
        seg_feature, hidden = self.seg_update(seg_feature)
        seg_feature = torch.relu(seg_feature)
        seg_feature = self.seg_combine(seg_feature)  # combine forward and backward features
        
        return tdu, seg_feature
    
    def temporal_upsample(self, tdu, seg_feature, frame_feature):
        # upsample segments to frames
        s2f = tdu.feature_seg2frame(seg_feature)
        
        # merge with original framewise features to keep low-level details
        frame_feature = self.sf_merge(torch.cat([s2f, frame_feature], dim=-1))
        
        return frame_feature
    
    def forward(self, frame_feature, action_feature, frame_pos, action_pos,
                visual_clip_features=None, text_embeddings_clip=None, temperature=1.0):
        # First compute frame similarity for temporal downsampling
        frame_clip_init = self.frame_to_clip(frame_feature)
        frame_sim_init = self.compute_similarity(frame_clip_init, text_embeddings_clip, temperature)
        
        # downsample frame features to segment features
        tdu, seg_feature = self.temporal_downsample(frame_feature, frame_sim_init)  # seg_feature: S, 1, H
        
        # f->a
        seg_center = torch.LongTensor([int((s.start+s.end)/2) for s in tdu.segs]).to(seg_feature.device)
        seg_pos = frame_pos[seg_center]
        action_feature = self.f2a_layer(seg_feature, action_feature, X_pos=seg_pos, Y_pos=action_pos)
        
        # a branch
        action_feature = self.action_branch(action_feature, action_pos)
        
        # a->f
        seg_feature = self.a2f_layer(action_feature, seg_feature, X_pos=action_pos, Y_pos=seg_pos)
        
        # upsample segment features to frame features
        frame_feature = self.temporal_upsample(tdu, seg_feature, frame_feature)
        
        # f branch
        frame_feature = self.frame_branch(frame_feature)
        
        # Project to CLIP space for final similarity computation
        frame_clip = self.frame_to_clip(frame_feature)
        action_clip = self.action_to_clip(action_feature)
        seg_clip = self.seg_to_clip(seg_feature)
        
        # Compute similarities
        self.frame_sim = self.compute_similarity(frame_clip, text_embeddings_clip, temperature)
        self.action_sim = self.compute_similarity(action_clip, text_embeddings_clip, temperature)
        self.seg_sim = self.compute_similarity(seg_clip, text_embeddings_clip, temperature)
        
        # save features for loss and evaluation       
        self.tdu = tdu
        
        self.f2a_attn_logit = self.f2a_layer.attn_logit[0].unsqueeze(0)
        self.f2a_attn = tdu.attn_seg2frame(self.f2a_layer.attn[0].transpose(2, 1)).transpose(2, 1)
        self.a2f_attn_logit = self.a2f_layer.attn_logit[0].unsqueeze(0) 
        self.a2f_attn = tdu.attn_seg2frame(self.a2f_layer.attn[0])
        
        return frame_feature, action_feature
    
    def compute_loss(self, criterion, match=None):
        frame_loss = criterion.frame_loss(self.frame_sim.squeeze(1))
        seg_loss = criterion.frame_loss_tdu(self.seg_sim, self.tdu)
        atk_loss = criterion.action_token_loss(match, self.action_sim.squeeze(1))
        f2a_loss = criterion.cross_attn_loss_tdu(match, torch.transpose(self.f2a_attn_logit, 1, 2), self.tdu, dim=1)
        a2f_loss = criterion.cross_attn_loss_tdu(match, self.a2f_attn_logit, self.tdu, dim=2)
        
        frame_sim_trans = torch.transpose(self.frame_sim, 0, 1) 
        smooth_loss = loss.smooth_loss(frame_sim_trans, is_logit=True)
        
        return (frame_loss + seg_loss) / 2 + atk_loss + f2a_loss + a2f_loss + self.cfg.Loss.sw * smooth_loss



