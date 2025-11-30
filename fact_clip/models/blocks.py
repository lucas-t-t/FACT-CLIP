import torch
import torch.nn as nn
import torch.nn.functional as F
from . import basic as basic
from ..utils import utils
from ..configs.utils import update_from
from . import loss
from .loss import MatchCriterion
from .basic import torch_class_label_to_segment_label, time_mask

# CLIP imports for open-vocabulary FACT
try:
    from transformers import CLIPModel, CLIPTokenizer
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: transformers library not available. CLIP functionality will be disabled.")

class FACT(nn.Module):

    def __init__(self, cfg, in_dim, n_classes):
        super().__init__()
        self.cfg = cfg
        self.num_classes = n_classes

        base_cfg = cfg.Bi
        self.frame_pe = basic.PositionalEncoding(base_cfg.hid_dim, max_len=10000, empty=(not cfg.FACT.fpos) )
        self.channel_masking_dropout = nn.Dropout2d(p=cfg.FACT.cmr)

        if not cfg.FACT.trans : # when video transcript is not available at training and inference
            self.action_query = nn.Parameter(torch.randn([cfg.FACT.ntoken, 1, base_cfg.a_dim]))
        else: # when video transcript is available
            self.action_pe = basic.PositionalEncoding(base_cfg.a_dim, max_len=1000)
            self.action_embed = nn.Embedding(n_classes, base_cfg.a_dim)

        # block configuration
        block_list = []
        for i, t in enumerate(cfg.FACT.block):
            if t == 'i':
                block = InputBlock(cfg, in_dim, n_classes)
            elif t == 'u':
                update_from(cfg.Bu, base_cfg, inplace=True)
                base_cfg = cfg.Bu
                block = UpdateBlock(cfg, n_classes)
            elif t == 'U':
                update_from(cfg.BU, base_cfg, inplace=True)
                base_cfg = cfg.BU
                block = UpdateBlockTDU(cfg, n_classes)

            block_list.append(block)

        self.block_list = nn.ModuleList(block_list)

        self.mcriterion = None

    def _forward_one_video(self, seq, transcript=None):
        # prepare frame feature
        frame_feature = seq
        frame_pe = self.frame_pe(seq)
        if self.cfg.FACT.cmr:
            frame_feature = frame_feature.permute([1, 2, 0])
            frame_feature = self.channel_masking_dropout(frame_feature)
            frame_feature = frame_feature.permute([2, 0, 1])

        if self.cfg.TM.use and self.training:
            frame_feature = time_mask(frame_feature, 
                        self.cfg.TM.t, self.cfg.TM.m, self.cfg.TM.p, 
                        replace_with_zero=True)

        # prepare action feature
        if not self.cfg.FACT.trans:
            action_pe = self.action_query # M, B(=1), H
            action_feature = torch.zeros_like(action_pe)
        else:
            action_pe = self.action_pe(transcript)
            action_feature = self.action_embed(transcript).unsqueeze(1)

            action_feature = action_feature + action_pe
            action_pe = torch.zeros_like(action_pe)

        # forward
        # frame_feature: T, B(=1), H
        # action_feature: M, B(=1), H
        block_output = []
        for i, block in enumerate(self.block_list):
            frame_feature, action_feature = block(frame_feature, action_feature, frame_pe, action_pe)
            block_output.append([frame_feature, action_feature])
        return block_output

    def _loss_one_video(self, label):
        mcriterion: MatchCriterion = self.mcriterion
        mcriterion.set_label(label)

        block : Block = self.block_list[-1]
        cprob = basic.logit2prob(block.action_clogit, dim=-1)
        match = mcriterion.match(cprob, block.a2f_attn)

        ######## per block loss
        loss_list = []
        for block in self.block_list:
            loss = block.compute_loss(mcriterion, match)
            loss_list.append(loss)

        self.loss_list = loss_list
        final_loss = sum(loss_list) / len(loss_list)
        return final_loss

    def forward(self, seq_list, label_list, compute_loss=False):

        save_list = []
        final_loss = []

        for i, (seq, label) in enumerate(zip(seq_list, label_list)):
            seq = seq.unsqueeze(1)
            trans = torch_class_label_to_segment_label(label)[0]
            self._forward_one_video(seq, trans)

            pred = self.block_list[-1].eval(trans)
            save_data = {'pred': utils.to_numpy(pred)}
            save_list.append(save_data)

            if compute_loss:
                loss = self._loss_one_video(label)
                final_loss.append(loss)
                save_data['loss'] = { 'loss': loss.item() }


        if compute_loss:
            final_loss = sum(final_loss) / len(final_loss)
            return final_loss, save_list
        else:
            return save_list

    def save_model(self, fname):
        torch.save(self.state_dict(), fname)

####################################################################
# CLIP Components for Open-Vocabulary FACT
####################################################################

class FeatureProjection(nn.Module):
    """
    Project features (action or frame) to CLIP embedding space.
    Maps from (..., feature_dim) features to (..., clip_dim) CLIP embeddings.
    Uses class-agnostic features for better zero-shot generalization.
    """
    def __init__(self, feature_dim, clip_dim=512, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.clip_dim = clip_dim
        
        # Project features to CLIP space
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, clip_dim)
        )
    
    def forward(self, feature):
        """
        Project features to CLIP space.
        
        Args:
            feature: (..., feature_dim) - input features
        
        Returns:
            projected_embeddings: (..., clip_dim) - CLIP embeddings
        """
        # Project features to CLIP space
        projected = self.projection(feature)
        # Normalize embeddings
        projected = F.normalize(projected, dim=-1)
        return projected

####################################################################
# Blocks

class Block(nn.Module):
    """
    Base Block class for common functions
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        lines = f"{type(self).__name__}(\n  f:{self.frame_branch},\n  a:{self.action_branch},\n  a2f:{self.a2f_layer if hasattr(self, 'a2f_layer') else None},\n  f2a:{self.f2a_layer if hasattr(self, 'f2a_layer') else None}\n)"
        return lines

    def __repr__(self):
        return str(self)

    def process_feature(self, feature, nclass):
        # use the last several dimension as logit of action classes
        clogit = feature[:, :, -nclass:] # class logit
        feature = feature[:, :, :-nclass] # feature without clogit
        cprob = basic.logit2prob(clogit, dim=-1)  # apply softmax
        feature = torch.cat([feature, cprob], dim=-1)

        return feature, clogit

    def create_fbranch(self, cfg, in_dim=None, f_inmap=False):
        if in_dim is None:
            in_dim = cfg.f_dim

        if cfg.f == 'm': # use MSTCN
            frame_branch = basic.MSTCN(in_dim, cfg.f_dim, cfg.hid_dim, cfg.f_layers, 
                                dropout=cfg.dropout, ln=cfg.f_ln, ngroup=cfg.f_ngp, in_map=f_inmap)
        elif cfg.f == 'm2': # use MSTCN++
            frame_branch = basic.MSTCN2(in_dim, cfg.f_dim, cfg.hid_dim, cfg.f_layers, 
                                dropout=cfg.dropout, ln=cfg.f_ln, ngroup=cfg.f_ngp, in_map=f_inmap)

        return frame_branch

    def create_abranch(self, cfg):
        if cfg.a == 'sa': # self-attention layers, for update blocks
            l = basic.SALayer(cfg.a_dim, cfg.a_nhead, dim_feedforward=cfg.a_ffdim, dropout=cfg.dropout, attn_dropout=cfg.dropout)
            action_branch = basic.SADecoder(cfg.a_dim, cfg.a_dim, cfg.hid_dim, l, cfg.a_layers, in_map=False)
        elif cfg.a == 'sca': # self+cross-attention layers, for input blocks when video transcripts are not available
            layer = basic.SCALayer(cfg.a_dim, cfg.hid_dim, cfg.a_nhead, cfg.a_ffdim, dropout=cfg.dropout, attn_dropout=cfg.dropout)
            norm = torch.nn.LayerNorm(cfg.a_dim)
            action_branch = basic.SCADecoder(cfg.a_dim, cfg.a_dim, cfg.hid_dim, layer, cfg.a_layers, norm=norm, in_map=False)
        elif cfg.a in ['gru', 'gru_om']: # GRU, for input blocks when video transcripts are available
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
    def _eval(action_clogit, a2f_attn, frame_clogit, weight):
        fbranch_prob = torch.softmax(frame_clogit.squeeze(1), dim=-1)

        action_clogit = action_clogit.squeeze(1)
        a2f_attn = a2f_attn.squeeze(0)
        qtk_cpred = action_clogit.argmax(1) 
        null_cid = action_clogit.shape[-1] - 1
        action_loc = torch.where(qtk_cpred!=null_cid)[0]

        if len(action_loc) == 0:
            return fbranch_prob.argmax(1)

        qtk_prob = torch.softmax(action_clogit[:, :-1], dim=1) # remove logit of null classes
        action_pred = a2f_attn[:, action_loc].argmax(-1)
        action_pred = action_loc[action_pred]
        abranch_prob = qtk_prob[action_pred]

        prob = (1-weight) * abranch_prob + weight * fbranch_prob
        return prob.argmax(1)

    @staticmethod
    def _eval_w_transcript(transcript, a2f_attn, frame_clogit, weight):
        fbranch_prob = torch.softmax(frame_clogit.squeeze(1), dim=-1)
        fbranch_prob = fbranch_prob[:, transcript] 

        N = len(transcript)
        a2f_attn = a2f_attn[0, :, :N] # 1, f, a -> f, s'
        abranch_prob = torch.softmax(a2f_attn, dim=-1) # f, s'

        prob = (1-weight) * abranch_prob + weight * fbranch_prob
        pred = prob.argmax(1) # f
        pred = transcript[pred]
        return pred

    def eval(self, transcript=None):
        if not self.cfg.FACT.trans:
            return self._eval(self.action_clogit, self.a2f_attn, self.frame_clogit, self.cfg.FACT.mwt)
        else:
            return self._eval_w_transcript(transcript, self.a2f_attn, self.frame_clogit, self.cfg.FACT.mwt)


class InputBlock(Block):
    def __init__(self, cfg, in_dim, nclass):
        super().__init__()
        self.cfg = cfg
        self.nclass = nclass

        cfg = cfg.Bi

        self.frame_branch = self.create_fbranch(cfg, in_dim, f_inmap=True)
        self.action_branch = self.create_abranch(cfg)

    def forward(self, frame_feature, action_feature, frame_pos, action_pos, action_clogit=None):
        # frame branch
        frame_feature = self.frame_branch(frame_feature)
        frame_feature, frame_clogit = self.process_feature(frame_feature, self.nclass)

        # action branch
        action_feature = self.action_branch(action_feature, frame_feature, pos=frame_pos, query_pos=action_pos)
        action_feature, action_clogit = self.process_feature(action_feature, self.nclass+1)
        
        # save features for loss and evaluation
        self.frame_clogit = frame_clogit 
        self.action_clogit = action_clogit
        # Save action feature without class probabilities for CLIP projection
        # action_feature contains [features, class_probs], we need only features
        self.action_feature = action_feature[:, :, :-(self.nclass+1)]

        return frame_feature, action_feature

    def compute_loss(self, criterion: loss.MatchCriterion, match=None):
        frame_loss = criterion.frame_loss(self.frame_clogit.squeeze(1))
        atk_loss = criterion.action_token_loss(match, self.action_clogit)

        frame_clogit = torch.transpose(self.frame_clogit, 0, 1) 
        smooth_loss = loss.smooth_loss(frame_clogit)

        return frame_loss + atk_loss + self.cfg.Loss.sw * smooth_loss

class UpdateBlock(Block):

    def __init__(self, cfg, nclass):
        super().__init__()
        self.cfg = cfg
        self.nclass = nclass

        cfg = cfg.Bu

        # fbranch
        self.frame_branch = self.create_fbranch(cfg)

        # f2a: query is action
        self.f2a_layer = self.create_cross_attention(cfg, cfg.a_dim)

        # abranch
        self.action_branch = self.create_abranch(cfg)

        # a2f: query is frame
        self.a2f_layer = self.create_cross_attention(cfg, cfg.f_dim)

    def forward(self, frame_feature, action_feature, frame_pos, action_pos):
        # a->f
        action_feature = self.f2a_layer(frame_feature, action_feature, X_pos=frame_pos, Y_pos=action_pos)

        # a branch
        action_feature = self.action_branch(action_feature, action_pos)
        action_feature, action_clogit = self.process_feature(action_feature, self.nclass+1)

        # f->a
        frame_feature = self.a2f_layer(action_feature, frame_feature, X_pos=action_pos, Y_pos=frame_pos)

        # f branch
        frame_feature = self.frame_branch(frame_feature)
        frame_feature, frame_clogit = self.process_feature(frame_feature, self.nclass)

        # save features for loss and evaluation
        self.frame_clogit = frame_clogit 
        self.action_clogit = action_clogit 
        # Save action feature without class probabilities for CLIP projection
        self.action_feature = action_feature[:, :, :-(self.nclass+1)]
        self.f2a_attn = self.f2a_layer.attn[0]
        self.a2f_attn = self.a2f_layer.attn[0]
        self.f2a_attn_logit = self.f2a_layer.attn_logit[0].unsqueeze(0)
        self.a2f_attn_logit = self.a2f_layer.attn_logit[0].unsqueeze(0)
        return frame_feature, action_feature

    def compute_loss(self, criterion: loss.MatchCriterion, match=None):
        frame_loss = criterion.frame_loss(self.frame_clogit.squeeze(1)) 
        atk_loss = criterion.action_token_loss(match, self.action_clogit)
        f2a_loss = criterion.cross_attn_loss(match, torch.transpose(self.f2a_attn_logit, 1, 2), dim=1)
        a2f_loss = criterion.cross_attn_loss(match, self.a2f_attn_logit, dim=2)

        # temporal smoothing loss
        al = loss.smooth_loss( self.a2f_attn_logit )
        fl = loss.smooth_loss( torch.transpose(self.f2a_attn_logit, 1, 2) )
        frame_clogit = torch.transpose(self.frame_clogit, 0, 1) # f, 1, c -> 1, f, c
        l = loss.smooth_loss( frame_clogit )
        smooth_loss = al + fl + l

        return atk_loss + f2a_loss + a2f_loss + frame_loss + self.cfg.Loss.sw * smooth_loss


class UpdateBlockTDU(Block):
    """
    Update Block with Temporal Downsampling and Upsampling
    """

    def __init__(self, cfg, nclass):
        super().__init__()
        self.cfg = cfg
        self.nclass = nclass

        cfg = cfg.BU

        # fbranch
        self.frame_branch = self.create_fbranch(cfg)

        # layers for temporal downsample and upsample
        self.seg_update = nn.GRU(cfg.hid_dim, cfg.hid_dim//2, cfg.s_layers, bidirectional=True)
        self.seg_combine = nn.Linear(cfg.hid_dim, cfg.hid_dim)

        # f2a: query is action
        self.f2a_layer = self.create_cross_attention(cfg, cfg.a_dim)

        # abranch
        self.action_branch = self.create_abranch(cfg)

        # a2f: query is frame
        self.a2f_layer = self.create_cross_attention(cfg, cfg.f_dim)

        # layers for temporal downsample and upsample
        self.sf_merge = nn.Sequential(nn.Linear((cfg.hid_dim+cfg.f_dim), cfg.f_dim), nn.ReLU())


    def temporal_downsample(self, frame_feature):

        # get action segments based on predictions
        cprob = frame_feature[:, :, -self.nclass:]
        _, pred = cprob[:, 0].max(dim=-1)
        pred = utils.to_numpy(pred)
        segs = utils.parse_label(pred)

        tdu = basic.TemporalDownsampleUpsample(segs)
        tdu.to(cprob.device)

        # downsample frames to segments
        seg_feature = tdu.feature_frame2seg(frame_feature)

        # refine segment features
        seg_feature, hidden = self.seg_update(seg_feature)
        seg_feature = torch.relu(seg_feature)
        seg_feature = self.seg_combine(seg_feature) # combine forward and backward features
        seg_feature, seg_clogit = self.process_feature(seg_feature, self.nclass)

        return tdu, seg_feature, seg_clogit

    def temporal_upsample(self, tdu, seg_feature, frame_feature):

        # upsample segments to frames
        s2f = tdu.feature_seg2frame(seg_feature)
        
        # merge with original framewise features to keep low-level details
        frame_feature = self.sf_merge(torch.cat([s2f, frame_feature], dim=-1))

        return frame_feature

    def forward(self, frame_feature, action_feature, frame_pos, action_pos):
        # downsample frame features to segment features
        tdu, seg_feature, seg_clogit = self.temporal_downsample(frame_feature) # seg_feature: S, 1, H

        # f->a
        seg_center = torch.LongTensor([ int( (s.start+s.end)/2 ) for s in tdu.segs ]).to(seg_feature.device)
        seg_pos = frame_pos[seg_center]
        action_feature = self.f2a_layer(seg_feature, action_feature, X_pos=seg_pos, Y_pos=action_pos)

        # a branch
        action_feature = self.action_branch(action_feature, action_pos)
        action_feature, action_clogit = self.process_feature(action_feature, self.nclass+1)

        # a->f
        seg_feature = self.a2f_layer(action_feature, seg_feature, X_pos=action_pos, Y_pos=seg_pos)

        # upsample segment features to frame features
        frame_feature = self.temporal_upsample(tdu, seg_feature, frame_feature)

        # f branch
        frame_feature = self.frame_branch(frame_feature)
        frame_feature, frame_clogit = self.process_feature(frame_feature, self.nclass)

        # save features for loss and evaluation       
        self.frame_clogit = frame_clogit 
        self.seg_clogit = seg_clogit
        self.tdu = tdu
        self.action_clogit = action_clogit 
        # Save action feature without class probabilities for CLIP projection
        self.action_feature = action_feature[:, :, :-(self.nclass+1)]

        self.f2a_attn_logit = self.f2a_layer.attn_logit[0].unsqueeze(0)
        self.f2a_attn = tdu.attn_seg2frame(self.f2a_layer.attn[0].transpose(2, 1)).transpose(2, 1)
        self.a2f_attn_logit = self.a2f_layer.attn_logit[0].unsqueeze(0) 
        self.a2f_attn = tdu.attn_seg2frame(self.a2f_layer.attn[0])

        return frame_feature, action_feature

    def compute_loss(self, criterion: MatchCriterion, match=None):
        frame_loss = criterion.frame_loss(self.frame_clogit.squeeze(1))
        seg_loss = criterion.frame_loss_tdu(self.seg_clogit, self.tdu)
        atk_loss = criterion.action_token_loss(match, self.action_clogit)
        f2a_loss = criterion.cross_attn_loss_tdu(match, torch.transpose(self.f2a_attn_logit, 1, 2), self.tdu, dim=1)
        a2f_loss = criterion.cross_attn_loss_tdu(match, self.a2f_attn_logit, self.tdu, dim=2)

        frame_clogit = torch.transpose(self.frame_clogit, 0, 1) 
        smooth_loss = loss.smooth_loss( frame_clogit )

        return (frame_loss + seg_loss)/ 2 + atk_loss + f2a_loss + a2f_loss + self.cfg.Loss.sw * smooth_loss


####################################################################
# FACT_CLIP: Open-Vocabulary FACT with CLIP
####################################################################

class FACT_CLIP(nn.Module):
    """
    Open-vocabulary FACT model with CLIP text encoder.
    Extends vanilla FACT by adding CLIP text embeddings and contrastive loss.
    """

    def __init__(self, cfg, in_dim, n_classes, text_embeddings=None):
        super().__init__()
        self.cfg = cfg
        self.num_classes = n_classes

        base_cfg = cfg.Bi
        self.frame_pe = basic.PositionalEncoding(base_cfg.hid_dim, max_len=10000, empty=(not cfg.FACT.fpos))
        self.channel_masking_dropout = nn.Dropout2d(p=cfg.FACT.cmr)

        if not cfg.FACT.trans:
            self.action_query = nn.Parameter(torch.randn([cfg.FACT.ntoken, 1, base_cfg.a_dim]))
        else:
            self.action_pe = basic.PositionalEncoding(base_cfg.a_dim, max_len=1000)
            self.action_embed = nn.Embedding(n_classes, base_cfg.a_dim)

        # CLIP components
        if not CLIP_AVAILABLE:
            raise ImportError("transformers library required for FACT_CLIP")
        
        # Calculate actual frame feature dimension
        # Frame branch outputs hid_dim dimensions (from Bi config)
        # But the actual frame_feature has `n_classes` removed in process_feature()
        # Wait, process_feature() removes the LAST n_classes (logits) and keeps the rest.
        # But frame_branch output depends on the architecture (MSTCN/MSTCN2).
        # In Block.process_feature:
        #   clogit = feature[:, :, -nclass:]
        #   feature = feature[:, :, :-nclass]
        # So the feature dimension is (output_dim - n_classes).
        
        # The output dimension of the frame branch (e.g. MSTCN) is cfg.f_dim (from Bi/Bu/BU config).
        # Let's check MSTCN... it outputs cfg.f_dim.
        # But wait, MSTCN output is passed to process_feature.
        # So MSTCN output MUST be (feature_dim + n_classes).
        # The config `cfg.f_dim` seems to be the output dimension of MSTCN.
        # So `feature_dim` = cfg.f_dim - n_classes.
        
        # However, FACT blocks use `base_cfg` which is updated.
        # The last block is typically an UpdateBlock (u/U).
        # Let's assume the last block uses the config from the last updated base_cfg.
        # But strictly speaking, we should look at the last block instance.
        
        # Let's look at how `action_feature_dim` was calculated previously:
        # action_feature_dim = base_cfg.hid_dim - (n_classes + 1)
        # This was using base_cfg which is the config of the LAST block added (because of the loop).
        
        # For frame features:
        # If last block is UpdateBlock(u) or UpdateBlockTDU(U):
        #   f_dim usually comes from Bu.f_dim or BU.f_dim ??
        # Actually, in UpdateBlock, create_fbranch uses cfg.f_dim.
        # And cfg here is the updated base_cfg.
        
        # Wait, `cfg.f_dim` in `create_fbranch` is the output dimension.
        # NO! In create_fbranch, it calls MSTCN(in_dim, cfg.f_dim, cfg.hid_dim)
        # In MSTCN init(in_dim, hid_dim, out_dim), the 3rd arg is OUT_DIM.
        # So out_dim = cfg.hid_dim!
        
        # So frame_feature_dim = base_cfg.hid_dim - n_classes.
        
        frame_feature_dim = base_cfg.hid_dim - n_classes
        
        print(f"\n[FACT_CLIP Init] Calculated dimensions:")
        print(f"  - Total classes: {n_classes}")
        print(f"  - Frame output dim (hid_dim): {base_cfg.hid_dim}")
        print(f"  - Frame feature dimension: {frame_feature_dim}")
        print(f"  - CLIP projection: {frame_feature_dim} -> 512\n")
        
        # Projection layer for frame features to CLIP space
        self.frame_projection = FeatureProjection(
            feature_dim=frame_feature_dim,
            clip_dim=512,
            hidden_dim=cfg.CLIP.projection_hidden_dim,
            dropout=cfg.CLIP.projection_dropout
        )
        
        # Store pre-computed text embeddings (frozen during training)
        if text_embeddings is not None:
            self.register_buffer('text_embeddings', text_embeddings)
        else:
            self.text_embeddings = None

        # block configuration (same as vanilla FACT)
        block_list = []
        for i, t in enumerate(cfg.FACT.block):
            if t == 'i':
                block = InputBlock(cfg, in_dim, n_classes)
            elif t == 'u':
                update_from(cfg.Bu, base_cfg, inplace=True)
                base_cfg = cfg.Bu
                block = UpdateBlock(cfg, n_classes)
            elif t == 'U':
                update_from(cfg.BU, base_cfg, inplace=True)
                base_cfg = cfg.BU
                block = UpdateBlockTDU(cfg, n_classes)

            block_list.append(block)

        self.block_list = nn.ModuleList(block_list)

        self.mcriterion = None

    def _forward_one_video(self, seq, transcript=None):
        # Same as vanilla FACT forward
        frame_feature = seq
        frame_pe = self.frame_pe(seq)
        if self.cfg.FACT.cmr:
            frame_feature = frame_feature.permute([1, 2, 0])
            frame_feature = self.channel_masking_dropout(frame_feature)
            frame_feature = frame_feature.permute([2, 0, 1])

        if self.cfg.TM.use and self.training:
            frame_feature = time_mask(frame_feature,
                        self.cfg.TM.t, self.cfg.TM.m, self.cfg.TM.p,
                        replace_with_zero=True)

        # prepare action feature
        if not self.cfg.FACT.trans:
            action_pe = self.action_query
            action_feature = torch.zeros_like(action_pe)
        else:
            action_pe = self.action_pe(transcript)
            action_feature = self.action_embed(transcript).unsqueeze(1)
            action_feature = action_feature + action_pe
            action_pe = torch.zeros_like(action_pe)

        # forward through blocks
        block_output = []
        for i, block in enumerate(self.block_list):
            frame_feature, action_feature = block(frame_feature, action_feature, frame_pe, action_pe)
            block_output.append([frame_feature, action_feature])
        
        # Project FRAME features from last block to CLIP space for contrastive loss
        # We want to align the learned frame representation with the text embedding of its ground truth action
        # frame_feature from last block: (T, B, feature_dim) - note: process_feature() was called in block, 
        # but block returns feature WITHOUT logits (concatenated with probs, wait...)
        
        # Let's verify what `block()` returns.
        # block.process_feature returns (feature, clogit). 
        # where feature = cat([feature_no_logit, cprob], dim=-1)
        # So frame_feature coming out of block contains probabilities!
        
        # We want the feature BEFORE the final classifier/softmax, but here we have them combined.
        # We need to slice off the probabilities to get the raw feature vector.
        
        # frame_feature shape: (T, B, f_dim - n_classes + n_classes) = (T, B, f_dim)
        # The last n_classes are probabilities.
        # The first (f_dim - n_classes) are the features.
        
        feat_dim = frame_feature.shape[-1] - self.num_classes
        frame_raw_features = frame_feature[:, :, :feat_dim] # (T, B, raw_feat_dim)
        
        self.projected_frame_embeddings = self.frame_projection(frame_raw_features)  # (T, B, 512)
        
        # Debug: Print shapes on first forward pass (only during training)
        if self.training and not hasattr(self, '_debug_printed'):
            print("\n" + "="*80)
            print("FACT_CLIP Debug Info (First Forward Pass)")
            print("="*80)
            print(f"Frame feature (with probs) shape: {frame_feature.shape}")
            print(f"Frame raw feature shape: {frame_raw_features.shape}")
            print(f"Projected frame embeddings shape: {self.projected_frame_embeddings.shape}")
            if self.text_embeddings is not None:
                print(f"Text embeddings shape: {self.text_embeddings.shape}")
            print("="*80 + "\n")
            self._debug_printed = True
        
        return block_output

    def _loss_one_video(self, label):
        mcriterion: MatchCriterion = self.mcriterion
        mcriterion.set_label(label)

        block: Block = self.block_list[-1]
        cprob = basic.logit2prob(block.action_clogit, dim=-1)
        match = mcriterion.match(cprob, block.a2f_attn)

        # Compute original FACT loss
        loss_list = []
        for block in self.block_list:
            loss = block.compute_loss(mcriterion, match)
            loss_list.append(loss)

        self.loss_list = loss_list
        fact_loss = sum(loss_list) / len(loss_list)
        
        # Compute contrastive loss if text embeddings are available
        # Aligns FRAME features with CLIP text embeddings for zero-shot generalization
        contrastive_loss = None
        if self.text_embeddings is not None and hasattr(self, 'projected_frame_embeddings'):
            from .loss import infonce_contrastive_loss
            
            # Mask out holdout classes from contrastive loss to avoid "negative suppression"
            # If unseen classes are included as negatives in the denominator, the model is explicitly 
            # trained to push frame features AWAY from them, effectively un-learning the semantic connection.
            
            train_text_embeddings = self.text_embeddings
            train_labels = mcriterion.class_label
            
            # Check if we have holdout classes defined in config
            if hasattr(self.cfg, 'holdout_classes') and self.cfg.holdout_classes:
                holdout_set = set(self.cfg.holdout_classes)
                n_total = self.text_embeddings.shape[0]
                
                # Identify seen classes (indices)
                seen_indices = [i for i in range(n_total) if i not in holdout_set]
                seen_indices_tensor = torch.tensor(seen_indices, device=self.text_embeddings.device)
                
                # Filter text embeddings to only SEEN classes
                train_text_embeddings = self.text_embeddings[seen_indices_tensor]
                
                # Remap ground truth labels from global indices to seen-subset indices
                # Create a mapping table: global_idx -> seen_subset_idx (or -1 if holdout)
                label_mapper = torch.full((n_total,), -1, device=self.text_embeddings.device, dtype=torch.long)
                label_mapper[seen_indices_tensor] = torch.arange(len(seen_indices), device=self.text_embeddings.device)
                
                # Apply mapping
                train_labels = label_mapper[mcriterion.class_label]
                
                # Safety check: ensure no -1 labels (which would mean a holdout class leaked into training data)
                if (train_labels == -1).any():
                    # This can happen if the dataset loader didn't perfectly filter holdout videos
                    # We should just mask these frames out of the loss
                    valid_mask = train_labels != -1
                    if valid_mask.sum() > 0:
                         # Select only valid frames for loss computation
                        train_labels = train_labels[valid_mask]
                        # We also need to filter the projected embeddings to match these frames
                        # projected_frame_embeddings is (T, B, 512). B=1 usually.
                        # valid_mask is (T,) or (T*B,)
                        
                        # Flatten embeddings to match mask
                        flat_embeddings = self.projected_frame_embeddings.view(-1, 512)
                        filtered_embeddings = flat_embeddings[valid_mask].unsqueeze(1) # Reshape back to (T_new, 1, 512) for consistency
                        
                        # Update embeddings for loss call
                        embeddings_for_loss = filtered_embeddings
                    else:
                        # No valid frames in this batch? Return 0 loss
                        return fact_loss
                else:
                    embeddings_for_loss = self.projected_frame_embeddings
            else:
                embeddings_for_loss = self.projected_frame_embeddings

            # Use InfoNCE loss to align frame features with their ground truth class text embeddings
            # mcriterion.class_label contains the frame-wise ground truth labels
            contrastive_loss = infonce_contrastive_loss(
                embeddings_for_loss,   # (T, B, 512) - frame features in CLIP space
                train_text_embeddings, # (n_seen, 512) - ONLY SEEN classes
                train_labels,          # (T,) - remapped labels
                temperature=self.cfg.CLIP.temp
            )
            
            # Combine losses with configurable weights
            fact_weight = self.cfg.CLIP.fact_loss_weight
            contrastive_weight = self.cfg.CLIP.contrastive_weight
            total_loss = fact_weight * fact_loss + contrastive_weight * contrastive_loss
            
            # Debug: Print loss values on first training iteration
            if not hasattr(self, '_loss_debug_printed'):
                print("\n" + "="*80)
                print("FACT_CLIP Loss Debug Info (First Iteration)")
                print("="*80)
                print(f"FACT loss: {fact_loss.item():.4f}")
                print(f"Contrastive loss: {contrastive_loss.item():.4f}")
                print(f"Combined loss (w_fact={fact_weight}, w_cont={contrastive_weight}): {total_loss.item():.4f}")
                print(f"Frame labels shape: {mcriterion.class_label.shape}")
                print(f"Total text embeddings available: {self.text_embeddings.shape[0]}")
                print("="*80 + "\n")
                self._loss_debug_printed = True
            
            # Store individual losses for logging
            self.fact_loss = fact_loss
            self.contrastive_loss = contrastive_loss
            return total_loss
        else:
            # Fallback to FACT loss only if text embeddings not available
            return fact_loss

    def eval_with_clip(self, transcript=None):
        """
        Zero-shot evaluation using CLIP text embeddings.
        Compares projected FRAME embeddings with ALL text embeddings via cosine similarity.
        
        This is THE KEY to zero-shot learning:
        - During training: Model learns to align frame features with seen-class text embeddings
        - During testing: Unseen-class frame features can match with unseen-class text embeddings
          because both live in CLIP's semantic space!
        
        Returns:
            pred: (T,) - frame-wise class predictions (can include unseen classes!)
        """
        if self.text_embeddings is None or not hasattr(self, 'projected_frame_embeddings'):
            # Fallback to vanilla FACT if CLIP not available
            return self.block_list[-1].eval(transcript)
        
        # Get projected frame embeddings: (T, B, 512) - already normalized
        frame_emb = self.projected_frame_embeddings
        text_emb = self.text_embeddings  # (n_classes, 512) - includes ALL classes (seen + unseen)
        
        # Get vanilla FACT components (still used for ensemble/weighted prediction)
        last_block = self.block_list[-1]
        frame_clogit = last_block.frame_clogit
        a2f_attn = last_block.a2f_attn
        
        # Frame branch prediction (vanilla FACT - uses learned logits)
        fbranch_prob_fact = torch.softmax(frame_clogit.squeeze(1), dim=-1)  # (T, n_classes)
        
        # Frame prediction via CLIP similarity (ZERO-SHOT capable!)
        frame_emb_2d = frame_emb.squeeze(1)  # (T, 512)
        
        # KEY STEP: Compute similarity between frame embeddings and ALL text embeddings
        # This includes unseen classes! Cosine similarity = dot product (both normalized)
        clip_similarity = torch.matmul(frame_emb_2d, text_emb.t())  # (T, n_classes)
        clip_similarity = clip_similarity / self.cfg.CLIP.temp  # Temperature scaling
        
        # Convert similarity to probabilities
        fbranch_prob_clip = torch.softmax(clip_similarity, dim=-1)  # (T, n_classes)
        
        # Combine CLIP-based probability with vanilla FACT probability
        # For zero-shot settings, we might want to rely more on CLIP for unseen classes
        # But `weight` (mwt) typically balances frame vs action branch.
        # Here we are enhancing the FRAME branch itself.
        
        # Current logic: Replace the "action branch" part of the ensemble with the CLIP prediction?
        # Or just ensemble CLIP prediction with FACT prediction?
        
        # Original eval uses:
        # prob = (1-weight) * abranch_prob + weight * fbranch_prob
        
        # Let's try to treat the CLIP prediction as an alternative "frame branch" prediction.
        # Or maybe we should just use the CLIP prediction alone if we want pure zero-shot?
        # But we still want to use the temporal modeling power of FACT.
        
        # Let's mix them for now, or maybe just return CLIP prediction to test efficacy.
        # But the user wants "compare via cosine similarity at test time".
        
        # Let's ensemble:
        # Final Prob = (1 - alpha) * FACT_prob + alpha * CLIP_prob
        # Using mwt as alpha for now? No, mwt is for action vs frame.
        
        # Let's assume we want to use CLIP for the final decision primarily if we trust the alignment.
        # But FACT's action branch (temporal consistency) is also important.
        
        # Let's compute the "Action Branch" part from FACT as usual.
        action_clogit = last_block.action_clogit.squeeze(1)
        a2f_attn = a2f_attn.squeeze(0)
        
        # ... (standard FACT action branch logic)
        qtk_cpred = action_clogit.argmax(1) 
        null_cid = action_clogit.shape[-1] - 1
        action_loc = torch.where(qtk_cpred!=null_cid)[0]

        if len(action_loc) == 0:
             # If no action detected by FACT, fallback to CLIP frame prediction
            return fbranch_prob_clip.argmax(1)

        qtk_prob = torch.softmax(action_clogit[:, :-1], dim=1) # remove logit of null classes
        action_pred = a2f_attn[:, action_loc].argmax(-1)
        action_pred = action_loc[action_pred]
        abranch_prob = qtk_prob[action_pred] # (T, n_classes)

        # So we have:
        # 1. abranch_prob (from FACT action tokens)
        # 2. fbranch_prob_fact (from FACT frame classifier)
        # 3. fbranch_prob_clip (from CLIP alignment)
        
        # Let's fuse them.
        # weight (mwt) usually gives low weight to frame branch (e.g. 0.1 or 0.2) and high to action branch.
        # We can replace fbranch_prob_fact with fbranch_prob_clip, or average them.
        
        # EXPERIMENTAL: Use pure CLIP probability for frame branch to test zero-shot capability
        # fbranch_prob_final = (fbranch_prob_fact + fbranch_prob_clip) / 2
        fbranch_prob_final = fbranch_prob_clip
        
        weight = self.cfg.FACT.mwt
        prob = (1 - weight) * abranch_prob + weight * fbranch_prob_final
        
        return prob.argmax(1)

    def forward(self, seq_list, label_list, compute_loss=False):
        save_list = []
        final_loss = []

        for i, (seq, label) in enumerate(zip(seq_list, label_list)):
            seq = seq.unsqueeze(1)
            trans = torch_class_label_to_segment_label(label)[0]
            self._forward_one_video(seq, trans)

            # Use CLIP-based evaluation for zero-shot capability!
            pred = self.eval_with_clip(trans)
            save_data = {'pred': utils.to_numpy(pred)}
            save_list.append(save_data)

            if compute_loss:
                loss = self._loss_one_video(label)
                final_loss.append(loss)
                loss_dict = {'loss': loss.item()}
                if hasattr(self, 'fact_loss'):
                    loss_dict['fact_loss'] = self.fact_loss.item()
                if hasattr(self, 'contrastive_loss'):
                    loss_dict['contrastive_loss'] = self.contrastive_loss.item()
                save_data['loss'] = loss_dict

        if compute_loss:
            final_loss = sum(final_loss) / len(final_loss)
            return final_loss, save_list
        else:
            return save_list

    def save_model(self, fname):
        torch.save(self.state_dict(), fname)




