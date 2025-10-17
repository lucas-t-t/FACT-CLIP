#!/usr/bin/python3

import numpy as np
import os
import torch
from ..home import get_project_base
from yacs.config import CfgNode
from .utils import shrink_frame_label

BASE = get_project_base()

def load_feature(feature_dir, video, transpose):
    file_name = os.path.join(feature_dir, video+'.npy')
    feature = np.load(file_name)

    if transpose:
        feature = feature.T
    if feature.dtype != np.float32:
        feature = feature.astype(np.float32)
    
    return feature #[::sample_rate]

def load_action_mapping(map_fname, sep=" "):
    label2index = dict()
    index2label = dict()
    with open(map_fname, 'r') as f:
        content = f.read().split('\n')[0:-1]
        for line in content:
            tokens = line.split(sep)
            l = sep.join(tokens[1:])
            i = int(tokens[0])
            label2index[l] = i
            index2label[i] = l

    return label2index, index2label

class Dataset(object):
    """
    self.features[video]: the feature array of the given video (frames x dimension)
    self.input_dimension: dimension of video features
    self.n_classes: number of classes
    """

    def __init__(self, video_list, nclasses, load_video_func, bg_class):
        """
        """

        self.video_list = video_list
        self.load_video = load_video_func

        # store dataset information
        self.nclasses = nclasses
        self.bg_class = bg_class
        self.data = {}
        self.data[video_list[0]] = load_video_func(video_list[0])
        self.input_dimension = self.data[video_list[0]][0].shape[1] 
    
    def __str__(self):
        string = "< Dataset %d videos, %d feat-size, %d classes >"
        string = string % (len(self.video_list), self.input_dimension, self.nclasses)
        return string
    
    def __repr__(self):
        return str(self)

    def get_vnames(self):
        return self.video_list[:]

    def __getitem__(self, video):
        if video not in self.video_list:
            raise ValueError(video)

        if video not in self.data:
            self.data[video] = self.load_video(video)

        return self.data[video]

    def __len__(self):
        return len(self.video_list)


class DataLoader():

    def __init__(self, dataset: Dataset, batch_size, shuffle=False):

        self.num_video = len(dataset)
        self.dataset = dataset
        self.videos = list(dataset.get_vnames())
        self.shuffle = shuffle
        self.batch_size = batch_size

        self.num_batch = int(np.ceil(self.num_video/self.batch_size))

        self.selector = list(range(self.num_video))
        self.index = 0
        if self.shuffle:
            np.random.shuffle(self.selector)
            # self.selector = self.selector.tolist()

    def __len__(self):
        return self.num_batch

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.num_video:
            if self.shuffle:
                np.random.shuffle(self.selector)
                # self.selector = self.selector.tolist()
            self.index = 0
            raise StopIteration

        else:
            video_idx = self.selector[self.index : self.index+self.batch_size]
            if len(video_idx) < self.batch_size:
                video_idx = video_idx + self.selector[:self.batch_size-len(video_idx)]
            videos = [self.videos[i] for i in video_idx]
            self.index += self.batch_size

            batch_sequence = []
            batch_train_label = []
            batch_eval_label = []
            for vname in videos:
                sequence, train_label, eval_label = self.dataset[vname]
                batch_sequence.append(torch.from_numpy(sequence))
                batch_train_label.append(torch.LongTensor(train_label))
                batch_eval_label.append(eval_label)


            return videos, batch_sequence, batch_train_label, batch_eval_label


#------------------------------------------------------------------
#------------------------------------------------------------------

def video_contains_holdout_classes(vname, groundTruth_path, label2index, holdout_classes):
    """
    Check if a video contains any holdout classes.
    
    Args:
        vname: Video name (without extension)
        groundTruth_path: Path to groundTruth directory
        label2index: Mapping from label string to class index
        holdout_classes: List of class indices to check for
        
    Returns:
        True if video contains any holdout class, False otherwise
    """
    try:
        with open(os.path.join(groundTruth_path, vname + '.txt'), 'rb') as f:
            raw_content = f.read().replace(b'\r\n', b'\n')
        try:
            content = raw_content.decode('utf-8')
        except UnicodeDecodeError:
            content = raw_content.decode('latin-1')
        
        labels = [label2index[line] for line in content.split('\n')[:-1] if line in label2index]
        
        # Check if any label is in holdout_classes
        for label in labels:
            if label in holdout_classes:
                return True
        return False
    except Exception as e:
        print(f"Warning: Could not read labels for video {vname}: {e}")
        return False

def create_dataset(cfg: CfgNode):

    if cfg.dataset == "breakfast":
        map_fname = BASE + 'data/breakfast/mapping.txt'
        dataset_path = BASE + 'data/breakfast/'
        train_split_fname = BASE + f'data/breakfast/splits/train.{cfg.split}.bundle'
        test_split_fname = BASE + f'data/breakfast/splits/test.{cfg.split}.bundle'
        feature_path = BASE + 'data/breakfast/features'
        feature_transpose = True
        average_transcript_len = 6.9 
        bg_class = [0] 

    elif cfg.dataset == "gtea":
        map_fname = BASE + 'data/gtea/mapping.txt'
        dataset_path = BASE + 'data/gtea/'
        feature_path = BASE + 'data/gtea/features/'
        train_split_fname = BASE + f'data/gtea/splits/train.{cfg.split}.bundle'
        test_split_fname = BASE + f'data/gtea/splits/test.{cfg.split}.bundle'
        feature_transpose = True
        average_transcript_len = 32.9
        bg_class = [10]

    elif cfg.dataset == "ego":
        map_fname = BASE + 'data/egoprocel/mapping.txt'
        dataset_path = BASE + 'data/egoprocel/'
        feature_path = BASE + 'data/egoprocel/features/'
        train_split_fname = BASE + 'data/egoprocel/%s.train' % cfg.split
        test_split_fname = BASE + 'data/egoprocel/%s.test' % cfg.split
        feature_transpose = False
        bg_class = [0]
        if cfg.Loss.match == 'o2o':
            average_transcript_len = 21.5
        else: # for one-to-many matching
            average_transcript_len = 7.4

    elif cfg.dataset == "epic":
        map_fname = BASE + 'data/epic-kitchens/processed/mapping.txt'
        dataset_path = BASE + 'data/epic-kitchens/processed/'
        bg_class = [0]
        feature_path = BASE + 'data/epic-kitchens/processed/features'
        train_split_fname = BASE + 'data/epic-kitchens/processed/%s.train' % cfg.split
        test_split_fname = BASE + 'data/epic-kitchens/processed/%s.test' % cfg.split
        feature_transpose = False
        if cfg.Loss.match == 'o2o':
            average_transcript_len = 165
        else:
            average_transcript_len = 52
    
    elif cfg.dataset.startswith("havid"):
        # HAViD dataset variants (e.g., "havid_view0_lh_pt", "havid_view1_rh_aa")
        variant = cfg.dataset.replace("havid_", "")  # e.g., "view0_lh_pt"
        havid_base = BASE + 'data/HAViD/ActionSegmentation/data'
        
        map_fname = f'{havid_base}/{variant}/mapping.txt'
        dataset_path = f'{havid_base}/{variant}/'
        feature_path = f'{havid_base}/features'
        train_split_fname = f'{havid_base}/{variant}/splits/train.{cfg.split}.bundle'
        test_split_fname = f'{havid_base}/{variant}/splits/test.{cfg.split}.bundle'
        
        feature_transpose = True  # HAViD features are (D, T), need (T, D)
        bg_class = [0]
        
        # Set average_transcript_len based on annotation type
        if variant.endswith('_pt'):  # primitive tasks
            average_transcript_len = 8.0  # approximate from our analysis
        elif variant.endswith('_aa'):  # atomic actions
            average_transcript_len = 15.0  # more granular, so more segments
        else:
            average_transcript_len = 10.0  # default
    
    groundTruth_path = os.path.join(dataset_path, 'groundTruth')

    ################################################
    ################################################
    print("Loading Feature from", feature_path)
    print("Loading Label from", groundTruth_path)

    label2index, index2label = load_action_mapping(map_fname)
    nclasses = len(label2index)

    """
    load video interface:
        Input: video name
        Output:
            feature, label_for_training, label_for_evaluation
    """
    def load_video(vname):
        feature = load_feature(feature_path, vname, feature_transpose) # should be T x D or T x D x H x W

        with open(os.path.join(groundTruth_path, vname + '.txt')) as f:
            gt_label = [ label2index[line] for line in f.read().split('\n')[:-1] ]


        if feature.shape[0] != len(gt_label):
            l = min(feature.shape[0], len(gt_label))
            feature = feature[:l]
            gt_label = gt_label[:l]

        # downsample if necessary
        sr = cfg.sr
        if sr > 1:
            feature = feature[::sr]
            gt_label_sampled = shrink_frame_label(gt_label, sr)
        else:
            gt_label_sampled = gt_label

        return feature, gt_label_sampled, gt_label

    
    ################################################
    ################################################
    
    with open(test_split_fname, 'r') as f:
        test_video_list = f.read().split('\n')[0:-1]
    if cfg.dataset in ['breakfast', '50salads', 'gtea']: 
        test_video_list = [ v[:-4] for v in test_video_list ] 
    elif cfg.dataset.startswith('havid'):
        test_video_list = [ v[:-4] for v in test_video_list if v.endswith('.txt') ] 
    test_dataset = Dataset(test_video_list, nclasses, load_video, bg_class)

    if cfg.aux.debug:
        dataset = test_dataset
    else:
        with open(train_split_fname, 'r') as f:
            video_list = f.read().split('\n')[0:-1]
        if cfg.dataset in ['breakfast', '50salads', 'gtea']: 
            video_list = [ v[:-4] for v in video_list ] 
        elif cfg.dataset.startswith('havid'):
            video_list = [ v[:-4] for v in video_list if v.endswith('.txt') ]
        
        # Apply holdout filtering if enabled
        if cfg.holdout_mode and len(cfg.holdout_classes) > 0:
            original_count = len(video_list)
            holdout_classes = list(cfg.holdout_classes)
            
            print(f"\n{'='*80}")
            print(f"HOLDOUT MODE ENABLED")
            print(f"{'='*80}")
            print(f"Holdout classes: {holdout_classes}")
            print(f"Holdout class names: {[index2label[c] for c in holdout_classes if c in index2label]}")
            print(f"Original training videos: {original_count}")
            
            # Filter out videos containing any holdout class
            filtered_video_list = []
            removed_videos = []
            for vname in video_list:
                if video_contains_holdout_classes(vname, groundTruth_path, label2index, holdout_classes):
                    removed_videos.append(vname)
                else:
                    filtered_video_list.append(vname)
            
            video_list = filtered_video_list
            print(f"Videos removed (contain holdout classes): {len(removed_videos)}")
            print(f"Remaining training videos: {len(video_list)} ({100*len(video_list)/original_count:.1f}%)")
            print(f"{'='*80}\n")
            
            if len(video_list) == 0:
                raise ValueError("No training videos remaining after holdout filtering!")
        
        dataset = Dataset(video_list, nclasses, load_video, bg_class)
        
    dataset.average_transcript_len = average_transcript_len
    dataset.label2index = label2index
    dataset.index2label = index2label
    test_dataset.average_transcript_len = average_transcript_len
    test_dataset.label2index = label2index
    test_dataset.index2label = index2label
    
    # Add holdout information for evaluation
    if cfg.holdout_mode and len(cfg.holdout_classes) > 0:
        holdout_classes_list = list(cfg.holdout_classes)
        seen_classes = [c for c in range(nclasses) if c not in holdout_classes_list]
        dataset.holdout_classes = holdout_classes_list
        dataset.seen_classes = seen_classes
        test_dataset.holdout_classes = holdout_classes_list
        test_dataset.seen_classes = seen_classes
    else:
        dataset.holdout_classes = []
        dataset.seen_classes = list(range(nclasses))
        test_dataset.holdout_classes = []
        test_dataset.seen_classes = list(range(nclasses))

    return dataset, test_dataset
