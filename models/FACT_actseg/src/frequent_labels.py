import os
from collections import Counter
base = '/cvhci/temp/lthomaz/models/FACT_actseg/data/HAViD/ActionSegmentation/data/view0_lh_pt'
map_path = os.path.join(base, 'mapping.txt')
label2index = {}
index2label = {}
with open(map_path, encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        idx, label = line.strip().split(' ', 1)
        idx = int(idx)
        label2index[label] = idx
        index2label[idx] = label

def read_labels(v):
    path = os.path.join(base, 'groundTruth', v + '.txt')
    with open(path, 'rb') as f:
        raw = f.read().replace(b'\r\n', b'\n')
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError:
        text = raw.decode('latin-1')
    labels = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
                       continue
        if line not in label2index:
            continue
        labels.append(label2index[line])
    return labels

train_split = os.path.join(base, 'splits', 'train.split1.bundle')
test_split = os.path.join(base, 'splits', 'test.split1.bundle')

with open(train_split) as f:
    train_videos = [ln.strip() for ln in f if ln.strip()]
train_videos = [v[:-4] if v.endswith('.txt') else v for v in train_videos]
with open(test_split) as f:
    test_videos = [ln.strip() for ln in f if ln.strip()]
test_videos = [v[:-4] if v.endswith('.txt') else v for v in test_videos]

train_counts = Counter()
for v in train_videos:
    train_counts.update(read_labels(v))

def print_stats(label_list, title):
    total_frames = sum(train_counts[idx] for idx in label_list)
    total_classes = len(label_list)
    print(f'Option {title}: {total_classes} classes, {total_frames} frames ({total_frames/ sum(train_counts.values()):.2%} of train frames)')
    for idx in label_list:
        print(f'  {idx:02d} {index2label[idx]:10s} frames={train_counts[idx]} videos=')

rare_candidates = [idx for idx, count in train_counts.most_common() if idx != 27][-15:]
common_candidates = [idx for idx, count in train_counts.most_common(15) if idx != 27]
mixed_candidates = rare_candidates[:7] + common_candidates[:8]

print('Top 15 common (excluding background):')
for idx in common_candidates:
    print(idx, index2label[idx], train_counts[idx])
print('\nBottom 15 rare (excluding background):')
for idx in rare_candidates:
    print(idx, index2label[idx], train_counts[idx])