import os
import sys

import numpy as np

sys.path.append(os.getcwd())
import _init_paths
from configs.judo import parse_trn_args as parse_args

if __name__ == '__main__':
    base_dir = os.getcwd()
    base_dir = base_dir.split('/')[-1]
    if base_dir != 'TRN.pytorch':
        raise Exception('Wrong base dir, this file must be run from TRN.pytorch/ directory.')

    args = parse_args()

    class_to_count = {}
    for i in range(args.num_classes):
        class_to_count[i] = 0

    TARGETS_BASE_DIR = os.path.join(args.data_root, 'target_frames_25fps')
    tot_samples = 0
    for video_name in os.listdir(TARGETS_BASE_DIR):
        if '.npy' not in video_name:
            continue

        target = np.load(os.path.join(TARGETS_BASE_DIR, video_name))
        num_frames = target.shape[0]
        num_frames = num_frames - (num_frames % args.chunk_size)
        target = target[:num_frames]
        # For each chunk, take only the central frame
        target = target[args.chunk_size // 2::args.chunk_size]

        target = target.argmax(axis=1)
        unique, counts = np.unique(target, return_counts=True)

        tot_samples += counts.sum()

        for idx, idx_class in enumerate(unique):
            class_to_count[idx_class] += counts[idx]

    for idx_class, count in class_to_count.items():
        class_name = args.class_index[idx_class]
        print('{:12s}:  {:.1f} %'.format(class_name, count / tot_samples * 100))