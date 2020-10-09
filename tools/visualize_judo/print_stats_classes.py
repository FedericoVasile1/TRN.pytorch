import os
import sys
import argparse

sys.path.append(os.getcwd())
import _init_paths
from configs.build import build_data_info
from lib.utils.visualize import print_stats_classes

if __name__ == '__main__':
    base_dir = os.getcwd()
    base_dir = base_dir.split('/')[-1]
    CORRECT_LAUNCH_DIR = 'TRN.pytorch'
    if base_dir != CORRECT_LAUNCH_DIR:
        raise Exception('Wrong base dir, this file must be run from ' + CORRECT_LAUNCH_DIR + ' directory.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/JUDO', type=str)
    parser.add_argument('--data_info', default='data/data_info.json', type=str)
    parser.add_argument('--chunk_size', default=6, type=int)
    parser.add_argument('--target_labels_dir', default='target_frames_25fps', type=str)
    parser.add_argument('--phase', default='', type=str)
    parser.add_argument('--downsample_backgr', action='store_true')
    parser.add_argument('--enc_steps', default=16, type=int)        # needed only when --downsample_backgr == True
    parser.add_argument('--show_bar', action='store_true')
    parser.add_argument('--save_bar', action='store_true')
    args = parser.parse_args()

    if not os.path.isdir(os.path.join(args.data_root)):
        raise Exception('{} not found'.format(os.path.join(args.data_root)))
    if not os.path.isfile(os.path.join(args.data_info)):
        raise Exception('{} not found'.format(os.path.join(args.data_info)))
    if not os.path.isdir(os.path.join(args.data_root, args.target_labels_dir)):
        raise Exception('{} not found'.format(os.path.join(args.data_root, args.target_labels_dir)))
    if args.phase not in ('train', 'val', 'test', ''):
        raise Exception('Wrong --phase argument. Expected one of: train|val|test')

    print_stats_classes(build_data_info(args, basic_build=True))