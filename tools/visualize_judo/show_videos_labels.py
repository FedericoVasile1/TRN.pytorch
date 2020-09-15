import os
import sys
import argparse
import numpy as np

sys.path.append(os.getcwd())
import _init_paths
from lib.utils.visualize import show_random_videos
from configs.build import build_data_info

if __name__ == '__main__':
    base_dir = os.getcwd()
    base_dir = base_dir.split('/')[-1]
    CORRECT_LAUNCH_DIR = 'TRN.pytorch'
    if base_dir != CORRECT_LAUNCH_DIR:
        raise Exception('Wrong base dir, this file must be run from ' + CORRECT_LAUNCH_DIR + ' directory.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/JUDO', type=str)
    parser.add_argument('--data_info', default='data/data_info.json', type=str)
    parser.add_argument('--frames_dir', default='video_frames_25fps', type=str)
    parser.add_argument('--targets_dir', default='target_frames_25fps', type=str)
    # the fps at which videos frames are previously extracted
    parser.add_argument('--fps', default=25, type=int)
    parser.add_argument('--chunk_size', default=6, type=int)
    parser.add_argument('--phase', default='train', type=str)
    parser.add_argument('--video_name', default='', type=str)
    parser.add_argument('--samples', default=2, type=int)
    parser.add_argument('--seed', default=-1, type=int)
    args = parser.parse_args()

    if not os.path.isdir(os.path.join(args.data_root)):
        raise Exception('{} not found'.format(os.path.join(args.data_root)))
    if not os.path.isfile(os.path.join(args.data_info)):
        raise Exception('{} not found'.format(os.path.join(args.data_info)))
    if not os.path.isdir(os.path.join(args.data_root, args.frames_dir)):
        raise Exception('{} not found'.format(os.path.join(args.data_root, args.frames_dir)))
    if not os.path.isdir(os.path.join(args.data_root, args.targets_dir)):
        raise Exception('{} not found'.format(os.path.join(args.data_root, args.targets_dir)))
    if args.phase not in ('train', 'val', 'test'):
        raise Exception('Wrong --phase argument. Expected one of: train|val|test')
    if str(args.fps) not in args.frames_dir:
        raise Exception('The folder {} should contain the number of fps in its name, or the number '
                        'indicated in its name does not correspond with --fps argument(i.e. they must be '
                        'the same)'.format(args.frames_dir))

    if args.seed != -1:
        np.random.seed(args.seed)

    args = build_data_info(args, basic_build=True)

    if args.video_name == '':
        videos_list = getattr(args, args.phase+'_session_set')
    else:
        videos_list = [args.video_name]

    show_random_videos(args,
                       videos_list,
                       samples=args.samples if len(videos_list) > args.samples else len(videos_list),
                       frames_dir=args.frames_dir,
                       targets_dir=args.targets_dir,
                       fps=args.fps)
