import os
import sys
import argparse
import numpy as np

from torchvision import transforms

sys.path.append(os.getcwd())
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
    parser.add_argument('--frames_dir', default='candidates_video_frames_25fps', type=str)
    parser.add_argument('--targets_dir', default='candidates_4s_target_frames_25fps', type=str)
    # the fps at which videos frames are previously extracted
    parser.add_argument('--fps', default=25, type=int)
    parser.add_argument('--chunk_size', default=1, type=int)
    parser.add_argument('--phase', default='train', type=str)
    parser.add_argument('--video_name', default='', type=str)
    parser.add_argument('--save_video', default=True, action='store_false')
    parser.add_argument('--seed', default=-1, type=int)
    args = parser.parse_args()

    if not os.path.isdir(os.path.join(args.data_root)):
        raise Exception('{} not found'.format(os.path.join(args.data_root)))
    if not os.path.isfile(os.path.join(args.data_info)):
        raise Exception('{} not found'.format(os.path.join(args.data_info)))
    if args.phase not in ('train', 'val', 'test'):
        raise Exception('Wrong --phase argument. Expected one of: train|val|test')
    if str(args.fps) not in args.frames_dir:
        raise Exception('The folder {} should contain the number of fps in its name, or the number '
                        'indicated in its name does not correspond with --fps argument(i.e. they must be '
                        'the same)'.format(args.frames_dir))

    if args.seed != -1:
        np.random.seed(args.seed)

    # do not modify
    args.use_untrimmed = True
    args.use_trimmed = False
    args.eval_on_untrimmed = False
    args.use_candidates = True
    args = build_data_info(args, basic_build=True)

    args.data_root = args.data_root + '/' + 'UNTRIMMED'
    args.train_session_set = args.train_session_set['UNTRIMMED']
    args.val_session_set = args.val_session_set['UNTRIMMED']
    args.test_session_set = args.test_session_set['UNTRIMMED']

    if not os.path.isdir(os.path.join(args.data_root, args.frames_dir)):
        raise Exception('{} not found'.format(os.path.join(args.data_root, args.frames_dir)))
    if not os.path.isdir(os.path.join(args.data_root, args.targets_dir)):
        raise Exception('{} not found'.format(os.path.join(args.data_root, args.targets_dir)))

    if args.video_name == '':
        videos_list = getattr(args, args.phase+'_session_set')
        idx = np.random.choice(len(videos_list), size=1, replace=False)[0]
        sample = videos_list[idx]
        valid_videos = os.listdir(os.path.join(args.data_root, args.targets_dir))
        valid_videos = [v for v in valid_videos if sample in v]
        idx = np.random.choice(len(valid_videos), size=1, replace=False)[0]
        videos_list = [valid_videos[idx][:-4]]
    else:
        videos_list = [args.video_name]

    '''
    transform = transforms.Compose([
        transforms.Resize((224, 320)),
        transforms.CenterCrop(224),
    ])
    '''
    transform = None

    show_random_videos(args,
                       videos_list,
                       samples=1,
                       frames_dir=args.frames_dir,
                       targets_dir=args.targets_dir,
                       fps=args.fps,
                       transform=transform)
