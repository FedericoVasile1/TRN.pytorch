import os
import sys

sys.path.append(os.getcwd())
import _init_paths
from configs.judo import parse_trn_args as parse_args
from lib.utils.net_utils import show_random_videos

if __name__ == '__main__':
    base_dir = os.getcwd()
    base_dir = base_dir.split('/')[-1]
    if base_dir != 'TRN.pytorch':
        raise Exception('Wrong base dir, this file must be run from TRN.pytorch/ directory.')

    args = parse_args()
    if args.video_name == '':
        videos_list = args.train_session_set
    else:
        videos_list = [args.video_name]
    show_random_videos(args,
                       videos_list,
                       samples=args.samples if len(videos_list) > args.samples else len(videos_list),
                       frames_dir='video_frames_25fps',
                       targets_dir='target_frames_25fps',
                       fps=25)
