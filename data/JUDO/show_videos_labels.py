import os
import sys

sys.path.append(os.getcwd())
import _init_paths
from configs.thumos import parse_trn_args as parse_args
from lib.utils.net_utils import show_random_videos

if __name__ == '__main__':
    base_dir = os.getcwd()
    base_dir = base_dir.split('/')[-1]
    if base_dir != 'TRN.pytorch':
        raise Exception('Wrong base dir, this file must be run from TRN.pytorch/ directory.')

    args = parse_args()
    show_random_videos(args,
                       ['video_validation_0000690', 'video_validation_0000690'],
                       samples=1)
