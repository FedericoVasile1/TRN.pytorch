import os
import sys
import argparse

sys.path.append(os.getcwd())
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
    parser.add_argument('--use_trimmed', action='store_true')
    parser.add_argument('--use_untrimmed', action='store_true')
    parser.add_argument('--data_info', default='data/data_info.json', type=str)
    parser.add_argument('--chunk_size', default=9, type=int)
    parser.add_argument('--target_labels_dir', default='target_frames_25fps', type=str)
    parser.add_argument('--phase', default='', type=str)
    parser.add_argument('--downsample_backgr', action='store_true')
    parser.add_argument('--steps', default=16, type=int)        # needed only when --downsample_backgr == True
    parser.add_argument('--show_bar', action='store_true')
    parser.add_argument('--save_bar', action='store_true')
    args = parser.parse_args()

    if not os.path.isdir(os.path.join(args.data_root)):
        raise Exception('{} not found'.format(os.path.join(args.data_root)))
    if not os.path.isfile(os.path.join(args.data_info)):
        raise Exception('{} not found'.format(os.path.join(args.data_info)))
    if args.phase not in ('train', 'val', 'test', ''):
        raise Exception('Wrong --phase argument. Expected one of: train|val|test')
    if args.use_trimmed == args.use_untrimmed == True:
        raise Exception('Wrong --use_trimmed and --use_untrimmed option. Here they can not be True together.')

    args.eval_on_untrimmed = False      # do not modify. will not be used, it is only to not generate errors
    args = build_data_info(args, basic_build=True)
    if args.use_trimmed:
        args.data_root = args.data_root + '/' + 'TRIMMED'
        args.train_session_set = args.train_session_set['TRIMMED']
        args.val_session_set = args.val_session_set['TRIMMED']
        args.test_session_set = args.test_session_set['TRIMMED']
    elif args.use_untrimmed:
        args.data_root = args.data_root + '/' + 'UNTRIMMED'
        args.train_session_set = args.train_session_set['UNTRIMMED']
        args.val_session_set = args.val_session_set['UNTRIMMED']
        args.test_session_set = args.test_session_set['UNTRIMMED']
    else:
        raise Exception('No dataset type specified.')

    if not os.path.isdir(os.path.join(args.data_root, args.target_labels_dir)):
        raise Exception('{} not found'.format(os.path.join(args.data_root, args.target_labels_dir)))

    print_stats_classes(args)