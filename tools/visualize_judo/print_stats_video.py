import os
import sys
import argparse

import numpy as np

sys.path.append(os.getcwd())
import _init_paths
from configs.build import build_data_info
from lib.utils.visualize import print_stats_video

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
    parser.add_argument('--all_videos', action='store_true')
    parser.add_argument('--seed', default=-1, type=int) # in case of single video
                                                        # visualization(i.e. --all_videos==False) this is the seed
                                                        # by means we sample the video
    parser.add_argument('--video_name', default='', type=str)   # in order to visualize a particular video
    parser.add_argument('--fps', default=25, type=int)
    args = parser.parse_args()

    if not os.path.isdir(os.path.join(args.data_root)):
        raise Exception('{} not found'.format(os.path.join(args.data_root)))
    if not os.path.isfile(os.path.join(args.data_info)):
        raise Exception('{} not found'.format(os.path.join(args.data_info)))
    if not os.path.isdir(os.path.join(args.data_root, args.target_labels_dir)):
        raise Exception('{} not found'.format(os.path.join(args.data_root, args.target_labels_dir)))
    if str(args.fps) not in args.target_labels_dir:
        raise Exception('The folder {} should contain the number of fps in its name, or the number '
                        'indicated in its name does not correspond with --fps argument(i.e. they must be '
                        'the same)'.format(args.target_labels_dir))
    if args.video_name != '' and (args.all_videos or args.seed != -1):
        raise Exception('When specifying a --video_name you do not have to specify --all_videos or --seed options')

    if args.seed != -1:
        np.random.seed(args.seed)

    args = build_data_info(args, basic_build=True)

    videos_name = os.listdir(os.path.join(args.data_root, args.target_labels_dir))
    if args.all_videos:
        class_to_segmentdurations_all = {}
        for i in range(args.num_classes):
            class_to_segmentdurations_all[args.class_index[i]] = []
        video_duration_all = []

        for video_name in videos_name:
            class_to_duration, _, video_duration = print_stats_video(video_name, args)
            for name_class, duration in class_to_duration.items():
                if duration == 0:
                    continue
                class_to_segmentdurations_all[name_class].append(duration)
            video_duration_all.append(video_duration)

        for name_class, list_durations in class_to_segmentdurations_all.items():
            if len(list_durations) > 0:
                class_to_segmentdurations_all[name_class] = str(round(sum(list_durations) / len(list_durations), 1)) + ' s'
            else:
                class_to_segmentdurations_all[name_class] = str(0) + ' s'

        video_duration_all = sum(video_duration_all) / len(video_duration_all)
        video_duration_all = round(video_duration_all, 1)

        print('=== ALL VIDEOS ===')
        print('MEAN VIDEO DURATION: ', video_duration_all, ' s')
        print('MEAN DURATION PER CLASS: ', class_to_segmentdurations_all)

    else:
        if args.video_name == '':
            num_videos = len(videos_name)
            idx_video_name = np.random.choice(num_videos, size=1, replace=False)[0]
            video_name = videos_name[idx_video_name]
        else:
            video_name = [i for i in videos_name if i[:-4] == args.video_name]
            video_name = video_name[0]
        class_to_segmentdurations, segment_list, video_duration = print_stats_video(video_name, args)
        print('VIDEO NAME: ', video_name[:-4])
        print('VIDEO DURATION: ', video_duration, ' s')
        print('MEAN DURATION PER CLASS: ', class_to_segmentdurations)
        print('VIDEO SEGMENTS: ', segment_list)