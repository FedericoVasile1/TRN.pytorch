import os
import csv
import sys
import numpy as np
import argparse

sys.path.append(os.getcwd())
from configs.build import build_data_info

def main(args):
    CLASS_INDEX = {}
    for i, class_name in enumerate(args.class_index):
        CLASS_INDEX[class_name] = i

    is_first_row = True
    with open(os.path.join(args.data_root, args.labels_file), encoding='utf16') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if is_first_row:
                is_first_row = False
                continue
            if int(row['fall_frame ']) == -1:
                continue

            video_name = row['video_name']
            num_frames_video = len(os.listdir(args.data_root, args.extracted_frames_dir, video_name))
            all_background_labels = np.zeros((num_frames_video, args.num_classes))
            all_background_labels[:, CLASS_INDEX['Background']] = 1
            np.save(os.path.join(args.data_root, args.target_labels_dir, video_name+'.npy'), all_background_labels)

    START_ACTION = args.fps
    END_ACTION = args.fps - 1
    with open(os.path.join(args.data_root, args.labels_file), encoding='utf16') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        is_first_row = True
        for row in csv_reader:
            if is_first_row:
                is_first_row = False
                continue
            if int(row['fall_frame']) == -1:
                continue

            video_name = row['video_name']
            class_idx = CLASS_INDEX[row['class_name']]
            fall_frame = int(row['fall_frame'])
            # fall frame represents the central frame of the action, and we put the actions to 2 seconds long
            start_frame = fall_frame - START_ACTION
            end_frame = fall_frame + END_ACTION
            labels = np.load(os.path.join(args.data_root, args.target_labels_dir, video_name+'.npy'))
            for i in range(start_frame, end_frame):
                labels[i-1, CLASS_INDEX['Background']] = 0
                labels[i-1, class_idx] = 1
            np.save(os.path.join(args.data_root, args.targets_labels_dir, video_name+'.npy'), labels)

if __name__ == '__main__':
    base_dir = os.getcwd()
    base_dir = base_dir.split('/')[-1]
    CORRECT_LAUNCH_DIR = 'TRN.pytorch'
    if base_dir != CORRECT_LAUNCH_DIR:
        raise Exception('Wrong base dir, this file must be run from ' + CORRECT_LAUNCH_DIR + ' directory.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/JUDO/TRIMMED', type=str)
    parser.add_argument('--data_info', default='data/data_info.json', type=str)
    parser.add_argument('--labels_file', default='fall_detections.csv', type=str)
    # the fps at which videos frames are previously extracted
    parser.add_argument('--fps', default=25, type=int)
    # the directory where frames of video has been extracted
    parser.add_argument('--extracted_frames_dir', default='video_frames_25fps', type=str)
    # the directory where labels arrays will be stored
    parser.add_argument('--target_labels_dir', default='target_frames_25fps', type=str)
    args = parser.parse_args()

    if not os.path.isdir(os.path.join(args.data_root)):
        raise Exception('{} not found'.format(os.path.join(args.data_root)))
    if not os.path.isfile(os.path.join(args.data_info)):
        raise Exception('{} not found'.format(os.path.join(args.data_info)))
    if not os.path.isfile(os.path.join(args.data_root, args.labels_file)):
        raise Exception('{} not found'.format(os.path.join(args.data_root, args.labels_file)))
    if not os.path.isdir(os.path.join(args.data_root, args.extracted_frames_dir)):
        raise Exception('{} not found'.format(os.path.join(args.data_root, args.extracted_frames_dir)))
    if str(args.fps) not in args.target_labels_dir:
        raise Exception('The folder {} should contain the number of fps in its name, or the number '
                        'indicated in its name does not correspond with --fps argument(i.e. they must be '
                        'the same)'.format(args.target_labels_dir))
    if str(args.fps) not in args.extracted_frames_dir:
        raise Exception('The folder {} should contain the number of fps in its name, or the number '
                        'indicated in its name does not correspond with --fps argument(i.e. they must be '
                        'the same)'.format(args.extracted_frames_dir))
    if not os.path.isdir(os.path.join(args.data_root, args.target_labels_dir)):
        os.makedir(os.path.join(args.data_root, args.target_labels_dir))
        print('{} folder created'.format(os.path.join(args.data_root, args.target_labels_dir)))

    main(build_data_info(args, basic_build=True))