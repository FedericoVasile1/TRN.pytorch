import os
import numpy as np
import csv
import sys
import argparse

sys.path.append(os.getcwd())
import _init_paths
from configs.build import build_data_info

def milliseconds_to_numframe(time_milliseconds, fps=25):
    fpms = fps * 0.001
    num_frame = (time_milliseconds * fpms)
    return int(num_frame)

def main(args):
    CLASS_INDEX = {}
    for i, class_name in enumerate(args.class_index):
        CLASS_INDEX[class_name] = i

    for video_dir in os.listdir(os.path.join(args.data_root, args.extracted_frames_dir)):
        if '.mp4' not in video_dir:
            continue

        num_frames_video = len(os.listdir(os.path.join(args.data_root, args.extracted_frames_dir, video_dir)))
        all_background_labels = np.zeros((num_frames_video, args.num_classes))
        all_background_labels[:, CLASS_INDEX['Background']] = 1
        np.save(os.path.join(args.data_root, args.target_labels_dir, video_dir+'.npy'), all_background_labels)

    column_time = 1     # column B
    column_class = 37
    column_filename = 40  # column AO
    is_first_row = True
    with open(os.path.join(args.data_root, args.labels_file), encoding="utf16") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if is_first_row:
                is_first_row = False
                continue
            if row[column_class] in (None, ''):
                # this row does not contain label information
                continue
            if not os.path.isdir(os.path.join(args.data_root, args.extracted_frames_dir, row[column_filename])):
                print('Video {} skipped, this name appears in {} but not in {} folder'.format(row[column_filename],
                                                                                              args.labels_file,
                                                                                              args.extracted_frames_dir))
                continue

            # there is a 5 seconds delay from the indicated time and the real action
            DELAY = 5000
            time = int(row[column_time]) + 5000
            # we decided that all actions are 2 seconds long
            DURATION_ACTION = 2000
            start_time = time - DURATION_ACTION // 2
            end_time = time + DURATION_ACTION // 2
            start_frame = milliseconds_to_numframe(start_time, fps=args.fps)
            end_frame = milliseconds_to_numframe(end_time, fps=args.fps)

            filename = row[column_filename]
            class_idx = CLASS_INDEX[row[column_class]]
            labels = np.load(os.path.join(args.data_root, args.target_labels_dir, filename+'.npy'))
            for i in range(start_frame, end_frame):
                labels[i, CLASS_INDEX['Background']] = 0        # remove background label
                labels[i, class_idx] = 1
            np.save(os.path.join(args.data_root, args.target_labels_dir, filename+'.npy'), labels)

if __name__ == '__main__':
    base_dir = os.getcwd()
    base_dir = base_dir.split('/')[-1]
    CORRECT_LAUNCH_DIR = 'TRN.pytorch'
    if base_dir != CORRECT_LAUNCH_DIR:
        raise Exception('Wrong base dir, this file must be run from ' + CORRECT_LAUNCH_DIR + ' directory.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/JUDO', type=str)
    parser.add_argument('--data_info', default='data/data_info.json', type=str)
    parser.add_argument('--labels_file', default='metadati.csv', type=str)
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
    if not os.path.isdir(os.path.join(args.data_root, args.target_labels_dir)):
        os.makedir(os.path.join(args.data_root, args.target_labels_dir))
        print('{} folder created'.format(os.path.join(args.data_root, args.target_labels_dir)))
    if str(args.fps) not in args.extracted_frames_dir:
        raise Exception('The folder {} should contain the number of fps in its name, or the number '
                        'indicated in its name does not correspond with --fps argument(i.e. they must be '
                        'the same)'.format(args.extracted_frames_dir))

    main(build_data_info(args, basic_build=True))