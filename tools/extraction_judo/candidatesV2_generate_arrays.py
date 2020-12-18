import os
import numpy as np
import csv
import sys
import argparse

import cv2

sys.path.append(os.getcwd())
from configs.build import build_data_info

def milliseconds_to_numframe(time_milliseconds, fps=25):
    fpms = fps * 0.001
    num_frame = (time_milliseconds * fpms)
    return int(num_frame)

def main(args):
    FEATURES_SHAPE = None
    LABELS_SHAPE = None

    column_filename = 0
    column_idxstartframe = 1
    column_idxendframe = 2
    column_idxfallframe = 3
    is_first_row = True
    with open(os.path.join(args.data_root, args.labels_file), mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if is_first_row:
                is_first_row = False
                continue

            filename = row[column_filename]

            idxstartframe = int(row[column_idxstartframe])
            idxendframe = int(row[column_idxendframe])
            # we start two seconds before and we finish two seconds after
            idxstartframe = idxstartframe - (2 * 25)
            idxendframe = idxendframe + (2 * 25)

            # need to adjust indices in order to be multiple of chunk_size
            # adjust start
            if idxstartframe % args.chunk_size == 0:
                new_idxstartframe = idxstartframe - args.chunk_size
                new_idxendframe = idxendframe - args.chunk_size
            else:
                new_idxstartframe = idxstartframe - (idxstartframe % args.chunk_size)
                new_idxendframe = idxendframe - (idxstartframe % args.chunk_size)
            new_idxstartframe += 1
            new_idxendframe += 1
            # adjust end
            new_idxendframe = new_idxendframe + (args.chunk_size - (new_idxendframe % args.chunk_size))
            new_idxendframe += 1

            if not os.path.isfile(os.path.join(args.data_root,
                                               args.new_labels_dir,
                                               filename+'.npy')):
                a = np.load(os.path.join(args.data_root, args.target_labels_dir, filename+'.npy'))
                b = np.zeros((a.shape[0], 2))
                b[:, 0] = 1     # initially is all background
                np.save(os.path.join(args.data_root, args.new_labels_dir, filename+'.npy'), b)

            l = np.load(os.path.join(args.data_root, args.new_labels_dir, filename + '.npy'))


            # minus 1 since there is a displacement of 1 between the index of the raw frame and
            # the arrays features and labels
            new_idxstartframe -= 1
            new_idxendframe -= 1

            for i in range(new_idxstartframe, new_idxendframe):
                l[i, 0] = 0
                l[i, 1] = 1

            np.save(os.path.join(args.data_root, args.new_labels_dir, filename+'.npy'), l)

if __name__ == '__main__':
    base_dir = os.getcwd()
    base_dir = base_dir.split('/')[-1]
    CORRECT_LAUNCH_DIR = 'TRN.pytorch'
    if base_dir != CORRECT_LAUNCH_DIR:
        raise Exception('Wrong base dir, this file must be run from ' + CORRECT_LAUNCH_DIR + ' directory.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/JUDO', type=str)
    parser.add_argument('--data_info', default='data/data_info.json', type=str)
    parser.add_argument('--labels_file', default='candidate_actions.csv', type=str)
    parser.add_argument('--chunk_size', default=9, type=int)
    parser.add_argument('--target_labels_dir', default='4s_target_frames_25fps', type=str)
    args = parser.parse_args()

    if not os.path.isdir(os.path.join(args.data_root + '/' + 'UNTRIMMED')):
        raise Exception('{} not found'.format(os.path.join(args.data_root + '/' + 'UNTRIMMED')))
    if not os.path.isfile(os.path.join(args.data_info)):
        raise Exception('{} not found'.format(os.path.join(args.data_info)))
    if not os.path.isfile(os.path.join(args.data_root + '/' + 'UNTRIMMED', args.labels_file)):
        raise Exception('{} not found'.format(os.path.join(args.data_root + '/' + 'UNTRIMMED', args.labels_file)))
    if not os.path.isdir(os.path.join(args.data_root + '/' + 'UNTRIMMED', args.extracted_frames_dir)):
        raise Exception('{} not found'.format(os.path.join(args.data_root + '/' + 'UNTRIMMED', args.extracted_frames_dir)))
    if not os.path.isdir(os.path.join(args.data_root + '/' + 'UNTRIMMED', args.extracted_features_dir)):
        raise Exception('{} not found'.format(os.path.join(args.data_root + '/' + 'UNTRIMMED', args.extracted_features_dir)))
    if not os.path.isdir(os.path.join(args.data_root + '/' + 'UNTRIMMED', args.target_labels_dir)):
        raise Exception('{} not found'.format(os.path.join(args.data_root + '/' + 'UNTRIMMED', args.target_labels_dir)))
    if str(args.chunk_size) not in args.extracted_features_dir:
        raise Exception('--extracted_features_dir and --chunk_size must contain the same chunk number')

    # do note modify these lines
    args.use_trimmed = False
    args.use_untrimmed = True
    args.eval_on_untrimmed = False
    args = build_data_info(args, basic_build=True)
    args.data_root = args.data_root + '/' + 'UNTRIMMED'
    args.train_session_set = args.train_session_set['UNTRIMMED']
    args.val_session_set = args.val_session_set['UNTRIMMED']
    args.test_session_set = args.test_session_set['UNTRIMMED']

    args.new_labels_dir = 'candidatesV2ALL_'+args.target_labels_dir
    #os.mkdir(os.path.join(args.data_root, args.new_frames_dir))
    #os.mkdir(os.path.join(args.data_root, args.new_features_dir))
    #os.mkdir(os.path.join(args.data_root, args.new_labels_dir))

    main(args)