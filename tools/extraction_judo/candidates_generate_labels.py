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

            for idx_frame in range(new_idxstartframe, new_idxendframe):
                img = cv2.imread(os.path.join(args.data_root, args.extracted_frames_dir, filename, str(idx_frame)+'.jpg'))
                if idx_frame == new_idxstartframe:
                    os.mkdir(os.path.join(args.data_root, args.new_frames_dir, str(new_idxstartframe)+'___'+filename))
                cv2.imwrite(os.path.join(args.data_root, args.new_frames_dir, str(new_idxstartframe)+'___'+filename, str(idx_frame)+'.jpg'), img)

            # minus 1 since there is a displacement of 1 between the index of the raw frame and
            # the arrays features and labels
            new_idxstartframe -= 1
            new_idxendframe -= 1

            features = np.load(os.path.join(args.data_root, args.extracted_features_dir, filename+'.npy'))
            features = features[new_idxstartframe//args.chunk_size:new_idxendframe//args.chunk_size]
            if FEATURES_SHAPE is None:
                FEATURES_SHAPE = features.shape
            assert FEATURES_SHAPE == features.shape, 'mismatch: '+str(FEATURES_SHAPE)+'  '+str(features.shape)

            np.save(os.path.join(args.data_root, args.new_features_dir, str(new_idxstartframe+1)+'___'+filename+'.npy'),
                    features)

            labels = np.load(os.path.join(args.data_root, args.target_labels_dir, filename+'.npy'))
            labels = labels[new_idxstartframe:new_idxendframe]

            if LABELS_SHAPE is None:
                LABELS_SHAPE = labels.shape
            assert LABELS_SHAPE == labels.shape, 'mismatch: '+str(LABELS_SHAPE)+'  '+str(labels.shape)
            num_frames = labels.shape[0]
            num_frames = num_frames - (num_frames % args.chunk_size)
            chunk_targets = labels[:num_frames]
            chunk_targets = chunk_targets[args.chunk_size // 2::args.chunk_size]
            assert chunk_targets.shape[0] == features.shape[0], 'mismatch: '+str(chunk_targets.shape[0])+'  '+str(features.shape[0])

            np.save(os.path.join(args.data_root, args.new_labels_dir, str(new_idxstartframe+1)+'___'+filename+'.npy'),
                    labels)

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
    # the directory where frames of video has been stored
    parser.add_argument('--extracted_frames_dir', default='video_frames_25fps', type=str)
    # the directory where labels arrays will be stored
    parser.add_argument('--extracted_features_dir', default='i3d_224x224_chunk9')
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

    args.new_frames_dir = 'candidates_'+args.extracted_frames_dir
    args.new_features_dir = 'candidates_'+args.extracted_features_dir
    args.new_labels_dir = 'candidates_'+args.target_labels_dir
    #os.mkdir(os.path.join(args.data_root, args.new_frames_dir))
    #os.mkdir(os.path.join(args.data_root, args.new_features_dir))
    #os.mkdir(os.path.join(args.data_root, args.new_labels_dir))

    main(args)