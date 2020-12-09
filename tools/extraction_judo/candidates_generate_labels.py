import os
import numpy as np
import csv
import sys
import argparse

sys.path.append(os.getcwd())
from configs.build import build_data_info

def milliseconds_to_numframe(time_milliseconds, fps=25):
    fpms = fps * 0.001
    num_frame = (time_milliseconds * fpms)
    return int(num_frame)

def main(args):
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
            idxfallframe = int(row[column_idxfallframe])

            features = np.load(os.path.join(args.data_root, args.extracted_features_dir, filename+'.npy'))
            features = features[idxstartframe//args.chunk_size:idxendframe//args.chunk_size]
            np.save(os.path.join(args.data_root, args.new_features_dir, str(idxfallframe)+'___'+filename+'.npy'),
                    features)

            labels = np.load(os.path.join(args.data_root, args.target_labels_dir, filename+'.npy'))
            num_frames = labels.shape[0]
            num_frames = num_frames - (num_frames % args.chunk_size)
            labels = labels[:num_frames]
            labels = labels[args.chunk_size // 2::args.chunk_size]
            labels = labels[idxstartframe//args.chunk_size:idxendframe//args.chunk_size]
            np.save(os.path.join(args.data_root, args.new_labels_dir, str(idxfallframe)+'___'+filename+'.npy'),
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

    args.new_features_dir = 'candidate_'+args.extracted_features_dir
    args.new_labels_dir = 'candidate_'+args.target_labels_dir
    os.mkdir(os.path.join(args.data_root, args.new_features_dir))
    os.mkdir(os.path.join(args.data_root, args.new_labels_dir))

    main(args)