import os
import sys
import argparse
import csv
import shutil
import numpy as np

sys.path.append(os.getcwd())
from configs.build import build_data_info

def milliseconds_to_numframe(time_milliseconds, fps=25):
    fpms = fps * 0.001
    num_frame = (time_milliseconds * fpms)
    return int(num_frame)

def main(args):
    # for each good point action, we take also the two preceding and two following seconds and label
    #  them as backgorund

    NEW_MODEL_FEATURES_DIR = 'goodpoints_'+args.model_features
    if os.path.isdir(os.path.join(args.data_root, NEW_MODEL_FEATURES_DIR)):
        shutil.rmtree(os.path.join(args.data_root, NEW_MODEL_FEATURES_DIR))
    os.mkdir(os.path.join(args.data_root, NEW_MODEL_FEATURES_DIR))
    NEW_MODEL_TARGETS_DIR = 'goodpoints_target_frames_25fps'
    if os.path.isdir(os.path.join(args.data_root, NEW_MODEL_TARGETS_DIR)):
        shutil.rmtree(os.path.join(args.data_root, NEW_MODEL_TARGETS_DIR))
    os.mkdir(os.path.join(args.data_root, NEW_MODEL_TARGETS_DIR))

    with open(os.path.join(args.data_root, args.labels_file), encoding='utf-16') as labels_file:
        COLUMN_LABEL = 37
        COLUMN_FILENAME = 40
        COLUMN_STARTTIME = 1
        COLUMN_POINT = 3

        CLASS_INDEX = {}
        for i, class_name in enumerate(args.class_index):
            CLASS_INDEX[class_name] = i

        CHUNK_SIZE = 9

        csv_reader = csv.reader(labels_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
                continue
            if row[COLUMN_LABEL] == '':
                continue
            if row[COLUMN_POINT] not in ('Ippon', 'Waza Hari'):
                continue
            if row[COLUMN_FILENAME] not in args.train_session_set and\
               row[COLUMN_FILENAME] not in args.val_session_set and\
               row[COLUMN_FILENAME] not in args.test_session_set:
                continue

            label = row[COLUMN_LABEL]
            video_name = row[COLUMN_FILENAME]
            starttime = int(row[COLUMN_STARTTIME])

            starttime_clip = starttime
            startframe_clip = milliseconds_to_numframe(starttime_clip) - 1
            endtime_clip = starttime + 10000
            endframe_clip = milliseconds_to_numframe(endtime_clip) - 1

            features = np.load(os.path.join(args.data_root, args.model_features, video_name + '.npy'))
            features = features[startframe_clip // CHUNK_SIZE: endframe_clip // CHUNK_SIZE]

            targets = np.zeros((len(features) * CHUNK_SIZE, args.num_classes))
            # len(targets) should be about 25*10
            j = len(targets) // 10
            targets[:j*2, 0] = 1
            targets[j*2:j*8, CLASS_INDEX[label]] = 1
            targets[j*8:, 0] = 1

            np.save(os.path.join(args.data_root, NEW_MODEL_FEATURES_DIR, str(starttime)+'___'+video_name+'.npy'),
                    features)
            np.save(os.path.join(args.data_root, NEW_MODEL_TARGETS_DIR, str(starttime) + '___' + video_name + '.npy'),
                    targets)

if __name__ == '__main__':
    base_dir = os.getcwd()
    base_dir = base_dir.split('/')[-1]
    CORRECT_LAUNCH_DIR = 'TRN.pytorch'
    if base_dir != CORRECT_LAUNCH_DIR:
        raise Exception('Wrong base dir, this file must be run from ' + CORRECT_LAUNCH_DIR + ' directory.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/JUDO', type=str)
    parser.add_argument('--data_info', default='data/data_info.json', type=str) # useless, needed only to do not generate errors
    parser.add_argument('--labels_file', default='metadati.csv')
    parser.add_argument('--model_features', default='i3d_224x224_chunk9')
    args = parser.parse_args()

    if not os.path.isdir(os.path.join(args.data_root + '/' + 'UNTRIMMED')):
        raise Exception('{} not found'.format(os.path.join(args.data_root + '/' + 'UNTRIMMED')))
    if not os.path.isfile(os.path.join(args.data_root + '/' + 'UNTRIMMED', args.labels_file)):
        raise Exception('{} not found'.format(os.path.join(args.data_root + '/' + 'UNTRIMMED', args.labels_file )))
    if not os.path.isdir(os.path.join(args.data_root + '/' + 'UNTRIMMED', args.model_features)):
        raise Exception('{} not found'.format(os.path.join(args.data_root + '/' + 'UNTRIMMED', args.model_features)))

    # do note modify these two lines
    args.use_trimmed = False
    args.use_untrimmed = True
    args.eval_on_untrimmed = False
    args = build_data_info(args, basic_build=True)
    args.data_root = args.data_root + '/' + 'UNTRIMMED'
    args.train_session_set = args.train_session_set['UNTRIMMED']
    args.val_session_set = args.val_session_set['UNTRIMMED']
    args.test_session_set = args.test_session_set['UNTRIMMED']

    main(args)