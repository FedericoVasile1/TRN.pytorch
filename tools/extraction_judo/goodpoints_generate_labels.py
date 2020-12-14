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
    # for each goodpoint action, we take a clip 10 seconds long

    NEW_MODEL_FEATURES_DIR = 'goodpoints_'+args.model_features
    if os.path.isdir(os.path.join(args.data_root, NEW_MODEL_FEATURES_DIR)):
        shutil.rmtree(os.path.join(args.data_root, NEW_MODEL_FEATURES_DIR))
    os.mkdir(os.path.join(args.data_root, NEW_MODEL_FEATURES_DIR))
    NEW_MODEL_TARGETS_DIR = 'goodpoints_'+args.model_targets
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

        CHUNK_SIZE = args.chunk_size
        FEATURES_SHAPE = None

        csv_reader = csv.reader(labels_file, delimiter=',')
        is_first_row = True
        for row in csv_reader:
            if is_first_row:
                is_first_row = False
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

            targets = np.load(os.path.join(args.data_root, args.model_targets, video_name + '.npy'))
            targets = targets[startframe_clip: endframe_clip]

            # sanity-check
            num_frames = targets.shape[0]
            num_frames = num_frames - (num_frames % CHUNK_SIZE)
            chunk_targets = targets[:num_frames]
            chunk_targets = chunk_targets[CHUNK_SIZE // 2::CHUNK_SIZE]
            assert chunk_targets.shape == features.shape, 'shape mismatch between targets and features: '+\
                                                           chunk_targets.shape+'  '+features.shape
            if FEATURES_SHAPE is None:
                FEATURES_SHAPE = features.shape
            assert FEATURES_SHAPE == features.shape, 'shape mismatch between different features'+\
                                                      FEATURES_SHAPE+'  '+features.shape

            np.save(os.path.join(args.data_root, NEW_MODEL_FEATURES_DIR, str(starttime)+'___'+video_name+'.npy'),
                    features)
            np.save(os.path.join(args.data_root, NEW_MODEL_TARGETS_DIR, str(starttime) + '___' + video_name + '.npy'),
                    targets)

            # TODO: SAVE ALSO FRAMES, I.E. goodpoints_video_frames_25fps

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
    parser.add_argument('--chunk_size', default=9, type=int)
    parser.add_argument('--model_features', default='i3d_224x224_chunk9')
    parser.add_argument('--model_targets', default='4s_target_frames_25fps')
    args = parser.parse_args()

    if not os.path.isdir(os.path.join(args.data_root + '/' + 'UNTRIMMED')):
        raise Exception('{} not found'.format(os.path.join(args.data_root + '/' + 'UNTRIMMED')))
    if not os.path.isfile(os.path.join(args.data_root + '/' + 'UNTRIMMED', args.labels_file)):
        raise Exception('{} not found'.format(os.path.join(args.data_root + '/' + 'UNTRIMMED', args.labels_file )))
    if not os.path.isdir(os.path.join(args.data_root + '/' + 'UNTRIMMED', args.model_features)):
        raise Exception('{} not found'.format(os.path.join(args.data_root + '/' + 'UNTRIMMED', args.model_features)))
    if not os.path.isdir(os.path.join(args.data_root + '/' + 'UNTRIMMED', args.model_targets)):
        raise Exception('{} not found'.format(os.path.join(args.data_root + '/' + 'UNTRIMMED', args.model_targets)))
    if not args.model_features.endswith('_chunk'+str(args.chunk_size)):
        raise Exception('Wrong --model_features or --chunk_size option. They must indicate the same chunk size')

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