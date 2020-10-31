import json
import csv
import os
import argparse

from sklearn.model_selection import train_test_split

def main(args):
    DATASET = 'JUDO'

    with open(args.data_info, 'r') as f:
        data_info = json.load(f)

    video_names_set = set()
    with open(os.path.join(args.data_root, args.labels_file), encoding="utf16") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if is_first_row:
                is_first_row = False
                continue
            video_names_set.add(row['video_name'])

    video_names_list = list(video_names_set)
    # Do the split: 70% training, 15% validation, 15% test
    SEED = 12
    X_train, X_test = train_test_split(video_names_list, test_size=0.3, shuffle=True, random_state=SEED)
    X_val, X_test = train_test_split(X_test, test_size=0.5, shuffle=True, random_state=SEED)

    data_info[DATASET]['TRIMMED']['train_session_set'] = X_train
    data_info[DATASET]['TRIMMED']['val_session_set'] = X_val
    data_info[DATASET]['TRIMMED']['test_session_set'] = X_test

    with open(args.data_info, 'w') as f:
        json.dump(data_info, f)

if __name__ == '__main__':
    base_dir = os.getcwd()
    base_dir = base_dir.split('/')[-1]
    CORRECT_LAUNCH_DIR = 'TRN.pytorch'
    if base_dir != CORRECT_LAUNCH_DIR:
        raise Exception('Wrong base dir, this file must be run from ' + CORRECT_LAUNCH_DIR + ' directory.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/JUDO/TRIMMED', type=str)
    parser.add_argument('--labels_file', default='fall_detections.csv', type=str)
    parser.add_argument('--data_info', default='data/data_info.json', type=str)
    args = parser.parse_args()

    if not os.path.isdir(os.path.join(args.data_root)):
        raise Exception('{} not found'.format(os.path.join(args.data_root)))
    if not os.path.isdir(os.path.join(args.data_root, args.labels_file)):
        raise Exception('{} not found'.format(os.path.join(args.data_root, args.labels_file)))
    if not os.path.isdir(os.path.join(args.data_info)):
        raise Exception('{} not found'.format(os.path.join(args.data_info)))

    main(args)
