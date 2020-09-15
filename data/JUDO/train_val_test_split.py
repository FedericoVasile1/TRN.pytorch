import json
import csv
import os
import argparse

from sklearn.model_selection import train_test_split

def main(args):
    DATASET = 'JUDO'

    with open(args.data_info, 'r') as f:
        data_info = json.load(f)

    data_info[DATASET] = {}
    data_info[DATASET]['class_index'] = ['Background', 'Ashi Waza', 'Te Waza', 'Koshi Waza', 'Sutemi Waza']

    # read all filenames
    column_filename = 40  # column AO
    filename_set = None
    is_first_row = True
    with open(os.path.join(args.data_root, args.labels_file), encoding="utf16") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if is_first_row:
                is_first_row = False
                continue
            filename = row[column_filename]
            if filename_set is None:
                filename_set = {filename}
            else:
                filename_set.add(filename)

    filename_list = list(filename_set)
    # Do the split: 60% training, 20% validation, 20% test
    SEED = 12
    X_train, X_test = train_test_split(filename_list, test_size=0.2, shuffle=True, random_state=SEED)
    X_train, X_val = train_test_split(X_train, test_size=0.25, shuffle=True, random_state=SEED)

    data_info[DATASET]['train_session_set'] = X_train
    data_info[DATASET]['val_session_set'] = X_val
    data_info[DATASET]['test_session_set'] = X_test

    with open(args.data_info, 'w') as f:
        json.dump(data_info, f)

if __name__ == '__main__':
    base_dir = os.getcwd()
    base_dir = base_dir.split('/')[-1]
    CORRECT_LAUNCH_DIR = 'TRN.pytorch'
    if base_dir != CORRECT_LAUNCH_DIR:
        raise Exception('Wrong base dir, this file must be run from ' + CORRECT_LAUNCH_DIR + ' directory.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/JUDO', type=str)
    parser.add_argument('--labels_file', default='metadati.csv', type=str)
    parser.add_argument('--data_info', default='data/data_info.json', type=str)
    args = parser.parse_args()

    if not os.path.isdir(os.path.join(args.data_root)):
        raise Exception('{} not found'.format(os.path.join(args.data_root)))
    if not os.path.isdir(os.path.join(args.data_root, args.labels_file)):
        raise Exception('{} not found'.format(os.path.join(args.data_root, args.labels_file)))
    if not os.path.isdir(os.path.join(args.data_info)):
        raise Exception('{} not found'.format(os.path.join(args.data_info)))

    main(args)
