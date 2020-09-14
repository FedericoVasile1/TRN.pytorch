import json
import csv
import os

from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # run from JUDO basedir
    base_dir = os.getcwd()
    base_dir = base_dir.split('/')[-1]
    if base_dir != 'TRN.pytorch':
        raise Exception('Wrong base dir, this file must be run from JUDO/ directory.')

    DATA_INFO = 'data/data_info.json'
    DATASET = 'JUDO'
    with open(DATA_INFO, 'r') as f:
        data_info = json.load(f)

    data_info[DATASET] = {}
    data_info[DATASET]['class_index'] = ['Background', 'Ashi Waza', 'Te Waza', 'Koshi Waza', 'Sutemi Waza']

    # read all filenames
    column_filename = 40  # column AO
    filename_set = None
    is_first_row = True
    with open('data/JUDO/metadati.csv', encoding="utf16") as csv_file:
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

    with open(DATA_INFO, 'w') as f:
        json.dump(data_info, f)