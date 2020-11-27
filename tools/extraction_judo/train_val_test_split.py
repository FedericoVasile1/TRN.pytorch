import json
import os
import argparse

from sklearn.model_selection import train_test_split

def main(args):
    DATASET = 'JUDO'
    dataset_type = 'TRIMMED'

    with open(args.data_info, 'r') as f:
        data_info = json.load(f)

    data_info[DATASET][dataset_type] = {}

    class_and_idxfile = None
    for class_name in os.listdir(os.path.join('media', 'data', 'vasile', 'judo_multiview', 'data', 'NageWaza')):
        for filename in os.listdir(os.path.join('media', 'data', 'vasile', 'judo_multiview', 'data', 'NageWaza', class_name)):
            idxfile = filename.split('_')[0]
            if class_and_idxfile is None:
                class_and_idxfile = {class_name+'_'+idxfile}
            else:
                class_and_idxfile.add(class_name+'_'+idxfile)

    class_and_idxfile = list(class_and_idxfile)
    SEED = 12
    X_train, X_test = train_test_split(class_and_idxfile, test_size=0.3, shuffle=True, random_state=SEED)
    X_val, X_test = train_test_split(X_test, test_size=0.5, shuffle=True, random_state=SEED)

    X_train_all = []
    for cls_idx in X_train:
        cls, idx = (cls_idx.split('_')[0], cls_idx.split('_')[1])
        for f in os.listdir(os.path.join('media', 'data', 'vasile', 'judo_multiview', 'data', 'NageWaza', cls)):
            if f.startswith(idx+'_'+'GoPro'):
                X_train_all.append(f)

    X_val_all = []
    for cls_idx in X_val:
        cls, idx = (cls_idx.split('_')[0], cls_idx.split('_')[1])
        for f in os.listdir(os.path.join('media', 'data', 'vasile', 'judo_multiview', 'data', 'NageWaza', cls)):
            if f.startswith(idx+'_'+'GoPro'):
                X_val_all.append(f)

    X_test_all = []
    for cls_idx in X_test:
        cls, idx = (cls_idx.split('_')[0], cls_idx.split('_')[1])
        for f in os.listdir(os.path.join('media', 'data', 'vasile', 'judo_multiview', 'data', 'NageWaza', cls)):
            if f.startswith(idx+'_'+'GoPro'):
                X_test_all.append(f)

    data_info[DATASET][dataset_type]['train_session_set'] = X_train_all
    data_info[DATASET][dataset_type]['val_session_set'] = X_val_all
    data_info[DATASET][dataset_type]['test_session_set'] = X_test_all

    with open(args.data_info+'_v2', 'w') as f:
        json.dump(data_info, f)

if __name__ == '__main__':
    base_dir = os.getcwd()
    base_dir = base_dir.split('/')[-1]
    CORRECT_LAUNCH_DIR = 'TRN.pytorch'
    if base_dir != CORRECT_LAUNCH_DIR:
        raise Exception('Wrong base dir, this file must be run from ' + CORRECT_LAUNCH_DIR + ' directory.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/JUDO', type=str)
    parser.add_argument('--data_info', default='data/data_info.json', type=str)
    args = parser.parse_args()

    if not os.path.isdir(os.path.join(args.data_root)):
        raise Exception('{} not found'.format(os.path.join(args.data_root)))
    if not os.path.isfile(os.path.join(args.data_info)):
        raise Exception('{} not found'.format(os.path.join(args.data_info)))

    main(args)