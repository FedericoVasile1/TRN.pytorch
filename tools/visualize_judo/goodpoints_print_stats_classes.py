import os
import sys
import argparse
import csv
sys.path.append(os.getcwd())

def main(args):
    with open(os.path.join(args.data_root, args.labels_file), encoding='utf-16') as csv_file:
        COLUMN_POINT = 3
        COLUMN_LABEL = 37
        labelpoint_to_count = {}

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
                continue
            if row[COLUMN_LABEL] == '':
                continue

            label = row[COLUMN_LABEL]
            point = row[COLUMN_POINT]
            if label not in labelpoint_to_count:
                labelpoint_to_count[label] = {}
            if point not in labelpoint_to_count[label]:
                labelpoint_to_count[label][point] = 0
            labelpoint_to_count[label][point] += 1

    for label in labelpoint_to_count:
        print('LABEL: ', label)
        for point in labelpoint_to_count[label]:
            print('   POINT: {:10s}  COUNT={}'.format(point, labelpoint_to_count[label][point]))
    print('===================')
    print('TOTAL POINTS')
    point_to_count = {}
    for label in labelpoint_to_count:
        for point in labelpoint_to_count[label]:
            if point not in point_to_count:
                point_to_count[point] = 0
            point_to_count[point] += labelpoint_to_count[label][point]
    for point in point_to_count:
        print('POINT: {:10s}  COUNT={}'.format(point, point_to_count[point]))

if __name__ == '__main__':
    base_dir = os.getcwd()
    base_dir = base_dir.split('/')[-1]
    CORRECT_LAUNCH_DIR = 'TRN.pytorch'
    if base_dir != CORRECT_LAUNCH_DIR:
        raise Exception('Wrong base dir, this file must be run from ' + CORRECT_LAUNCH_DIR + ' directory.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/JUDO/UNTRIMMED', type=str)
    parser.add_argument('--labels_file', default='metadati.csv')
    # TODO: add --phase option, i.e. read from data/data_info.json the file name together with
    #        the split they belong split, so print per-split stats instead of all together
    args = parser.parse_args()

    if not os.path.isdir(os.path.join(args.data_root)):
        raise Exception('{} not found'.format(os.path.join(args.data_root)))
    if not os.path.isfile(os.path.join(args.data_root, args.labels_file)):
        raise Exception('{} not found'.format(os.path.join(args.data_root, args.labels_file )))

    main(args)