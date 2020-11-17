import os
import sys
import argparse
import csv
import json
sys.path.append(os.getcwd())

def main(args):
    with open(os.path.join(args.data_root, args.labels_file), encoding='utf-16') as csv_file:
        COLUMN_POINT = 3
        COLUMN_LABEL = 37
        COLUMN_FILENAME = 40

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        points_set = {'No Score'}
        filename = None
        count = 1
        nogoodpoints_video = []
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
                continue
            if row[COLUMN_LABEL] == '':
                continue
            if filename is None:
                filename = row[COLUMN_FILENAME]

            if row[COLUMN_FILENAME] != filename:
                if points_set == {'No Score'}:
                    nogoodpoints_video.append(filename)
                    print(count, filename)
                    count += 1
                points_set = {'No Score'}

            points_set.add(row[COLUMN_POINT])
            filename = row[COLUMN_FILENAME]
    f = open(os.path.join(args.data_root, 'nogoodpoints_video.json'), 'w')
    f.write(json.dumps(nogoodpoints_video))
    f.close()

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