import cv2

import csv
import os

def main():
    column_filename = 40    # column AO
    filename_set = None
    is_first_row = True
    with open('metadati.csv', encoding="utf16") as csv_file:
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

    print(len(filename_set))
    for filename in filename_set:
        if os.path.isdir('video_frames_25fps/'+filename):
            continue
        print(filename)

        os.mkdir('video_frames_25fps/'+filename)

        filename = filename.replace(' ', '\ ')
        folder = filename.split('_')[4]
        path = 'data_00/'+folder+'/'+filename
        savedir = 'video_frames_25fps/'+filename+'/'
        os.system('ffmpeg -i '+path+' -r 25.0 '+savedir+'%d.jpg')

if __name__ == '__main__':
    # run from JUDO basedir
    base_dir = os.getcwd()
    base_dir = base_dir.split('/')[-1]
    if base_dir != 'JUDO':
        raise Exception('Wrong base dir, this file must be run from JUDO/ directory.')

    main()
