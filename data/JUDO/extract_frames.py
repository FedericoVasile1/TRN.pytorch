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
        print(filename)
        os.mkdir('video_frames_30fps/'+filename)
        folder = filename.split('_')[4]
        vidcap = cv2.VideoCapture('data_00/'+folder+'/'+filename)

        success, image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite('video_frames_30fps/'+filename+"/%d.jpg" % count, image)  # save frame as JPEG file
            success, image = vidcap.read()
            count += 1

if __name__ == '__main__':
    # run from JUDO basedir
    main()
