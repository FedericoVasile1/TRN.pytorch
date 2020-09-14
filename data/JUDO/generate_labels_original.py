import os
import numpy as np
import csv

def milliseconds_to_numframe(time_milliseconds, fps=25):
    fpms = fps * 0.001
    num_frame = (time_milliseconds * fpms)
    return int(num_frame)

def main():
    CLASS_INDEX = {'Background': 0, 'Ashi Waza': 1, 'Te Waza': 2, 'Koshi Waza': 3, 'Sutemi Waza': 4}
    NUM_CLASSES = len(CLASS_INDEX)

    for video_dir in os.listdir('video_frames_25fps'):
        if '.mp4' not in video_dir:
            continue
        num_frames_video = len(os.listdir('video_frames_25fps/'+video_dir))
        all_background_labels = np.zeros((num_frames_video, NUM_CLASSES))
        all_background_labels[:, CLASS_INDEX['Background']] = 1
        np.save('target_frames_25fps_original/'+video_dir+'.npy', all_background_labels)

    column_time = 1     # column B
    column_class = 37
    column_filename = 40  # column AO
    is_first_row = True
    with open('metadati.csv', encoding="utf16") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if is_first_row:
                is_first_row = False
                continue
            if row[column_class] in (None, ''):
                # this row does not contain label information
                continue
            if not os.path.isdir('video_frames_25fps/'+row[column_filename]):
                continue

            time = int(row[column_time])
            DURATION_ACTION = 10000
            start_time = time
            end_time = time + DURATION_ACTION
            start_frame = milliseconds_to_numframe(start_time)
            end_frame = milliseconds_to_numframe(end_time)

            filename = row[column_filename]
            class_idx = CLASS_INDEX[row[column_class]]
            labels = np.load('target_frames_25fps_original/'+filename+'.npy')
            for i in range(start_frame, end_frame):
                labels[i, CLASS_INDEX['Background']] = 0        # remove background label
                labels[i, class_idx] = 1
            np.save('target_frames_25fps_original/'+filename+'.npy', labels)

if __name__ == '__main__':
    # run from JUDO basedir
    base_dir = os.getcwd()
    base_dir = base_dir.split('/')[-1]
    if base_dir != 'JUDO':
        raise Exception('Wrong base dir, this file must be run from JUDO/ directory.')

    main()