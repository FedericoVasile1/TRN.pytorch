import os, glob
import copy
import numpy as np

def enumerate_labels(data_root='data/THUMOS', annotation_folder='TH14_Temporal_annotations_validation/annotation'):
    count_2_file = {}
    for a, i in enumerate(sorted(glob.glob(os.path.join(data_root, annotation_folder, '*.txt')))):
      count_2_file[a] = i.split('/')[-1]

    # move ambiguous class from start to end
    count_2_file[21] = copy.deepcopy(count_2_file[0])
    del count_2_file[0]

    return count_2_file       # it should be with keys from 1 to 21, with ambiguous at the end

def read_files(annotations_folder, count_2_file, target_folder='target_frames_24fps', data_root='data/THUMOS'):
    for count, filename in count_2_file.items():
        f = open(os.path.join(data_root, annotations_folder, filename), "r")
        for line in f:
            videoname = line.split(' ')[0]
            start = float(line.split(' ')[2])
            end = float(line.split(' ')[3].split('\\')[0])

            dest = os.path.join(data_root, target_folder, videoname+'.npy')
            target_array = np.load(dest)

            start_frame = int(start * 24)        # 24 fps
            end_frame = int(end * 24)
            one_hot_vect = np.zeros(22)     # 22 classes
            one_hot_vect[count] = 1
            frames = np.arange(start_frame, end_frame)
            try:
                target_array[frames, :] = one_hot_vect
            except:
                print(filename, videoname, start, end)
                return

            np.save(dest, target_array)

def create_zeros_labels(frames_folder='video_frames_24fps', data_root='data/THUMOS', target_folder='target_frames_24fps'):
    for video_dir in os.listdir(os.path.join(data_root, frames_folder)):
        #if 'video_validation' not in video_dir and 'video_test' not in video_dir:
        if 'video_test' not in video_dir:
            continue
        num_frames = len([name for name in os.listdir(os.path.join(data_root, frames_folder, video_dir))])
        all_background_vect = np.zeros((num_frames, 22))    # 22 classes
        all_background_vect[:, 0] = 1      # for each frame, set label to background
        np.save(os.path.join(data_root, target_folder, video_dir), all_background_vect)

if __name__ == '__main__':
    create_zeros_labels()

    #count_2_file = enumerate_labels()
    #read_files(os.path.join('TH14_Temporal_annotations_validation', 'annotation'), count_2_file)
    count_2_file = enumerate_labels(annotation_folder='TH14_Temporal_Annotations_Test/annotations/annotation')
    read_files(os.path.join('TH14_Temporal_Annotations_Test', 'annotations', 'annotation'), count_2_file)


    #print(np.load('data/THUMOS/target_frames_24fps/video_validation_0000170.npy')[590:630, :])
    #print(np.load('data/THUMOS/target_frames/video_validation_0000162.npy').shape)
