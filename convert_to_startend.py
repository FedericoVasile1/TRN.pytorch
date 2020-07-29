import numpy as np
import os

def convert():
    DATA_ROOT = 'data/THUMOS'
    TARGET = 'target'

    targets_files = os.listdir(os.path.join(DATA_ROOT, TARGET))
    targets = [target for target in targets_files if 'video' in target]
    for target_name in targets:
        target_vect = np.load(os.path.join(DATA_ROOT, TARGET, target_name))
        # target.shape == (num_frames, num_classes)

        new_vect = None
        for idx_frame in range(target_vect.shape[0]):
            if target_vect[idx_frame, 0] == 1:
                if new_vect is not None:
                    new_vect = np.array(new_vect)
                    second_half = new_vect[new_vect.shape[0]//2:]
                    second_half = (second_half != 0) * (second_half + 21)
                    new_vect[new_vect.shape[0] // 2:] = second_half

                    target_vect[idx_frame - new_vect.shape[0]:idx_frame, :] = new_vect

                    new_vect = None
                continue

            if new_vect is None:
                new_vect = []
            new_vect.append(target_vect[idx_frame])

        np.save(os.path.join(DATA_ROOT, TARGET+'_startend', target_name), target_vect)

if __name__ == '__main__':
    convert()