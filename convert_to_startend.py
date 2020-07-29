import numpy as np
import os

def convert():
    DATA_ROOT = 'data/THUMOS'
    TARGET = 'target'
    NUM_CLASSES = 22

    backgr_vect = np.zeros((1, NUM_CLASSES * 2))
    backgr_vect[0, 0] = 1
    targets_files = os.listdir(os.path.join(DATA_ROOT, TARGET))
    targets = [target for target in targets_files if 'video' in target]
    for target_name in targets:
        target_vect = np.load(os.path.join(DATA_ROOT, TARGET, target_name))
        # target.shape == (num_frames, num_classes)

        new_vect = np.zeros((target_vect.shape[0], target_vect.shape[1] * 2))
        class_vect = None
        for idx_frame in range(target_vect.shape[0]):
            if target_vect[idx_frame, 0] == 1:
                new_vect[idx_frame] = backgr_vect

                if class_vect is not None:
                    class_vect = np.array(class_vect)
                    class_vect = np.concatenate((class_vect, class_vect), axis=1)
                    class_vect[class_vect.shape[0]//2:] = 0
                    class_vect[:, class_vect.shape[1]//2:] = 0

                    idx = class_vect[0].argmax()
                    appo = np.zeros((1, NUM_CLASSES))
                    appo[0, idx] = 1

                    class_vect[class_vect.shape[0]//2:, class_vect.shape[1]//2:] = appo

                    new_vect[idx_frame - class_vect.shape[0]:idx_frame] = class_vect
                    class_vect = None
                continue

            if class_vect is None:
                class_vect = []
            class_vect.append(target_vect[idx_frame])

        np.save(os.path.join(DATA_ROOT, TARGET+'_startend', target_name), new_vect)

if __name__ == '__main__':
    convert()