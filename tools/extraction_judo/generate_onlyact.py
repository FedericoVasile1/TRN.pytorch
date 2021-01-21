import os

import numpy as np
import cv2

if __name__ == '__main__':
    BASE_FOLDER_FRAMES = 'video_frames_25fps'
    BASE_FOLDER_TARGETS = '4s_target_frames_25fps'
    NEW_FOLDER_ACT_FRAMES = 'onlyact_video_frames_25fps'
    NEW_FOLDER_ACT_TARGETS = 'onlyact_4s_target_frames_25fps'
    DATA_ROOT = 'data/JUDO/UNTRIMMED'

    os.mkdir(os.path.join(DATA_ROOT, NEW_FOLDER_ACT_FRAMES))
    os.mkdir(os.path.join(DATA_ROOT, NEW_FOLDER_ACT_TARGETS))

    videos = os.listdir(os.path.join(DATA_ROOT, BASE_FOLDER_FRAMES))
    videos = [v for v in videos if '.mp4' in v]
    for v in videos:
        target = np.load(os.path.join(DATA_ROOT, BASE_FOLDER_TARGETS, v+'.npy'))

        count = 0
        for i in range(len(target)):
            if target[i].argmax() == 0:
                if count == 100:
                    start = i - 100
                    end = i - 1
                    os.mkdir(os.path.join(DATA_ROOT, NEW_FOLDER_ACT_FRAMES, str(start)+'___'+v))

                    # save frames
                    imgs = []
                    for k in range(start, end+1):
                        imgs.append(cv2.imread(os.path.join(DATA_ROOT, BASE_FOLDER_FRAMES, v, str(k+1)+'.jpg')))
                    for idx, k in enumerate(range(start, end+1)):
                        cv2.imwrite(os.path.join(DATA_ROOT, NEW_FOLDER_ACT_FRAMES, str(start)+'___'+v, str(k+1)+'.jpg'), imgs[idx])

                    # save target
                    np.save(os.path.join(DATA_ROOT, NEW_FOLDER_ACT_TARGETS, str(start)+'___'+v+'.npy'), target[end-50])

                elif count > 100:
                    global_start = i - count
                    global_end = i - 1

                    for j in range(global_start, global_end+1, 100):
                        os.mkdir(os.path.join(DATA_ROOT, NEW_FOLDER_ACT_FRAMES, str(j)+'___'+v))

                        imgs = []
                        for k in range(j, j+100):
                            imgs.append(cv2.imread(os.path.join(DATA_ROOT, BASE_FOLDER_FRAMES, v, str(k+1)+'.jpg')))
                        for idx, k in enumerate(range(j, j+100)):
                            cv2.imwrite(os.path.join(DATA_ROOT, NEW_FOLDER_ACT_FRAMES, str(j)+'___'+v, str(k+1)+'.jpg'), imgs[idx])

                        np.save(os.path.join(DATA_ROOT, NEW_FOLDER_ACT_TARGETS, str(j)+'___'+v+'.npy'), target[j+50])

                    end = i - 1
                    start = i - 100
                    os.mkdir(os.path.join(DATA_ROOT, NEW_FOLDER_ACT_FRAMES, str(start)+'___'+v))

                    # save frames
                    imgs = []
                    for k in range(start, end+1):
                        imgs.append(cv2.imread(os.path.join(DATA_ROOT, BASE_FOLDER_FRAMES, v, str(k+1)+'.jpg')))
                    for idx, k in enumerate(range(start, end+1)):
                        cv2.imwrite(os.path.join(DATA_ROOT, NEW_FOLDER_ACT_FRAMES, str(start)+'___'+v, str(k+1)+'.jpg'), imgs[idx])

                    # save target
                    np.save(os.path.join(DATA_ROOT, NEW_FOLDER_ACT_TARGETS, str(start)+'___'+v+'.npy'), target[end-50])

                count = 0
                continue

            count += 1