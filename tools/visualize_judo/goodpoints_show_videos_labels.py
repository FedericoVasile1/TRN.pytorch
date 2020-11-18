import os
import os.path as osp
import sys
import argparse
import numpy as np
import cv2
import random

import torch
from PIL import Image
from torchvision import transforms

sys.path.append(os.getcwd())
from configs.build import build_data_info
from lib import utils as utl

def milliseconds_to_numframe(time_milliseconds, fps=25):
    fpms = fps * 0.001
    num_frame = (time_milliseconds * fpms)
    return int(num_frame)

def goodpoints_show_video_predictions(args,
                                      video_name,
                                      target_metrics,
                                      frames_dir='video_frames_25fps',
                                      fps=25,
                                      transform=None):
    target_metrics = torch.argmax(torch.tensor(target_metrics), dim=1)
    speed = 1.0
    if args.save_video:
        print('Loading and saving video: ' + video_name + ' . . . .\n.\n.')
        frames = []

    num_frames = target_metrics.shape[0]
    start_millisecond = int(video_name.split(':')[0]) + 4000
    start_frame = milliseconds_to_numframe(start_millisecond)
    idx = start_frame
    while idx < num_frames:
        #idx_frame = idx * args.chunk_size + args.chunk_size // 2
        idx_frame = idx
        pil_frame = Image.open(osp.join(args.data_root,
                                        frames_dir,
                                        video_name,
                                        str(idx_frame) + '.jpg')).convert('RGB')
        if transform is not None:
            pil_frame = transform(pil_frame)

        open_cv_frame = np.array(pil_frame)

        # Convert RGB to BGR
        open_cv_frame = open_cv_frame[:, :, ::-1].copy()

        open_cv_frame = cv2.copyMakeBorder(open_cv_frame, 60, 0, 30, 30, borderType=cv2.BORDER_CONSTANT, value=0)
        target_label = args.class_index[target_metrics[idx]]

        cv2.putText(open_cv_frame,
                    target_label,
                    (0, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    1)

        cv2.putText(open_cv_frame,
                    '{:.2f}s'.format(idx_frame / fps),
                    (295, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 255, 255),
                    1)
        cv2.putText(open_cv_frame,
                    'speed: {}x'.format(speed),
                    (295, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 255, 255),
                    1)
        cv2.putText(open_cv_frame,
                    str(idx_frame),
                    (295, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 255, 255),
                    1)

        if args.save_video:
            frames.append(open_cv_frame)

        idx += 1

    if args.save_video:
        H, W, _ = open_cv_frame.shape
        out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps / args.chunk_size, (W, H))
        for frame in frames:
            out.write(frame)
        out.release()
        print('. . . video saved at ' + os.path.join(os.getcwd(), video_name))

if __name__ == '__main__':
    base_dir = os.getcwd()
    base_dir = base_dir.split('/')[-1]
    CORRECT_LAUNCH_DIR = 'TRN.pytorch'
    if base_dir != CORRECT_LAUNCH_DIR:
        raise Exception('Wrong base dir, this file must be run from ' + CORRECT_LAUNCH_DIR + ' directory.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/JUDO', type=str)
    parser.add_argument('--data_info', default='data/data_info.json', type=str) # useless, it is here only to do not generate errors
    parser.add_argument('--frames_dir', default='video_frames_25fps', type=str)
    parser.add_argument('--targets_dir', default='goodpoints_target_frames_25fps', type=str)
    # the fps at which videos frames are previously extracted
    parser.add_argument('--fps', default=25, type=int)
    parser.add_argument('--chunk_size', default=9, type=int)
    parser.add_argument('--phase', default='train', type=str)
    parser.add_argument('--video_name', default='', type=str)
    parser.add_argument('--seed', default=-1, type=int)
    args = parser.parse_args()

    if not os.path.isdir(os.path.join(args.data_root)):
        raise Exception('{} not found'.format(os.path.join(args.data_root)))
    if not os.path.isfile(os.path.join(args.data_info)):
        raise Exception('{} not found'.format(os.path.join(args.data_info)))
    if args.phase not in ('train', 'val', 'test'):
        raise Exception('Wrong --phase argument. Expected one of: train|val|test')
    if str(args.fps) not in args.frames_dir:
        raise Exception('The folder {} should contain the number of fps in its name, or the number '
                        'indicated in its name does not correspond with --fps argument(i.e. they must be '
                        'the same)'.format(args.frames_dir))

    if args.seed != -1:
        np.random.seed(args.seed)

    # do not modify
    args.eval_on_untrimmed = False
    args.use_trimmed = False
    args.use_untrimmed = True
    args.save_video = True

    args = build_data_info(args, basic_build=True)
    args.data_root = args.data_root + '/' + 'UNTRIMMED'
    args.train_session_set = args.train_session_set['UNTRIMMED']
    args.val_session_set = args.val_session_set['UNTRIMMED']
    args.test_session_set = args.test_session_set['UNTRIMMED']

    if not os.path.isdir(os.path.join(args.data_root, args.frames_dir)):
        raise Exception('{} not found'.format(os.path.join(args.data_root, args.frames_dir)))
    if not os.path.isdir(os.path.join(args.data_root, args.targets_dir)):
        raise Exception('{} not found'.format(os.path.join(args.data_root, args.targets_dir)))

    if args.video_name == '':
        videos_list = getattr(args, args.phase+'_session_set')
        utl.set_seed(int(args.seed))
        random.shuffle(videos_list)
        video_name = videos_list[0]
    else:
        video_name = [args.video_name]

    '''
    transform = transforms.Compose([
        transforms.Resize((224, 320)),
        transforms.CenterCrop(224),
    ])
    '''
    transform = None

    for file in os.listdir(osp.join(args.data_root, args.targets_dir)):
        if video_name in file:
            video_name = file
            target_metrics = np.load(args.data_root, args.targets_dir, video_name)
            goodpoints_show_video_predictions(args,
                                              video_name,
                                              target_metrics,
                                              )
            break