import os
import shutil
import os.path as osp
import sys
import time
import random
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import cv2

sys.path.append(os.getcwd())
from lib import utils as utl
from configs.judo import parse_model_args as parse_args
from lib.models import build_model

def to_device(x, device):
    return x.unsqueeze(0).to(device)

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if osp.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint)
    else:
        raise(RuntimeError('Cannot find the checkpoint {}'.format(args.checkpoint)))
    model = build_model(args).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train(False)

    transform = transforms.Compose([
        transforms.Resize((227, 324)),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        # transforms.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989])       # TODO ????
    ])

    utl.set_seed(int(args.seed))
    if args.video_name != '':
        args.test_session_set = [args.video_name]
    else:
        args.test_session_set = args.test_session_set['UNTRIMMED']
        random.shuffle(args.test_session_set)
        args.test_session_set = args.test_session_set[:1]

    for session_idx, session in enumerate(args.test_session_set, start=1):
        if 'Grand Prix' in session:
            dataset_type = 'UNTRIMMED'
        else:
            raise Exception('Unknown video name: ' + session)

        start = time.time()
        with torch.set_grad_enabled(False):
            target = np.load(osp.join(args.data_root, dataset_type, '10s_target_frames_25fps', session + '.npy'))
            os.mkdir(os.getcwd(), session[:-4])

            frames = []
            for idx in range(len(target)):
                if idx == 0:
                    continue

                if target[idx, 0] == 1:
                    if target[idx-1, 0] == 0:
                        errors = []
                        tensor_frames = torch.stack(frames)
                        for i in range(len(tensor_frames) - args.steps + 1):
                            clip = []
                            for j in range(args.steps):
                                clip.append(tensor_frames[i+j])
                            clip = torch.stack(clip)

                            clip = to_device(clip, device)
                            outputs = model(clip)

                            clip = clip.flatten(start_dim=1)
                            outputs = outputs.flatten(start_dim=1).to(device)
                            s = torch.norm(clip - outputs, dim=1)       # scalar
                            errors.append(s)

                        errors = torch.cat(errors).view(-1, len(tensor_frames)-args.steps+1).numpy()
                        appo = 1 - (errors[0,:] - np.min(errors[0,:]))/(np.max(errors[0,:]) - np.min(errors[0,:]))

                        min_pos = np.argmin(errors[0])

                        start_millisecond = (idx-len(frames)) / 25 * 1000

                        plt.plot(appo)
                        plt.savefig(osp.join(os.getcwd(), session[:-4]), str(start_millisecond)+'.jpg')

                        H, W, _ = frames[0].shape
                        out = cv2.VideoWriter(osp.join(os.getcwd(), session[:-4], str(start_millisecond)+'.mp4'),
                                              cv2.VideoWriter_fourcc(*'mp4v'),
                                              25.0,
                                              (W, H))
                        for k, frame in enumerate(frames, start=0):
                            frame = np.array(frame)
                            frame = frame[:, :, ::-1].copy()
                            frame = cv2.putText(frame,
                                                'Score: '+str(errors[0, k]),
                                                (0, 30),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                1,
                                                (0, 0, 255),
                                                2)
                            if k < min_pos and k > min_pos - 50:
                                frame = cv2.putText(frame,
                                                    'Action',
                                                    (0, 60),
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    1,
                                                    (0, 0, 255),
                                                    2)
                            out.write(frame)
                        out.release()

                        frames = []
                    else:
                        frames = []
                else:
                        frame = Image.open(osp.join(args.data_root,
                                                    dataset_type,
                                                    args.model_input,
                                                    session,
                                                    str(idx+1)+'.jpg')).convert('RGB')
                        frame = transform(frame)
                        frames.append(frame)

        end = time.time()

        print('Processed session {}, {:2} of {}, running time {:.2f} sec'.format(session,
                                                                                 session_idx,
                                                                                 len(args.test_session_set),
                                                                                 end - start))

if __name__ == '__main__':
    main(parse_args())