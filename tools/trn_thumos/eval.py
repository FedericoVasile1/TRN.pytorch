import os
import os.path as osp
import sys
import time

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

import numpy as np

import _init_paths
import utils as utl
from configs.thumos import parse_trn_args as parse_args
from models import build_model

def to_device(x, device):
    return x.unsqueeze(0).to(device)

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    enc_score_metrics = []
    enc_target_metrics = []
    dec_score_metrics = [[] for i in range(args.dec_steps)]
    dec_target_metrics = [[] for i in range(args.dec_steps)]

    if osp.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint)
    else:
        raise(RuntimeError('Cannot find the checkpoint {}'.format(args.checkpoint)))
    model = build_model(args).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train(False)

    softmax = nn.Softmax(dim=1).to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    for session_idx, session in enumerate(args.test_session_set, start=1):
        start = time.time()
        with torch.set_grad_enabled(False):
            idx_to_frame = {}
            name_frames = os.listdir(osp.join(args.data_root, args.camera_frames_root, session))
            for name_frame in name_frames:
                idx_frame = int(name_frame[:-4])-1        # e.g. name_frame == '7.jpg'
                if args.sample_frames != 0:
                    if not (idx_frame == args.sample_frames//2 or (idx_frame-args.sample_frames//2) % 6 == 0):
                        continue

                frame = Image.open(osp.join(args.data_root, args.camera_frames_root, session, name_frame))
                frame = transform(frame).unsqueeze(0)
                idx_to_frame[int(name_frame[:-4])] = frame

            target = np.load(osp.join(args.data_root, args.camera_target_root, session+'.npy'))
            if args.sample_frames != 0:
                target = target[args.sample_frames//2::args.sample_frames]
            future_input = to_device(torch.zeros(model.future_size), device)
            enc_hx = to_device(torch.zeros(model.hidden_size), device)
            enc_cx = to_device(torch.zeros(model.hidden_size), device)

            for l in range(target.shape[0]):
                appo = l
                if args.sample_frames != 0:
                    appo = (appo * args.sample_frames) + args.sample_frames // 2

                future_input, enc_hx, enc_cx, enc_score, dec_score_stack = \
                        model.step(idx_to_frame[appo+1], future_input, enc_hx, enc_cx)

                enc_score_metrics.append(softmax(enc_score).cpu().numpy()[0])
                enc_target_metrics.append(target[l])

                for step in range(args.dec_steps):
                    dec_score_metrics[step].append(softmax(dec_score_stack[step]).cpu().numpy()[0])
                    dec_target_metrics[step].append(target[min(l + step, target.shape[0] - 1)])
        end = time.time()

        print('Processed session {}, {:2} of {}, running time {:.2f} sec'.format(
            session, session_idx, len(args.test_session_set), end - start))

    save_dir = osp.dirname(args.checkpoint)
    result_file  = osp.basename(args.checkpoint).replace('.pth', '.json')
    # Compute result for encoder
    utl.compute_result_multilabel(args.class_index,
                                  enc_score_metrics, enc_target_metrics,
                                  None, None, None, None, ignore_class=[0,21], save=False, verbose=True)

    # Compute result for decoder
    for step in range(args.dec_steps):
        utl.compute_result_multilabel(args.class_index,
                                      dec_score_metrics[step], dec_target_metrics[step],
                                      None, None, None, None, ignore_class=[0,21], save=False, verbose=True)

if __name__ == '__main__':
    main(parse_args())
