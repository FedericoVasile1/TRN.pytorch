import os
import os.path as osp
import sys
import time

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
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

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    softmax = nn.Softmax(dim=1).to(device)

    for session_idx, session in enumerate(args.test_session_set, start=1):
        start = time.time()
        with torch.set_grad_enabled(False):
            camera_inputs = None
            num_frames = len(os.listdir(osp.join(args.data_root, args.camera_feature, session)))
            # round to multiple of 6
            num_frames = num_frames - (num_frames % 6)
            for idx_frame in range(3, num_frames, 6):
                frame = Image.open(osp.join(args.data_root, args.camera_feature, session, str(idx_frame+1)+'.jpg'))
                frame = transform(frame)
                if camera_inputs is None:
                    camera_inputs = torch.zeros(num_frames//6, frame.shape[0], frame.shape[1], frame.shape[2], dtype=torch.float32)
                camera_inputs[(idx_frame-3)//6] = frame.to(dtype=torch.float32)

            #motion_inputs = np.load(osp.join(args.data_root, args.motion_feature, session+'.npy'), mmap_mode='r')
            motion_inputs = np.zeros((num_frames))   # optical flow will not be used
            # take target frames and round to multiple of 6 ( -> [:num_frames])
            target = np.load(osp.join(args.data_root, 'target_frames_24fps', session+'.npy'))[:num_frames]
            target = target[3::6]
            future_input = to_device(torch.zeros(model.future_size), device)
            enc_hx = to_device(torch.zeros(model.hidden_size), device)
            enc_cx = to_device(torch.zeros(model.hidden_size), device)

            for l in range(target.shape[0]):
                camera_input = to_device(
                    camera_inputs[l], device)
                motion_input = to_device(
                    torch.as_tensor(motion_inputs[l].astype(np.float32)), device)

                future_input, enc_hx, enc_cx, enc_score, dec_score_stack = \
                        model.step(camera_input, motion_input, future_input, enc_hx, enc_cx)

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
                                  save_dir, result_file, ignore_class=[0,21], save=True, verbose=True)

    # Compute result for decoder
    for step in range(args.dec_steps):
        utl.compute_result_multilabel(args.class_index,
                                      dec_score_metrics[step], dec_target_metrics[step],
                                      save_dir, result_file, ignore_class=[0,21], save=False, verbose=True)

if __name__ == '__main__':
    main(parse_args())
