import os
import os.path as osp
import sys
import time
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

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
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989])
    ])

    CHUNK_SIZE = 6

    softmax = nn.Softmax(dim=1).to(device)

    for session_idx, session in enumerate(args.test_session_set, start=1):
        start = time.time()
        with torch.set_grad_enabled(False):
            target = np.load(osp.join(args.data_root, 'target_frames_24fps', session+'.npy'))
            num_frames = target.shape[0]
            num_frames = num_frames - (num_frames % CHUNK_SIZE)
            target = target[:num_frames]
            target = target[CHUNK_SIZE//2::CHUNK_SIZE]

            # now, for the 'session' video, to the forward pass chunk by chunk till the end of the video
            for l in range(target.shape[0]):
                # load chunk (this chunk will be fed into the 3Dmodel and it will returns us a feature vector)
                camera_input = None     # chunk
                idx_central_frame = l * CHUNK_SIZE + (CHUNK_SIZE // 2)
                start_f = idx_central_frame - CHUNK_SIZE // 2
                end_f = idx_central_frame + CHUNK_SIZE // 2
                for idx_frame in range(start_f, end_f):
                    frame = Image.open(osp.join(args.data_root, args.camera_feature, session, str(idx_frame+1)+'.jpg')).convert('RGB')
                    frame = transform(frame).to(dtype=torch.float32)
                    if camera_input is None:
                        camera_input = torch.zeros(CHUNK_SIZE, frame.shape[0], frame.shape[1], frame.shape[2], dtype=torch.float32)
                    camera_input[idx_frame - start_f] = frame

                # same to what is done during the training pipeline, every enc_steps the encoder states will be zeroed
                if l % args.enc_steps == 0:
                    enc_h_n = torch.zeros(1, model.hidden_size_enc, device=device, dtype=camera_input.dtype)
                    enc_c_n = torch.zeros(1, model.hidden_size_enc, device=device, dtype=camera_input.dtype)

                camera_input = to_device(camera_input, device)
                enc_score, dec_scores, enc_h_n, enc_c_n = model.step(camera_input, enc_h_n, enc_c_n)

                enc_score_metrics.append(softmax(enc_score).cpu().numpy()[0])
                enc_target_metrics.append(target[l])

                for dec_step in range(args.dec_steps):
                    dec_score_metrics[dec_step].append(softmax(dec_scores[dec_step]).cpu().numpy()[0])
                    dec_target_metrics[dec_step].append(target[min(l + dec_step, target.shape[0] - 1)])
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
