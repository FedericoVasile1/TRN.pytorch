'''
PYTHONPATH=/Users/federicovasile/Documents/Tirocinio/trn_repo/TRN.pytorch python tools/trn2_thumos/eval.py --epochs 1 --enc_steps 8 --dec_steps 2 --hidden_size 16 --neurons 8 --feat_vect_dim 512 --data_info data/small_data_info.json --model TRN2V2 --checkpoint tools/trn2_thumos/checkpoints/inputs-camera-epoch-1.pth
'''

import os
import os.path as osp
import sys
import time
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import _init_paths
import utils as utl
from configs.thumos import parse_trn_args as parse_args
from models import build_model

def to_device(x, device):
    return x.unsqueeze(0).to(device)

def add_pr_curve_tensorboard(writer, class_name, class_index, test_probs, test_preds, global_step=0):
    '''
    Takes in a "class_index" and plots the corresponding
    precision-recall curve
    '''
    tensorboard_preds = test_preds == class_index
    tensorboard_probs = test_probs[:, class_index]

    writer.add_pr_curve(class_name,
                        tensorboard_preds,
                        tensorboard_probs,
                        global_step=global_step)

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

    for session_idx, session in enumerate(args.test_session_set, start=1):
        start = time.time()
        with torch.set_grad_enabled(False):
            camera_inputs = np.load(osp.join(args.data_root, args.camera_feature, session+'.npy'), mmap_mode='r')
            camera_inputs = torch.as_tensor(camera_inputs.astype(np.float32))
            target = np.load(osp.join(args.data_root, 'target', session+'.npy'))

            for l in range(target.shape[0]):
                if l % args.enc_steps == 0:
                    enc_h_n = torch.zeros(1, model.hidden_size_enc, device=device, dtype=camera_inputs.dtype)
                    enc_c_n = torch.zeros(1, model.hidden_size_enc, device=device, dtype=camera_inputs.dtype)

                camera_input = to_device(camera_inputs[l], device)
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

    writer = SummaryWriter()
    # Log precision recall curve for encoder
    enc_score_metrics = torch.tensor(enc_score_metrics)    # shape == (num_videos * num_frames_in_video, num_classes)
    enc_pred_metrics = torch.max(enc_score_metrics, 1)[1]
    for idx_class in range(len(args.class_index)):
        if idx_class == 21:
            continue  # ignore ambiguos class
        add_pr_curve_tensorboard(writer, args.class_index[idx_class], idx_class,
                                 enc_score_metrics, enc_pred_metrics)
    writer.close()

    # Log confusion matrix for encoder
    enc_target_metrics = torch.max(torch.tensor(enc_target_metrics), 1)[1]
    conf_mat = confusion_matrix(enc_pred_metrics, enc_target_metrics)
    args.class_index.pop(5)
    df_cm = pd.DataFrame(conf_mat,
                         index=[i for i in args.class_index],
                         columns=[i for i in args.class_index])
    fig = plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, linewidths=.2)
    plt.xlabel('Actual class')
    plt.ylabel('Predicted class')
    timestamp = str(datetime.now())[:-7]
    writer.add_figure(timestamp+'_conf-mat.jpg', fig)
    writer.close()

if __name__ == '__main__':
    main(parse_args())
