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
from PIL import Image

import cv2

import _init_paths
import utils as utl
from configs.thumos import parse_trn_args as parse_args
from models import build_model

def to_device(x, device):
    return x.unsqueeze(0).to(device)

def add_pr_curve_tensorboard(writer, class_name, class_index, labels, probs_predicted, global_step=0):
    '''
    Takes in a "class_index" and plots the corresponding
    precision-recall curve
    '''
    # Labels from all classes must be binarize to the only label of the current class
    class_labels = labels == class_index
    # For each sample, take only the probability of the current class
    class_probs_predicted = probs_predicted[:, class_index]

    writer.add_pr_curve(class_name,
                        class_labels,
                        class_probs_predicted,
                        global_step=global_step)

def show_video_predictions(args, camera_inputs, session, enc_score_metrics):
    enc_pred_metrics = torch.max(torch.tensor(enc_score_metrics), 1)[1]

    for idx in range(camera_inputs.shape[0]):
        idx_frame = idx * 6 + 3  # because features are extracted by taking the central frame every 6 frames
        pil_frame = Image.open(osp.join(args.data_root, 'video_frames_24fps', session,
                                        str(idx_frame + 1) + '.jpg')).convert('RGB')
        open_cv_frame = np.array(pil_frame)
        # Convert RGB to BGR
        open_cv_frame = open_cv_frame[:, :, ::-1].copy()

        open_cv_frame = cv2.copyMakeBorder(open_cv_frame, 50,0,0,0, borderType=cv2.BORDER_CONSTANT, value=0)
        label = args.class_index[enc_pred_metrics[idx]]
        cv2.putText(open_cv_frame, label, (0, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)
        cv2.putText(open_cv_frame, str(idx_frame + 1), (180, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
        # [ (idx_frame + 1) / 24 ]    => 24 because frames has been extracted at 24 fps
        cv2.putText(open_cv_frame, '{:.2f}s'.format((idx_frame + 1) / 24), (250, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)

        # display the frame to screen
        cv2.imshow(session, open_cv_frame)
        key = cv2.waitKey(int(41.6*6))  # time is in milliseconds
        if key == ord('q'):
            # quit
            cv2.destroyAllWindows()
            break
        if key == ord('p'):
            # pause
            cv2.waitKey(-1)  # wait until any key is pressed

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

    count_frames = 0
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

        show_video_predictions(args, camera_inputs, session, enc_score_metrics[count_frames:count_frames + target.shape[0]])
        count_frames += target.shape[0]

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

    enc_score_metrics = np.array(enc_score_metrics)
    # Assign cliff diving (5) as diving (8)
    switch_index = np.where(enc_score_metrics[:, 5] > enc_score_metrics[:, 8])[0]
    enc_score_metrics[switch_index, 8] = enc_score_metrics[switch_index, 5]
    # Prepare variables
    enc_score_metrics = torch.tensor(enc_score_metrics)  # shape == (num_videos * num_frames_in_video, num_classes)
    enc_target_metrics = torch.max(torch.tensor(enc_target_metrics), 1)[1]  # shape == (num_videos * num_frames_in_video)

    # Log precision recall curve for encoder
    for idx_class in range(len(args.class_index)):
        if idx_class == 20 or idx_class == 5:
            continue  # ignore ambiguos class and cliff diving class
        add_pr_curve_tensorboard(writer, args.class_index[idx_class], idx_class,
                                 enc_target_metrics, enc_score_metrics)
    writer.close()

    # For each sample, takes the predicted class based on his scores
    enc_pred_metrics = torch.max(enc_score_metrics, 1)[1]

    # Log unnormalized confusion matrix for encoder
    conf_mat = confusion_matrix(enc_target_metrics, enc_pred_metrics)
    df_cm = pd.DataFrame(conf_mat,
                         index=[i for i in args.class_index],
                         columns=[i for i in args.class_index])
    fig = plt.figure(figsize=(26, 26))
    sn.heatmap(df_cm, annot=True, linewidths=.2, fmt="d")
    plt.ylabel('Actual class')
    plt.xlabel('Predicted class')
    timestamp = str(datetime.now())[:-7]
    writer.add_figure(timestamp+'_conf-mat_unnorm.jpg', fig)

    # Log normalized confusion matrix for encoder
    conf_mat_norm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(conf_mat_norm,
                         index=[i for i in args.class_index[:3]],
                         columns=[i for i in args.class_index[:3]])
    fig = plt.figure(figsize=(26, 26))
    sn.heatmap(df_cm, annot=True, linewidths=.2)
    plt.ylabel('Actual class')
    plt.xlabel('Predicted class')
    writer.add_figure(timestamp + '_conf-mat_norm.jpg', fig)

    writer.close()

if __name__ == '__main__':
    main(parse_args())
