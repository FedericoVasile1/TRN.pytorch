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
from torchvision import transforms

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

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    enc_score_metrics = []
    enc_target_metrics = []

    if osp.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint)
    else:
        raise(RuntimeError('Cannot find the checkpoint {}'.format(args.checkpoint)))
    model = build_model(args).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train(False)

    softmax = nn.Softmax(dim=1).to(device)

    if args.feature_extractor == 'RESNET2+1D':
        transform = transforms.Compose([
            transforms.Resize((112, 112)),  # preferable size for resnet(2+1)D model
            transforms.ToTensor(),
            transforms.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989])
        ])
    elif args.feature_extractor == 'VGG16':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        raise Exception('Wrong feature_extractor option')

    count_frames = 0
    for session_idx, session in enumerate(args.test_session_set, start=1):
        start = time.time()
        with torch.set_grad_enabled(False):
            target = np.load(osp.join(args.data_root, 'target_frames_24fps', session + '.npy'))
            # round to multiple of CHUNK_SIZE
            num_frames = target.shape[0]
            num_frames = num_frames - (num_frames % args.chunk_size)
            target = target[:num_frames]
            # For each chunk, take only the central frame
            target = target[args.chunk_size // 2::args.chunk_size]

            attn_weights = torch.zeros(target.shape[0], 1, 7, 7, dtype=torch.float32)
            for l in range(target.shape[0]):
                # load chunk/frame
                if args.feature_extractor == 'RESNET2+1D':
                    camera_input = torch.zeros(args.chunk_size, 3, 112, 112, dtype=torch.float32)
                    idx_central_frame = l * args.chunk_size + (args.chunk_size // 2)
                    start_f = idx_central_frame - args.chunk_size // 2
                    end_f = idx_central_frame + args.chunk_size // 2
                    for idx_frame in range(start_f, end_f):
                        frame = Image.open(
                            osp.join(args.data_root, args.model_input, session, str(idx_frame + 1) + '.jpg')).convert(
                            'RGB')
                        frame = transform(frame)
                        camera_input[idx_frame - start_f] = frame
                    camera_input = camera_input.permute(1, 0, 2, 3)
                elif args.feature_extractor == 'VGG16':
                    camera_input = torch.zeros(3, 224, 224, dtype=torch.float32)
                    idx_central_frame = l * args.chunk_size + (args.chunk_size // 2)
                    frame = Image.open(
                        osp.join(args.data_root, args.model_input, session, str(idx_central_frame + 1) + '.jpg')).convert(
                        'RGB')
                    camera_input = transform(frame)
                else:
                    raise Exception('Wrong feature_extractor option')

                camera_input = to_device(camera_input, device)

                if l % args.steps == 0:
                    enc_h_n = torch.zeros(1, model.hidden_size, device=device, dtype=camera_input.dtype)
                    enc_c_n = torch.zeros(1, model.hidden_size, device=device, dtype=camera_input.dtype)

                enc_score, enc_h_n, enc_c_n, attn_weights[l] = model.step(camera_input, enc_h_n, enc_c_n)

                enc_score_metrics.append(softmax(enc_score).cpu().numpy()[0])
                enc_target_metrics.append(target[l])

        end = time.time()

        print('Processed session {}, {:2} of {}, running time {:.2f} sec'.format(
            session, session_idx, len(args.test_session_set), end - start))

        if args.show_predictions:
            utl.show_video_predictions(args, target, session,
                                       enc_score_metrics[count_frames:count_frames + target.shape[0]],
                                       enc_target_metrics[count_frames:count_frames + target.shape[0]],
                                       attn_weights)
            count_frames += target.shape[0]

    save_dir = osp.dirname(args.checkpoint)
    result_file  = osp.basename(args.checkpoint).replace('.pth', '.json')
    # Compute result for encoder
    utl.compute_result_multilabel(args.class_index,
                                  enc_score_metrics, enc_target_metrics,
                                  save_dir, result_file, ignore_class=[0,21], save=True, verbose=True)

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

    args.class_index.pop(5)

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
                         index=[i for i in args.class_index],
                         columns=[i for i in args.class_index])
    fig = plt.figure(figsize=(26, 26))
    sn.heatmap(df_cm, annot=True, linewidths=.2)
    plt.ylabel('Actual class')
    plt.xlabel('Predicted class')
    writer.add_figure(timestamp + '_conf-mat_norm.jpg', fig)

    writer.close()

if __name__ == '__main__':
    main(parse_args())