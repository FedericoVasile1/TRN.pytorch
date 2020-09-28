import os
import os.path as osp
import sys
import time
import random
import json
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

sys.path.append(os.getcwd())
import _init_paths
import utils as utl
from lib.utils.visualize import show_video_predictions
from configs.judo import parse_trn_args as parse_args
from lib.utils.visualize import plot_perclassap_bar, plot_to_image, add_pr_curve_tensorboard
from models import build_model

def to_device(x, device):
    return x.unsqueeze(0).to(device)

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

    if args.camera_feature == 'i3d_224x224' and args.chunk_size != 9:
        raise Exception('Wrong pair of arguments. Regarding --camera_feature == i3d_224x224 we actually'
                        'only have features extracted with chunk of 9, so put --chunk_size == 9')
    if args.camera_feature == 'resnet2+1d_224x224' and args.chunk_size != 6:
        raise Exception('Wrong pair of arguments. Regarding --camera_feature == resnet2+1d_224x224 we actually'
                        'only have features extracted with chunk of 6, so put --chunk_size == 6')
    if args.camera_feature not in ('i3d_224x224', 'resnet2+1d_224x224'):
        raise Exception('Wrong --camera_feature option. Supported values: [i3d_224x224|resnet2+1d_224x224]')

    if args.show_predictions:
        count_frames = 0
        if args.seed_show_predictions != -1:
            random.seed(args.seed_show_predictions)
        random.shuffle(args.test_session_set)
    for session_idx, session in enumerate(args.test_session_set, start=1):
        start = time.time()
        with torch.set_grad_enabled(False):
            target = np.load(osp.join(args.data_root, 'target_frames_25fps', session + '.npy'))
            # round to multiple of CHUNK_SIZE
            num_frames = target.shape[0]
            num_frames = num_frames - (num_frames % args.chunk_size)
            target = target[:num_frames]
            # For each chunk, take only the central frame
            target = target[args.chunk_size // 2::args.chunk_size]

            features_extracted = np.load(osp.join(args.data_root, args.camera_feature, session+'.npy'), mmap_mode='r')
            features_extracted = torch.as_tensor(features_extracted.astype(np.float32))

            for count in range(target.shape[0]):
                if count % args.enc_steps == 0:
                    h_n = to_device(torch.zeros(model.hidden_size), device)
                    c_n = to_device(torch.zeros(model.hidden_size), device)

                sample = to_device(features_extracted[count], device)
                scores, h_n, c_n = model.step(sample, torch.zeros(args.enc_steps, 1), h_n, c_n)

                scores = softmax(scores).cpu().detach().numpy()[0]
                enc_score_metrics.append(scores)
                enc_target_metrics.append(target[count])

        end = time.time()

        print('Processed session {}, {:2} of {}, running time {:.2f} sec'.format(session,
                                                                                 session_idx,
                                                                                 len(args.test_session_set),
                                                                                 end - start))
        if args.show_predictions:
            show_video_predictions(args,
                                   session,
                                   enc_score_metrics[count_frames:count_frames + target.shape[0]],
                                   enc_target_metrics[count_frames:count_frames + target.shape[0]],
                                   frames_dir=args.camera_feature,
                                   fps=25)
            count_frames += target.shape[0]

    save_dir = osp.dirname(args.checkpoint)
    result_file  = osp.basename(args.checkpoint).replace('.pth', '.json')
    # Compute result for encoder
    utl.compute_result_multilabel(args.class_index,
                                  enc_score_metrics,
                                  enc_target_metrics,
                                  save_dir,
                                  result_file,
                                  ignore_class=[0],
                                  save=True,
                                  switch=False,
                                  verbose=True)

    writer = SummaryWriter()

    # get per class AP: the function utl.compute_result_multilabel(be sure that it has the parameter save == True) has
    #  stored a JSON file containing per class AP: we load this json into a dict and add an histogram to tensorboard
    with open(osp.join(save_dir, result_file), 'r') as f:
        per_class_ap = json.load(f)['AP']
    for class_name in per_class_ap:
        per_class_ap[class_name] = round(per_class_ap[class_name], 2)
    figure = plot_perclassap_bar(per_class_ap.keys(), per_class_ap.values(), title=args.dataset + ': per-class AP')
    figure = plot_to_image(figure)
    writer.add_image(args.dataset + ': per-class AP', np.transpose(figure, (2, 0, 1)), 0)
    writer.close()

    enc_score_metrics = np.array(enc_score_metrics)
    # Prepare variables
    enc_score_metrics = torch.tensor(enc_score_metrics)  # shape == (num_videos * num_frames_in_video, num_classes)
    enc_target_metrics = torch.max(torch.tensor(enc_target_metrics), 1)[1]  # shape == (num_videos * num_frames_in_video)

    # Log precision recall curve for encoder
    for idx_class in range(len(args.class_index)):
        add_pr_curve_tensorboard(writer,
                                 args.class_index[idx_class],
                                 idx_class,
                                 enc_target_metrics,
                                 enc_score_metrics)
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