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
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

sys.path.append(os.getcwd())
import _init_paths
import utils as utl
from lib.utils.visualize import show_video_predictions
from configs.judo import parse_trn_args as parse_args
from lib.utils.visualize import plot_perclassap_bar, plot_to_image, add_pr_curve_tensorboard
from models import build_model
from lib.datasets.thumos_data_layer_e2e import I3DNormalization

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

    if args.feature_extractor == 'I3D':
        transform = transforms.Compose([
            transforms.Resize((224, 320)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            I3DNormalization(),
        ])
    elif args.feature_extractor == 'RESNET2+1D':
        transform = transforms.Compose([
            transforms.Resize((224, 320)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989])
        ])
    else:
        raise Exception('Wrong feature_extractor option, ' + args.feature_extractor + ' is not supported')

    if args.show_predictions:
        count_frames = 0
        if args.seed_show_predictions != -1:
            random.seed(args.seed_show_predictions)
        random.shuffle(args.test_session_set)
    for session_idx, session in enumerate(args.test_session_set, start=1):
        start = time.time()
        with torch.set_grad_enabled(False):
            original_target = np.load(osp.join(args.data_root, 'target_frames_25fps', session + '.npy'))
            # round to multiple of CHUNK_SIZE
            num_frames = original_target.shape[0]
            num_frames = num_frames - (num_frames % args.chunk_size)
            original_target = original_target[:num_frames]
            # For each chunk, take only the central frame
            target = original_target[args.chunk_size // 2::args.chunk_size]

            batch_samples = None
            for count in range(target.shape[0]):
                idx_central_frame = count * args.chunk_size + (args.chunk_size // 2)
                start_f = idx_central_frame - args.chunk_size // 2
                end_f = idx_central_frame + args.chunk_size // 2
                for idx_frame in range(start_f, end_f):
                    frame = Image.open(osp.join(args.data_root,
                                                args.model_input,
                                                session,
                                                str(idx_frame + 1) + '.jpg')).convert('RGB')
                    frame = transform(frame)

                    if batch_samples is None:
                        batch_samples = torch.zeros(args.batch_size,
                                                    args.chunk_size,
                                                    frame.shape[0],
                                                    frame.shape[1],
                                                    frame.shape[2],
                                                    dtype=torch.float32)
                    batch_samples[count % args.batch_size, idx_frame - start_f] = frame

                if count % args.batch_size == args.batch_size - 1:
                    # forward pass
                    batch_samples = batch_samples.permute(0, 2, 1, 3, 4)
                    batch_samples = batch_samples.to(device)
                    scores = model(batch_samples)

                    scores = softmax(scores).cpu().detach().numpy()
                    for i in range(scores.shape[0]):
                        for c in range(args.chunk_size):
                            enc_score_metrics.append(scores[i])
                            enc_target_metrics.append(
                                original_target[((count+1) - args.batch_size + i) * args.chunk_size + c])

                    batch_samples = None
        # do the last forward pass, because there will probably be the last batch with samples < batch_size
        if batch_samples is not None:
            # forward pass
            batch_samples = batch_samples.permute(0, 2, 1, 3, 4)
            batch_samples = batch_samples.to(device)
            scores = model(batch_samples)

            scores = softmax(scores).cpu().detach().numpy()
            for i in range(scores.shape[0]):
                for c in range(args.chunk_size):
                    enc_score_metrics.append(scores[i])
                    enc_target_metrics.append(
                        original_target[((count + 1) - args.batch_size + i) * args.chunk_size + c])

        end = time.time()

        print('Processed session {}, {:2} of {}, running time {:.2f} sec'.format(session,
                                                                                 session_idx,
                                                                                 len(args.test_session_set),
                                                                                 end - start))
        if args.show_predictions:
            appo = args.chunk_size
            args.chunk_size = 1
            show_video_predictions(args,
                                   session,
                                   enc_target_metrics[count_frames:count_frames + original_target.shape[0]],
                                   enc_score_metrics[count_frames:count_frames + original_target.shape[0]],
                                   frames_dir='video_frames_25fps',
                                   fps=25)
            args.chunk_size = appo
            count_frames += original_target.shape[0]

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