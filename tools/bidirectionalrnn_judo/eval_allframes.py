import os
import shutil
import os.path as osp
import json
import sys
import time
import random
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())
from lib import utils as utl
from configs.judo import parse_model_args as parse_args
from lib.utils.visualize import plot_bar, plot_to_image, add_pr_curve_tensorboard, get_segments, show_video_predictions
from lib.models import build_model

def to_device(x, device):
    return x.unsqueeze(0).to(device)

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    score_metrics = []
    target_metrics = []

    if osp.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint)
    else:
        raise(RuntimeError('Cannot find the checkpoint {}'.format(args.checkpoint)))
    model = build_model(args).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train(False)

    if not args.show_predictions and not args.save_video:
        # i.e. we want to do a full evaluation of the test set and then compute stuff like confusion matrix, etc..
        tensorboard_dir = args.checkpoint.split('/')[:-1]
        eval_dir = osp.join(*tensorboard_dir, 'eval_allframes')
        if osp.isdir(eval_dir):
            shutil.rmtree(eval_dir)
        os.mkdir(eval_dir)
        writer = SummaryWriter(log_dir=eval_dir)
        logger = utl.setup_logger(osp.join(writer.log_dir, 'log.txt'))
        command = 'python ' + ' '.join(sys.argv)
        logger._write(command)

    softmax = nn.Softmax(dim=1).to(device)

    if args.eval_on_untrimmed:
        args.test_session_set = args.test_session_set['UNTRIMMED']
    else:
        if args.use_untrimmed == args.use_trimmed == True:
            args.test_session_set['TRIMMED'].extend(args.test_session_set['UNTRIMMED'])
            args.test_session_set = args.test_session_set['TRIMMED']
        elif args.use_trimmed:
            args.test_session_set = args.test_session_set['TRIMMED']
        elif args.use_untrimmed:
            args.test_session_set = args.test_session_set['UNTRIMMED']

    utl.set_seed(int(args.seed))
    random.shuffle(args.test_session_set)

    if args.video_name != '':
        args.test_session_set = [args.video_name]
    if args.save_video:
        # when this option is activated, we evaluate and save ONLY ONE video, without showing it.
        args.test_session_set = args.test_session_set[:1]
        args.show_predictions = False
    if args.show_predictions:
        count_frames = 0

    for session_idx, session in enumerate(args.test_session_set, start=1):
        if 'Grand Prix' in session:
            dataset_type = 'UNTRIMMED'
        elif 'GoPro' in session:
            dataset_type = 'TRIMMED'
        else:
            raise Exception('Unknown video name: ' + session)
        if not osp.isfile(osp.join(args.data_root, dataset_type, args.model_target, session + '.npy')):
            # skip videos in which the pose model does not detect any fall(i.e. fall==-1  in fall_detections.csv).
            # TODO: fix these videos later on, in order to incluso also them
            continue

        start = time.time()
        with torch.set_grad_enabled(False):
            original_target = np.load(osp.join(args.data_root, dataset_type, args.model_target, session + '.npy'))
            # round to multiple of CHUNK_SIZE
            num_frames = original_target.shape[0]
            num_frames = num_frames - (num_frames % args.chunk_size)
            original_target = original_target[:num_frames]
            # For each chunk, take only the central frame
            target = original_target[args.chunk_size // 2::args.chunk_size]

            features_extracted = np.load(osp.join(args.data_root, dataset_type, args.model_input, session + '.npy'),
                                         mmap_mode='r')
            features_extracted = torch.as_tensor(features_extracted.astype(np.float32))

            samples = []
            for count in range(target.shape[0]):
                samples.append(features_extracted[count])

                if count % args.steps == 0 and count != 0:
                    samples = torch.stack(samples).unsqueeze(0).to(device)
                    scores = model(samples)         # scores.shape == (1, steps, num_classes)

                    scores = scores.squeeze(0)
                    scores = softmax(scores).cpu().detach().numpy()
                    for s in range(len(scores)):
                        for c in range(args.chunk_size):
                            score_metrics.append(scores[s])
                    samples = []

                for c in range(args.chunk_size):
                    target_metrics.append(original_target[count * args.chunk_size + c])

            if samples != []:
                appo = len(samples)
                for i in range(appo, args.steps):
                    samples.append(torch.zeros_like(features_extracted[0]))
                samples = torch.stack(samples).unsqueeze(0).to(device)
                scores = model(samples)
                scores = scores[:, :appo]
                scores = scores.squeeze(0)
                scores = softmax(scores).cpu().detach().numpy()
                for s in range(len(scores)):
                    for c in range(args.chunk_size):
                        score_metrics.append(scores[s])

        end = time.time()

        print('Processed session {}, {:2} of {}, running time {:.2f} sec'.format(session,
                                                                                 session_idx,
                                                                                 len(args.test_session_set),
                                                                                 end - start))
        if args.show_predictions:
            appo = args.chunk_size
            args.chunk_size = 1
            appo2 = args.data_root
            args.data_root = args.data_root + '/' + dataset_type
            show_video_predictions(args,
                                   session,
                                   target_metrics[count_frames:count_frames + original_target.shape[0]],
                                   score_metrics[count_frames:count_frames + original_target.shape[0]],
                                   frames_dir='video_frames_25fps',
                                   fps=25)
            args.chunk_size = appo
            args.data_root = appo2
            count_frames += original_target.shape[0]

    if args.save_video:
        # here the video will be saved
        args.chunk_size = 1
        appo = args.data_root
        args.data_root = args.data_root + '/' + dataset_type
        show_video_predictions(args,
                               session,
                               target_metrics,
                               score_metrics,
                               frames_dir='video_frames_25fps',
                               fps=25)
        args.data_root = appo
        # print some stats about the video labels and predictions, then kill the program
        print('\n=== LABEL SEGMENTS ===')
        segments_list = get_segments(target_metrics, args.class_index, 25, args.chunk_size)
        print(segments_list)
        print('\n=== SCORE SEGMENTS ===')
        segments_list = get_segments(score_metrics, args.class_index, 25, args.chunk_size)
        print(segments_list)
        print('\n=== RESULTS ===')
        utl.compute_result_multilabel(args.dataset,
                                      args.class_index,
                                      score_metrics,
                                      target_metrics,
                                      save_dir=None,
                                      result_file=None,
                                      save=False,
                                      ignore_class=[0],
                                      return_APs=False,
                                      samples_all_valid=True,
                                      verbose=True, )
        return

    result = utl.compute_result_multilabel(args.dataset,
                                           args.class_index,
                                           score_metrics,
                                           target_metrics,
                                           save_dir=None,
                                           result_file=None,
                                           save=False,
                                           ignore_class=[0],
                                           return_APs=True,
                                           samples_all_valid=True,
                                           verbose=True, )
    logger._write(json.dumps(result, indent=2))

    per_class_ap = {}
    for cls in range(args.num_classes):
        #if cls == 0:
        #    # ignore background class
        #    continue
        per_class_ap[args.class_index[cls]] = round(result['AP'][args.class_index[cls]], 2)
    figure = plot_bar(per_class_ap.keys(), list(per_class_ap.values()), title=args.dataset + ': per-class AP')
    figure = plot_to_image(figure)
    writer.add_image(args.dataset + ': per-class AP', np.transpose(figure, (2, 0, 1)), 0)
    writer.close()

    # Prepare variables
    score_metrics = torch.tensor(score_metrics)  # shape == (num_videos * num_frames_in_video, num_classes)
    target_metrics = torch.tensor(target_metrics).argmax(dim=1)  # shape == (num_videos * num_frames_in_video)

    # Log precision recall curve for encoder
    for idx_class in range(len(args.class_index)):
        add_pr_curve_tensorboard(writer,
                                 args.class_index[idx_class],
                                 idx_class,
                                 target_metrics,
                                 score_metrics)
    writer.close()

    # For each sample, takes the predicted class based on his scores
    pred_metrics = score_metrics.argmax(dim=1)

    result = classification_report(target_metrics, pred_metrics, target_names=args.class_index, output_dict=True)
    logger._write(json.dumps(result, indent=2))

    # Log unnormalized confusion matrix for encoder
    conf_mat = confusion_matrix(target_metrics, pred_metrics)
    df_cm = pd.DataFrame(conf_mat,
                         index=[i for i in args.class_index],
                         columns=[i for i in args.class_index])
    fig = plt.figure(figsize=(6, 6))
    sn.heatmap(df_cm, annot=True, linewidths=.2, fmt="d")
    plt.ylabel('Actual class')
    plt.xlabel('Predicted class')
    plt.yticks(rotation=45)
    plt.xticks(rotation=45)
    plt.tight_layout()
    writer.add_figure('eval-allframes_judo_conf-mat_unnorm.jpg', fig)

    # Log normalized confusion matrix for encoder
    conf_mat_norm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(conf_mat_norm,
                         index=[i for i in args.class_index],
                         columns=[i for i in args.class_index])
    fig = plt.figure(figsize=(6, 6))
    sn.heatmap(df_cm, annot=True, linewidths=.2)
    plt.ylabel('Actual class')
    plt.xlabel('Predicted class')
    plt.yticks(rotation=45)
    plt.xticks(rotation=45)
    plt.tight_layout()
    writer.add_figure('eval-allframes_judo_conf-mat_norm.jpg', fig)

    writer.close()

if __name__ == '__main__':
    base_dir = os.getcwd()
    base_dir = base_dir.split('/')[-1]
    CORRECT_LAUNCH_DIR = 'TRN.pytorch'
    if base_dir != CORRECT_LAUNCH_DIR:
        raise Exception('Wrong base dir, this file must be run from ' + CORRECT_LAUNCH_DIR + ' directory.')

    main(parse_args())