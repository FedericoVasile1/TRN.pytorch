import random

import numpy as np
import os.path as osp
import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torchvision import transforms
import cv2
from PIL import Image

from datasets import build_dataset

__all__ = [
    'set_seed',
    'build_data_loader',
    'weights_init',
    'count_parameters',
    'show_video_predictions',
    'soft_argmax',
]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def build_data_loader(args, phase='train'):
    data_loaders = data.DataLoader(
        dataset=build_dataset(args, phase),
        batch_size=args.batch_size,
        shuffle=phase=='train',
        num_workers=args.num_workers,
    )
    return data_loaders

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        #m.weight.data.normal_(0.0, 0.001)
        pass
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, 0.001)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def show_video_predictions(args,
                           video_name,
                           enc_target_metrics,
                           enc_score_metrics=None,
                           attn_weights=None,
                           frames_dir='video_frames_24fps',
                           fps=24):
    '''
    :param args: ParserArgument object containing main arguments
    :param video_name: string containing the name of the video
    :param enc_target_metrics: numpy array of shape(num_frames, num_classes) containing the ground truth of the video
    :param enc_score_metrics: numpy array of shape(num_frames, num_classes) containing the output scores of the model
    :param attn_weights:
    :param frames_dir: string containing the base folder name, i.e. under the base folder there will be
                        one folder(i.e. video_name) for each video, this will contains the frames of that video
    :return:
    '''
    if enc_score_metrics is not None:
        enc_pred_metrics = torch.argmax(torch.tensor(enc_score_metrics), dim=1)
    enc_target_metrics = torch.argmax(torch.tensor(enc_target_metrics), dim=1)

    num_frames = enc_target_metrics.shape[0]
    idx = 0
    #for idx in range(num_frames):
    while idx < num_frames:
        idx_frame = idx * args.chunk_size + args.chunk_size // 2
        pil_frame = Image.open(osp.join(args.data_root,
                                        frames_dir,
                                        video_name,
                                        str(idx_frame + 1) + '.jpg')).convert('RGB')
        open_cv_frame = np.array(pil_frame)

        # Convert RGB to BGR
        open_cv_frame = open_cv_frame[:, :, ::-1].copy()

        if args.dataset == 'JUDO':
            H, W, _ = open_cv_frame.shape
            open_cv_frame = cv2.resize(open_cv_frame, (W // 2, H // 2), interpolation=cv2.INTER_AREA)

        if attn_weights is not None:
            original_H, original_W, _ = open_cv_frame.shape
            if args.feature_extractor == 'VGG16':
                open_cv_frame = cv2.resize(open_cv_frame, (224, 224), interpolation=cv2.INTER_AREA)
            else:
                # RESNET2+1D feature extraction
                open_cv_frame = cv2.resize(open_cv_frame, (112, 112), interpolation=cv2.INTER_AREA)

            H, W, _ = open_cv_frame.shape

            attn_weights_t = attn_weights[idx]
            attn_weights_t = attn_weights_t.squeeze(0)
            attn_weights_t = cv2.resize(attn_weights_t.data.numpy().copy(), (W, H), interpolation=cv2.INTER_NEAREST)
            attn_weights_t = np.repeat(np.expand_dims(attn_weights_t, axis=2), 3, axis=2)
            attn_weights_t *= 255
            attn_weights_t = attn_weights_t.astype('uint8')
            # mask original image according to the attention weights
            open_cv_frame = cv2.addWeighted(attn_weights_t, 0.5, open_cv_frame, 0.5, 0)

            open_cv_frame = cv2.resize(open_cv_frame, (original_W, original_H), interpolation=cv2.INTER_AREA)

        open_cv_frame = cv2.copyMakeBorder(open_cv_frame, 60, 0, 20, 20, borderType=cv2.BORDER_CONSTANT, value=0)
        pred_label = args.class_index[enc_pred_metrics[idx]] if enc_score_metrics is not None else 'junk'
        target_label = args.class_index[enc_target_metrics[idx]]

        cv2.putText(open_cv_frame,
                    pred_label,
                    (0, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 0) if enc_score_metrics is None else (0, 255, 0) if pred_label == target_label else (0, 0, 255),
                    1)
        cv2.putText(open_cv_frame,
                    target_label,
                    (0, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255) if enc_score_metrics is not None else
                    (255, 255, 255) if target_label == 'Background' else (0, 0, 255),
                    1)
        cv2.putText(open_cv_frame,
                    'prob:{:.2f}'.format(
                        torch.tensor(enc_score_metrics)[idx, enc_pred_metrics[idx]].item()
                    ) if enc_score_metrics is not None else 'junk',
                    (210, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0) if enc_score_metrics is None else (0, 255, 0) if pred_label == target_label else (0, 0, 255),
                    1)

        # [ (idx_frame + 1) / 24 ]    => 24 because frames has been extracted at 24 fps
        cv2.putText(open_cv_frame,
                    '{:.2f}s'.format((idx_frame + 1) / fps),
                    (275, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 255, 255),
                    1)
        cv2.putText(open_cv_frame,
                    'speed: {}x'.format(args.speed),
                    (275, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 255, 255),
                    1)
        cv2.putText(open_cv_frame,
                    str(idx_frame + 1),
                    (275, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 255, 255),
                    1)

        # display the frame to screen
        cv2.imshow(video_name, open_cv_frame)
        # e.g. since video are extracted at 24 fps, we will display a frame every 1000[ms] / 24[f] = 41.6[ms]
        delay = 1000 / fps
        # since in our model we do not take all of the 24 frames, but only the central frame every chunk_size frames
        delay *= args.chunk_size
        # depending on the speed the video will be displayed faster(e.g. 2x if args.speed == 2.0) or slower
        delay /= args.speed
        key = cv2.waitKey(int(delay))  # time is in milliseconds
        if key == ord('q'):
            # quit
            cv2.destroyAllWindows()
            break
        if key == ord('p'):
            # pause
            cv2.waitKey(-1)  # wait until any key is pressed
        if key == ord('e'):
            delay /= 2
            args.speed *= 2
        if key == ord('w'):
            delay *= 2
            args.speed /= 2
        if key == ord('a'):
            idx -= fps
            if idx < 0:
                idx = 0
        if key == ord('s'):
            idx += fps

        idx += 1

def show_random_videos(args,
                       samples_list,
                       samples=1,
                       frames_dir='video_frames_24fps',
                       targets_dir='target_frames_24fps',
                       fps=24):
    '''
    It shows samples videos from samples_list , randomly sampled. Furthermore, labels are attached to video.
    :param args: ParserArgument object containing main arguments
    :param video_name_list: a list containing the video names from which to sample a video
    :param samples: int representing the number of video to show for each class
    :param frames_dir: string containing the base folder name, i.e. under the base folder there will be
                        one folder(i.e. video_name) for each video, this will contains the frames of that video
    :return:
    '''
    num_samples = len(samples_list)
    idx_samples = np.random.choice(num_samples, size=samples, replace=False)
    for i in idx_samples:
        video_name = samples_list[i]

        target = np.load(osp.join(args.data_root, targets_dir, video_name + '.npy'))
        num_frames = target.shape[0]
        num_frames = num_frames - (num_frames % args.chunk_size)
        target = target[:num_frames]
        target = target[args.chunk_size // 2::args.chunk_size]

        print(video_name)
        show_video_predictions(args, video_name, target, frames_dir=frames_dir, fps=fps)

def soft_argmax(scores):
    # scores.shape == (batch_size, num_classes).   scores are NOT passed through softmax
    softmax = F.softmax(scores, dim=1)
    pos = torch.arange(scores.shape[1]).to(scores.device)
    softargmax = torch.sum(pos * softmax, dim=1)
    return softargmax