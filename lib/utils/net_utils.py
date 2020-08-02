import random

import numpy as np
import os.path as osp
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
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

def show_video_predictions(args, camera_inputs, session, enc_score_metrics, enc_target_metrics):
    enc_pred_metrics = torch.max(torch.tensor(enc_score_metrics), 1)[1]
    enc_target_metrics = torch.max(torch.tensor(enc_target_metrics), 1)[1]

    for idx in range(camera_inputs.shape[0]):
        idx_frame = idx * 6 + 3  # because features are extracted by taking the central frame every 6 frames
        pil_frame = Image.open(osp.join(args.data_root, 'video_frames_24fps', session,
                                        str(idx_frame + 1) + '.jpg')).convert('RGB')
        open_cv_frame = np.array(pil_frame)
        # Convert RGB to BGR
        open_cv_frame = open_cv_frame[:, :, ::-1].copy()

        open_cv_frame = cv2.copyMakeBorder(open_cv_frame, 60, 0, 0, 0, borderType=cv2.BORDER_CONSTANT, value=0)
        pred_label = args.class_index[enc_pred_metrics[idx]]
        target_label = args.class_index[enc_target_metrics[idx]]

        cv2.putText(open_cv_frame, pred_label, (0, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0) if pred_label == target_label else (0, 0, 255), 1)
        cv2.putText(open_cv_frame, target_label, (0, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 1)
        cv2.putText(open_cv_frame,
                    'prob:{:.2f}'.format(torch.tensor(enc_score_metrics)[idx, enc_pred_metrics[idx]].item()),
                    (210, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0) if pred_label == target_label else (0, 0, 255), 1)

        # [ (idx_frame + 1) / 24 ]    => 24 because frames has been extracted at 24 fps
        cv2.putText(open_cv_frame, '{:.2f}s'.format((idx_frame + 1) / 24), (275, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, (255, 255, 255), 1)
        cv2.putText(open_cv_frame, str(idx_frame + 1), (275, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, (255, 255, 255), 1)

        # display the frame to screen
        cv2.imshow(session, open_cv_frame)
        key = cv2.waitKey(int(41.6 * 6))  # time is in milliseconds
        if key == ord('q'):
            # quit
            cv2.destroyAllWindows()
            break
        if key == ord('p'):
            # pause
            cv2.waitKey(-1)  # wait until any key is pressed

def soft_argmax(scores):
    # scores.shape == (batch_size, num_classes).   scores are NOT passed through softmax
    softmax = F.softmax(scores, dim=1)
    pos = torch.arange(scores.shape[1]).to(scores.device)
    softargmax = torch.sum(pos * softmax, dim=1)
    return softargmax