import _init_paths
import utils as utl

import os
import numpy as np

from torchvision import models, transforms
import torch
import torch.nn as nn

from PIL import Image

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.resnet18(pretrained=True)
    model.fc = nn.Identity()

    model = model.to(device)
    model.train(False)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    DATA_ROOT = 'data/THUMOS'

    VIDEO_FRAMES = 'extracted_optical_flow'   # base folder where the video folders (containing the frames) are
    VIDEO_FEATURES = 'resnet18_extracted_optical_flow'

    with torch.set_grad_enabled(False):
        videos_dir = os.listdir(os.path.join(DATA_ROOT, VIDEO_FRAMES))
        videos_dir = [dir for dir in videos_dir if 'video' in dir]
        for dir in videos_dir:
            all_frames = np.load(os.path.join(DATA_ROOT, VIDEO_FRAMES, str(dir)))  # (num_frames, 3, 184, 320)
            num_frames = all_frames.shape[0]
            extracted_feat_vect = []
            for idx_frame in range(num_frames):
                frame = all_frames[idx_frame]
                frame = torch.as_tensor(frame.astype(np.float32))
                frame = transform(frame).to(device)
                # forward pass
                feat_vect = model(frame.unsqueeze(0))   # TODO: load a batch instead of a single sample
                extracted_feat_vect.append(feat_vect.squeeze(0))

            extracted_feat_vect = torch.stack(extracted_feat_vect)
            np.save(os.path.join(DATA_ROOT, VIDEO_FEATURES, str(dir)), extracted_feat_vect)

if __name__ == '__main__':
    main()