import os
import sys
import numpy as np

from torchvision import transforms
import torch
import torch.nn as nn
from PIL import Image

sys.path.append(os.getcwd())
from lib.models.r2p1d.r2p1d import generate_model

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = generate_model(50)
    model.fc = nn.Linear(model.fc.in_features, 700)
    pretrain = torch.load('lib/models/r2p1d/r2p1d50_K_200ep.pth', map_location='cpu')
    model.load_state_dict(pretrain['state_dict'])
    model.fc = nn.Identity()

    model = model.to(device)
    model.train(False)

    transform = transforms.Compose([
        transforms.Resize((224, 320)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4345, 0.4051, 0.3775], std=[0.2768, 0.2713, 0.2737]),
    ])

    SAMPLE_FRAMES = 9     # generate a feature vector every SAMPLE_FRAMES frames
    BATCH_SIZE = 128

    DATA_ROOT = 'data/JUDO/UNTRIMMED'
    VIDEO_FRAMES = 'video_frames_25fps'   # base folder where the video folders (containing the frames) are
    VIDEO_FEATURES = 'r2p1d_224x224_chunk9'

    with torch.set_grad_enabled(False):
        videos_dir = os.listdir(os.path.join(DATA_ROOT, VIDEO_FRAMES))
        videos_dir = [dir for dir in videos_dir if '.mp4' in dir]
        for dir in videos_dir:
            num_frames = len(os.listdir(os.path.join(DATA_ROOT, VIDEO_FRAMES, dir)))
            num_frames = num_frames - (num_frames % SAMPLE_FRAMES)

            feat_vects_video = []
            sample = []
            batch = []
            for i in range(0, num_frames, SAMPLE_FRAMES):
                sample = []
                for idx_frame in range(i, i+SAMPLE_FRAMES):
                    # idx_frame+1 because frames start from 1.  e.g. 1.jpg
                    frame = Image.open(os.path.join(DATA_ROOT, VIDEO_FRAMES, dir, str(idx_frame+1)+'.jpg')).convert('RGB')
                    frame = transform(frame).to(dtype=torch.float32)
                    sample.append(frame)

                sample = torch.stack(sample)
                batch.append(sample.permute(1, 0, 2, 3))
                if len(batch) == BATCH_SIZE:
                    batch = torch.stack(batch)
                    # forward pass
                    feat_vect = model(batch)
                    feat_vects_video.append(feat_vect)
                    batch = []

            if len(batch) != 0:
                batch = torch.stack(batch)
                # forward pass
                feat_vect = model(batch)
                feat_vects_video.append(feat_vect)

            feat_vects_video = torch.cat(feat_vects_video)
            np.save(os.path.join(DATA_ROOT, VIDEO_FEATURES, str(dir)+'.npy'), feat_vects_video.cpu().numpy())

if __name__ == '__main__':
    main()