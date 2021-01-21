import os
import sys
import numpy as np

from torchvision import transforms
import torch
import torch.nn as nn
from PIL import Image

sys.path.append(os.getcwd())
from lib.models.i3d.i3d import InceptionI3d, I3DNormalization

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = InceptionI3d()
    model.load_state_dict(torch.load('lib/models/i3d/rgb_imagenet.pt'))
    model.dropout = nn.Identity()
    model.logits = nn.Identity()

    model = model.to(device)
    model.train(False)

    transform = transforms.Compose([
        transforms.Resize((224, 320)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        I3DNormalization(),
    ])

    CHUNK_SIZE = 100
    BATCH_SIZE = 32

    DATA_ROOT = 'data/JUDO/UNTRIMMED'
    VIDEO_FRAMES = 'onlyact_video_frames_25fps'   # base folder where the video folders (containing the frames) are
    VIDEO_FEATURES = 'onlyact_i3d_224x224_chunk'+str(CHUNK_SIZE)

    with torch.set_grad_enabled(False):
        videos_dir = os.listdir(os.path.join(DATA_ROOT, VIDEO_FRAMES))
        videos_dir = [dir for dir in videos_dir if '.mp4' in dir]
        for dir in videos_dir:
            num_frames = len(os.listdir(os.path.join(DATA_ROOT, VIDEO_FRAMES, dir)))
            num_frames = num_frames - (num_frames % CHUNK_SIZE)

            start_frame = int(dir.split('___')[0])

            feat_vects_video = []
            batch = []
            for i in range(0, num_frames, CHUNK_SIZE):
                sample = []
                for idx_frame in range(i, i+CHUNK_SIZE):
                    # idx_frame+1 because frames start from 1.  e.g. 1.jpg
                    frame = Image.open(os.path.join(DATA_ROOT,
                                                    VIDEO_FRAMES,
                                                    dir,
                                                    str(start_frame+idx_frame+1)+'.jpg')).convert('RGB')
                    frame = transform(frame).to(dtype=torch.float32)
                    sample.append(frame)

                sample = torch.stack(sample)
                batch.append(sample.permute(1, 0, 2, 3))
                if len(batch) == BATCH_SIZE:
                    batch = torch.stack(batch).to(device)
                    # forward pass
                    feat_vect = model(batch)
                    feat_vects_video.append(feat_vect)
                    batch = []

            if len(batch) != 0:
                batch = torch.stack(batch).to(device)
                # forward pass
                feat_vect = model(batch)
                feat_vects_video.append(feat_vect)

            feat_vects_video = torch.cat(feat_vects_video).cpu()
            np.save(os.path.join(DATA_ROOT,
                                 VIDEO_FEATURES,
                                 str(dir)+'.npy'), feat_vects_video.numpy())

if __name__ == '__main__':
    main()