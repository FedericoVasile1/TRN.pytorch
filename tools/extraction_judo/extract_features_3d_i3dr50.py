import os
import sys
import numpy as np

from torchvision import transforms
import torch
import torch.nn as nn
from PIL import Image

sys.path.append(os.getcwd())
from lib.models.i3d_resnet.i3d_resnet import i3d_resnet

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = i3d_resnet(50, 400, 0.5, without_t_stride=False)

    checkpoint = torch.load('lib/models/i3d_resnet/'
                            'kinetics400-rgb-i3d-resnet-50-f32-s2-precise_bn-warmupcosine-bs1024-e196.pth',
                            map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    model.dropout = nn.Identity()
    model.fc = nn.Identity()

    model = model.to(device)
    model.train(False)

    transform = transforms.Compose([
        transforms.Resize((224, 320)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    CHUNK_SIZE = 9
    BATCH_SIZE = 64

    DATA_ROOT = 'data/JUDO/UNTRIMMED'
    VIDEO_FRAMES = 'video_frames_25fps'   # base folder where the video folders (containing the frames) are
    VIDEO_FEATURES = 'i3dr50_224x224_chunk'+str(CHUNK_SIZE)

    with torch.set_grad_enabled(False):
        videos_dir = os.listdir(os.path.join(DATA_ROOT, VIDEO_FRAMES))
        videos_dir = [dir for dir in videos_dir if '.mp4' in dir]
        for dir in videos_dir:
            num_frames = len(os.listdir(os.path.join(DATA_ROOT, VIDEO_FRAMES, dir)))
            num_frames = num_frames - (num_frames % CHUNK_SIZE)

            feat_vects_video = []
            batch = []
            for i in range(0, num_frames, CHUNK_SIZE):
                sample = []
                for idx_frame in range(i, i+CHUNK_SIZE):
                    # idx_frame+1 because frames start from 1.  e.g. 1.jpg
                    frame = Image.open(os.path.join(DATA_ROOT,
                                                    VIDEO_FRAMES,
                                                    dir,
                                                    str(idx_frame+1)+'.jpg')).convert('RGB')
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