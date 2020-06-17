import os

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

import _init_paths
import utils as utl
from configs.thumos import parse_trn_args as parse_args
from lib.models.lstm import LSTMmodel

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utl.set_seed(int(args.seed))

    model = LSTMmodel(args.hidden_size).to(device)

    criterion = utl.MultiCrossEntropyLoss(ignore_index=21).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    softmax = nn.Softmax(dim=1).to(device)

    writer = SummaryWriter()
    batch_idx_train = 1
    batch_idx_test = 1

    for epoch in range(args.epochs):
        losses = {phase: 0.0 for phase in args.phases}
        score_metrics = {phase: [] for phase in args.phases}
        target_metrics = {phase: [] for phase in args.phases}
        mAP = {phase: 0.0 for phase in args.phases}

        data_loaders = {
            phase: utl.build_data_loader(args, phase)
            for phase in args.phases
        }

        for phase in args.phases:
            training = phase=='train'
            if training:
                model.train(True)
            elif not training and args.debug:
                model.train(False)
            else:
                continue

            with torch.set_grad_enabled(training):
                avg_loss = 0
                for batch_idx, (camera_inputs, motion_inputs, enc_target, dec_target) \
                        in enumerate(data_loaders[phase], start=1):
                    camera_inputs = camera_inputs.to(device)
                    target = enc_target.to(device)
                    if training:
                        optimizer.zero_grad()

                    score = model(camera_inputs)
                    loss = criterion(score, target)

                    if training:
                        loss.backward()
                        optimizer.step()

                    # Prepare metrics for encoder
                    score = softmax(score).cpu().detach().numpy()
                    target = target.cpu().detach().numpy()
                    score_metrics[phase].extend(score)
                    target_metrics[phase].extend(target)
                    avg_loss += loss.item()

                    if training:
                        writer.add_scalar('Loss_iter/train', loss.item(), batch_idx_train)
                        batch_idx_train += 1
                    else:
                        writer.add_scalar('Loss_iter/val', loss.item(), batch_idx_test)
                        batch_idx_test += 1
                    print('{:5s} Epoch:{}  Iteration:{}  Loss:{:.3f}'.format(phase, epoch + 1, batch_idx, loss.item()))
                if training:
                writer.add_scalars('Loss_epoch/train_val_enc',
                                   {phase: enc_losses[phase] / len(data_loaders[phase].dataset) for phase in
                                    args.phases}, epoch)
                print('-- {:5s} Epoch:{} avg_loss:{:.3f}'.format(phase, epoch + 1, avg_loss / batch_idx))

if __name__ == '__main__':
    main(parse_args())