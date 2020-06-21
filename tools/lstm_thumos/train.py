'''
python tools/lstm_thumos/train.py --epochs 10 --enc_steps 2 --dec_steps 2 --hidden_size 32 --neurons 12 --feat_vect_dim 2048 --data_info data/small_data_info.json --model LSTM
python3 tools/lstm_thumos/train.py --epochs 5 --enc_steps 64 --dec_steps 8 --hidden_size 2048 --neurons 128 --model TRN
'''
import os.path as osp
import os
import sys
import time

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

import _init_paths
import utils as utl
from configs.thumos import parse_trn_args as parse_args
from models import build_model

def main(args):
    this_dir = osp.join(osp.dirname(__file__), '.')
    save_dir = osp.join(this_dir, 'checkpoints')
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utl.set_seed(int(args.seed))

    if args.model != 'LSTMV2':
        raise Exception('wrong model name, this pipeline is only for LSTMmodelV2 class')
    model = build_model(args)
    if osp.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.apply(utl.weights_init)
    if args.distributed:
        model = nn.DataParallel(model)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=21).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    softmax = nn.Softmax(dim=1).to(device)

    writer = SummaryWriter()
    batch_idx_train = 1
    batch_idx_test = 1

    for epoch in range(args.epochs):
        data_loaders = {
            phase: utl.build_data_loader(args, phase)
            for phase in args.phases
        }

        losses = {phase: 0.0 for phase in args.phases}
        score_metrics = {phase: [] for phase in args.phases}
        target_metrics = {phase: [] for phase in args.phases}

        start = time.time()
        for phase in args.phases:
            training = phase=='train'
            if training:
                model.train(True)
            elif not training and args.debug:
                model.train(False)
            else:
                continue

            with torch.set_grad_enabled(training):
                for batch_idx, (camera_inputs, motion_inputs, enc_target, dec_target) \
                        in enumerate(data_loaders[phase], start=1):
                    batch_size = camera_inputs.shape[0]
                    camera_inputs = camera_inputs.to(device)
                    target = enc_target.to(device)
                    if training:
                        optimizer.zero_grad()

                    # forward pass
                    score = model(camera_inputs)        # score.shape == (batch_size, enc_steps, num_classes)

                    # sum losses along all timesteps
                    loss = criterion(score[:, 0], target[:, 0].max(axis=1)[1])
                    for step in range(1, camera_inputs.shape[1]):
                        loss += criterion(score[:, step], target[:, step].max(axis=1)[1])
                    loss /= camera_inputs.shape[1]

                    losses[phase] += loss.item() * batch_size

                    if training:
                        loss.backward()
                        optimizer.step()

                    score = softmax(score).cpu().detach().numpy()
                    target = target.cpu().detach().numpy()
                    score_metrics[phase].extend(score)
                    target_metrics[phase].extend(target)

                    if training:
                        writer.add_scalar('Loss_iter/train_enc', loss.item(), batch_idx_train)
                        batch_idx_train += 1
                    else:
                        writer.add_scalar('Loss_iter/val_enc', loss.item(), batch_idx_test)
                        batch_idx_test += 1

                    print('[{:5s}] Epoch: {:2}  Iteration: {:3}  Loss: {:.5f}'.format(phase, epoch+1, batch_idx, loss.item()))
        end = time.time()

        writer.add_scalars('Loss_epoch/train_val_enc',
                           {phase: losses[phase] / len(data_loaders[phase].dataset) for phase in args.phases},
                           epoch+1)
        print('Epoch: {:2} | [train] avg_loss: {:.5f} | [test] avg_loss: {:.5f} | running_time: {:.2f} sec'
              .format(epoch+1, losses['train'] / len(data_loaders['train'].dataset),
                      losses['test'] / len(data_loaders['test'].dataset), end-start))

        '''
        result_file = {phase: 'phase-{}-epoch-{}.json'.format(phase, epoch) for phase in args.phases}
        mAP = {phase: utl.compute_result_multilabel(
            args.class_index,
            score_metrics[phase],
            target_metrics[phase],
            save_dir,
            result_file[phase],
            ignore_class=[0, 21],
            save=True,
        ) for phase in args.phases}
        writer.add_scalars('mAP_epoch/train_val', {phase: mAP[phase] for phase in args.phases}, epoch)
        '''

        checkpoint_file = 'inputs-{}-epoch-{}.pth'.format(args.inputs, epoch)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if args.distributed else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, osp.join(save_dir, checkpoint_file))

if __name__ == '__main__':
    main(parse_args())