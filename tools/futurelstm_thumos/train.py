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
    if osp.isfile(args.checkpoint):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
        args.start_epoch += checkpoint['epoch']
    softmax = nn.Softmax(dim=1).to(device)

    writer = SummaryWriter()
    batch_idx_train = 1
    batch_idx_test = 1

    with torch.set_grad_enabled(False):
        temp = utl.build_data_loader(args, 'train')
        dataiter = iter(temp)
        camera_inputs, _, _, _ = dataiter.next()
        writer.add_graph(model, camera_inputs.to(device))
        writer.close()

    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        data_loaders = {
            phase: utl.build_data_loader(args, phase)
            for phase in args.phases
        }

        dec_avg_losses = {phase: 0.0 for phase in args.phases}
        dec_score_metrics = {phase: [] for phase in args.phases}
        dec_target_metrics = {phase: [] for phase in args.phases}

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
                    if training:
                        optimizer.zero_grad()

                    # forward pass
                    dec_scores = model(camera_inputs)   # dec_scores.shape == (batch_size, enc_steps, dec_steps, feat_vect_dim)

                    # sum decoder losses along all timesteps
                    dec_scores = dec_scores.to(device)
                    dec_target = dec_target.view(dec_target.shape[0], args.enc_steps, args.dec_steps, dec_target.shape[2]).to(device)
                    for enc_step in range(camera_inputs.shape[1]):
                        for dec_step in range(0, dec_scores.shape[2]):
                            if enc_step == dec_step == 0:
                                dec_loss = criterion(dec_scores[:, enc_step, dec_step], dec_target[:, enc_step, dec_step].max(axis=1)[1])
                            else:
                                dec_loss += criterion(dec_scores[:, enc_step, dec_step], dec_target[:, enc_step, dec_step].max(axis=1)[1])
                    dec_loss /= (camera_inputs.shape[1] * dec_scores.shape[2])      # scale by enc_steps*dec_steps

                    dec_avg_losses[phase] += dec_loss.item() * batch_size

                    if training:
                        loss = dec_loss
                        loss.backward()
                        optimizer.step()

                    # Prepare metrics for decoder
                    if True:
                        dec_scores = dec_scores.view(-1, args.feat_vect_dim)
                        dec_target = dec_target.view(-1, args.feat_vect_dim)
                    else:
                        dec_scores = dec_scores.view(-1, args.num_classes)
                        dec_target = dec_target.view(-1, args.num_classes)
                    dec_scores = softmax(dec_scores).cpu().detach().numpy()
                    dec_target = dec_target.cpu().detach().numpy()
                    dec_score_metrics[phase].extend(dec_scores)
                    dec_target_metrics[phase].extend(dec_target)

                    if training:
                        writer.add_scalar('Loss_iter/train_enc', dec_loss.item(), batch_idx_train)
                        batch_idx_train += 1
                    else:
                        writer.add_scalar('Loss_iter/val_enc', dec_loss.item(), batch_idx_test)
                        batch_idx_test += 1

                    print('[{:5s}] Epoch: {:2}  Iteration: {:3}  Enc_Loss: {:.5f}'.format(phase,
                                                                                          epoch,
                                                                                          batch_idx,
                                                                                          dec_loss.item()))
        end = time.time()

        writer.add_scalars('Loss_epoch/train_val_enc',
                           {phase: dec_avg_losses[phase] / len(data_loaders[phase].dataset) for phase in args.phases},
                           epoch)

        result_file = {phase: 'enc-phase-{}-epoch-{}.json'.format(phase, epoch) for phase in args.phases}
        dec_mAP = {phase: utl.compute_result_multilabel(
            args.class_index,
            dec_score_metrics[phase],
            dec_target_metrics[phase],
            save_dir,
            result_file[phase],
            ignore_class=[0, 21],
            save=True,
        ) for phase in args.phases}

        writer.add_scalars('mAP_epoch/train_val_enc', {phase: dec_mAP[phase] for phase in args.phases}, epoch)

        log = 'Epoch: {:2} | [train] enc_avg_loss: {:.5f}  enc_mAP: {:.4f} |'
        log += ' [test] enc_avg_loss: {:.5f}  enc_mAP: {:.4f} |\n'
        log += 'running_time: {:.2f} sec'
        log = str(log).format(epoch,
                              dec_avg_losses['train'] / len(data_loaders['train'].dataset),
                              dec_mAP['train'],
                              dec_avg_losses['test'] / len(data_loaders['test'].dataset),
                              dec_mAP['test'],
                              end - start)
        print(log)

        checkpoint_file = 'inputs-{}-epoch-{}.pth'.format(args.inputs, epoch)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if args.distributed else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, osp.join(save_dir, checkpoint_file))

if __name__ == '__main__':
    main(parse_args())