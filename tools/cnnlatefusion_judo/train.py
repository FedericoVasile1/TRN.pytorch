import os.path as osp
import os
import sys
import time
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())
import _init_paths
import utils as utl
from configs.judo import parse_trn_args as parse_args
from models import build_model

def main(args):
    # fix between batch_size and enc_steps, due to the way the dataset class works(i.e. this model do not
    # have a recurrent part, so we do not have the concept of 'enc_steps', so we arrange it manually here
    # only to make the dataset class working properly)
    # e.g. args.batch_size == 64
    args.enc_steps = args.batch_size // 2    # 32
    args.batch_size = 2                     # 2
    # now, since after we will fuse batch_size and enc_steps(i.e. batch_size * enc_steps) we will
    # get back to the original batch_size, i.e. 32 * 2 = 64

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

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.reduce_lr_count, verbose=True)
    if osp.isfile(args.checkpoint):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
        args.start_epoch += checkpoint['epoch']
    softmax = nn.Softmax(dim=1).to(device)

    writer = SummaryWriter()
    print('Tensorboard log dir: ' + writer.log_dir)

    logger = utl.setup_logger(osp.join(writer.log_dir, 'log.txt'))
    command = 'python ' + ' '.join(sys.argv)
    logger._write(command)

    logger_APs = utl.setup_logger(osp.join(writer.log_dir, 'APs_per_epoch.txt'))

    with torch.set_grad_enabled(False):
        temp = utl.build_data_loader(args, 'train')
        dataiter = iter(temp)
        camera_inputs, _, _, _ = dataiter.next()
        camera_inputs = camera_inputs.view(-1,
                                           camera_inputs.shape[2],
                                           camera_inputs.shape[3],
                                           camera_inputs.shape[4],
                                           camera_inputs.shape[5])
        writer.add_graph(model, camera_inputs.to(device))
        writer.close()

    batch_idx_train = 1
    batch_idx_val = 1
    best_val_map = -1
    epoch_best_val_map = -1
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        if epoch == args.reduce_lr_epoch:
            print('=== Learning rate reduction planned for epoch ' + str(args.reduce_lr_epoch) + ' ===')
            args.lr = args.lr * 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr

        data_loaders = {phase: utl.build_data_loader(args, phase) for phase in args.phases}

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
                for batch_idx, (camera_inputs, _, targets, _) in enumerate(data_loaders[phase], start=1):
                    # camera.inputs.shape == (batch_size, enc_steps, chunk_size C, H, W)
                    # targets.shape == (batch_size, enc_steps, num_classes)

                    # fuse batch_size and enc_steps
                    camera_inputs = camera_inputs.view(-1,
                                                       camera_inputs.shape[2],
                                                       camera_inputs.shape[3],
                                                       camera_inputs.shape[4],
                                                       camera_inputs.shape[5])
                    targets = targets.view(-1, targets.shape[2])

                    batch_size = camera_inputs.shape[0]
                    camera_inputs = camera_inputs.to(device)

                    if training:
                        optimizer.zero_grad()

                    scores = model(camera_inputs)       # scores.shape == (batch_size, num_classes)

                    scores = scores.to(device)
                    targets = targets.to(device)
                    loss = criterion(scores, targets.max(axis=1)[1])

                    losses[phase] += loss.item() * batch_size

                    if training:
                        loss.backward()
                        optimizer.step()

                    # Prepare metrics
                    scores = softmax(scores).cpu().detach().numpy()
                    targets = targets.cpu().detach().numpy()
                    score_metrics[phase].extend(scores)
                    target_metrics[phase].extend(targets)

                    if training:
                        writer.add_scalar('Loss_iter/train', loss.item(), batch_idx_train)
                        batch_idx_train += 1
                    else:
                        writer.add_scalar('Loss_iter/val', loss.item(), batch_idx_val)
                        batch_idx_val += 1

                    if args.verbose:
                        print('[{:5s}] Epoch: {:2}  Iteration: {:3}  Loss: {:.5f}'.format(phase,
                                                                                          epoch,
                                                                                          batch_idx,
                                                                                          loss.item()))
        end = time.time()

        lr_sched.step(losses['val'] / (len(data_loaders['val'].dataset) * args.enc_steps))

        writer.add_scalars('Loss_epoch/train_val',
                           {phase: losses[phase] / (len(data_loaders[phase].dataset) * args.enc_steps)
                            for phase in args.phases},
                           epoch)

        result_file = {phase: 'phase-{}-epoch-{}.json'.format(phase, epoch) for phase in args.phases}
        result = {phase: utl.compute_result_multilabel(
            args.class_index,
            score_metrics[phase],
            target_metrics[phase],
            save_dir,
            result_file[phase],
            ignore_class=[0],
            save=True,
            switch=False,
            return_APs=True,
        ) for phase in args.phases}

        log = 'Epoch: ' + str(epoch)
        log += '\n[train] '
        for cls in range(1, args.num_classes):  # starts from 1 in order to drop background class
            log += '| ' + args.class_index[cls] + ' AP: ' + str(result['train']['AP'][args.class_index[cls]] * 100)[:4] + ' %'
        log += '| mAP: ' + str(result['train']['mAP'] * 100)[:4] + ' %'
        log += '\n[val  ] '
        for cls in range(1, args.num_classes):  # starts from 1 in order to drop background class
            log += '| ' + args.class_index[cls] + ' AP: ' + str(result['val']['AP'][args.class_index[cls]] * 100)[:4] + ' %'
        log += '| mAP: ' + str(result['val']['mAP'] * 100)[:4] + ' %'
        log += '\n'
        logger_APs._write(str(log))

        mAP = {phase: result[phase]['mAP'] for phase in args.phases}
        writer.add_scalars('mAP_epoch/train_val', {phase: mAP[phase] for phase in args.phases}, epoch)

        log = 'Epoch: {:2} | [train] loss: {:.5f}  mAP: {:.4f}  |'
        log += ' [val] loss: {:.5f}  mAP: {:.4f}|\n'
        log += 'running_time: {:.2f} sec'
        log = str(log).format(epoch,
                              losses['train'] / (len(data_loaders['train'].dataset) * args.enc_steps),
                              mAP['train'],
                              losses['val'] / (len(data_loaders['val'].dataset) * args.enc_steps),
                              mAP['val'],
                              end - start)
        print(log)
        logger._write(log)

        if best_val_map < mAP['val']:
            best_val_map = mAP['val']
            epoch_best_val_map = epoch

            # only the best validation map model is saved
            checkpoint_file = 'model-{}-feature_extractor-{}.pth'.format(args.model, args.feature_extractor)
            torch.save({
                'val_mAP': best_val_map,
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if args.distributed else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, osp.join(save_dir, checkpoint_file))
            torch.save({
                'val_mAP': best_val_map,
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if args.distributed else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, osp.join(writer.log_dir, checkpoint_file))

    log = '--- Best validation mAP is {:.1f} % obtained at epoch {} ---'.format(best_val_map * 100, epoch_best_val_map)
    print(log)
    logger._write(log)

if __name__ == '__main__':
    main(parse_args())