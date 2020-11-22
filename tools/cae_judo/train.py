import os.path as osp
import os
import sys
import time

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())
from lib import utils as utl
from configs.judo import parse_model_args as parse_args
from lib.models import build_model

def main(args):
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

    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.reduce_lr_count, verbose=True)
    if osp.isfile(args.checkpoint):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
        args.start_epoch += checkpoint['epoch']

    writer = SummaryWriter()
    print('Tensorboard log dir: ' + writer.log_dir)

    logger = utl.setup_logger(osp.join(writer.log_dir, 'log.txt'))

    command = 'python ' + ' '.join(sys.argv)
    logger._write(command)


    with torch.set_grad_enabled(False):
        temp = utl.build_data_loader(args, 'train')
        dataiter = iter(temp)
        inputs = dataiter.next()
        writer.add_graph(model, inputs.to(device))
        writer.close()

    best_val_loss = None
    epoch_best_val_loss = None
    batch_idx_train = 1
    batch_idx_val = 1
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        if epoch == args.reduce_lr_epoch:
            print('=== Learning rate reduction planned for epoch ' + str(args.reduce_lr_epoch) + ' ===')
            args.lr = args.lr * 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr

        data_loaders = {phase: utl.build_data_loader(args, phase) for phase in args.phases}

        losses = {phase: 0.0 for phase in args.phases}

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
                for batch_idx, inputs in enumerate(data_loaders[phase], start=1):
                    # inputs.shape == (batch_size, steps, 3, H, W)
                    batch_size = inputs.shape[0]
                    inputs = inputs.to(device)

                    if training:
                        optimizer.zero_grad()

                    scores = model(inputs)            # scores.shape == (batch_size, steps, 3, H, W)

                    scores = scores.to(device)
                    loss = criterion(inputs, scores)

                    losses[phase] += loss.item() * batch_size

                    if training:
                        loss.backward()
                        optimizer.step()

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

        lr_sched.step(losses['val'] / len(data_loaders['val'].dataset))

        log = 'Epoch: {:2} | [train] loss: {:.5f}  |'
        log += ' [val] loss: {:.5f}  |\n'
        log += 'running_time: {:.2f} sec'
        log = str(log).format(epoch,
                              losses['train'] / len(data_loaders['train'].dataset),
                              losses['val'] / len(data_loaders['val'].dataset),
                              end - start)
        print(log)
        logger._write(log)

        if best_val_loss is None or best_val_loss > losses['val']:
            best_val_loss = losses['val']
            epoch_best_val_loss = epoch

            # only the best validation map model is saved
            checkpoint_file = 'model-{}_features-{}.pth'.format(args.model, args.model_input)
            torch.save({
                'val_loss': best_val_loss,
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if args.distributed else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, osp.join(writer.log_dir, checkpoint_file))

    log = '--- Best validation loss is {} obtained at epoch {} ---'.format(best_val_loss, epoch_best_val_loss)
    print(log)
    logger._write(log)

if __name__ == '__main__':
    main(parse_args())