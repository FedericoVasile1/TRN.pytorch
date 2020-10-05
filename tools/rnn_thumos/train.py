import os.path as osp
import os
import sys
import time

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())
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
        camera_inputs, motion_inputs, _, _ = dataiter.next()
        writer.add_graph(model, [camera_inputs.to(device), motion_inputs.to(device)])
        writer.close()

    batch_idx_train = 1
    batch_idx_test = 1
    best_test_mAP = -1
    epoch_best_test_mAP = -1
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
                for batch_idx, (camera_inputs, motion_inputs, targets, _) in enumerate(data_loaders[phase], start=1):
                    # camera_inputs.shape == (batch_size, enc_steps, feat_vect_dim [if starting from features])
                    # targets.shape == (batch_size, enc_steps, num_classes)
                    batch_size = camera_inputs.shape[0]
                    camera_inputs = camera_inputs.to(device)
                    motion_inputs = motion_inputs.to(device)

                    if training:
                        optimizer.zero_grad()

                    scores = model(camera_inputs, motion_inputs)            # scores.shape == (batch_size, enc_steps, num_classes)

                    scores = scores.to(device)
                    targets = targets.to(device)
                    # sum losses along all timesteps
                    loss = criterion(scores[:, 0], targets[:, 0].max(axis=1)[1])
                    for step in range(1, camera_inputs.shape[1]):
                        loss += criterion(scores[:, step], targets[:, step].max(axis=1)[1])
                    loss /= camera_inputs.shape[1]      # scale by enc_steps

                    losses[phase] += loss.item() * batch_size

                    if training:
                        loss.backward()
                        optimizer.step()

                    # Prepare metrics
                    scores = scores.view(-1, args.num_classes)
                    targets = targets.view(-1, args.num_classes)
                    scores = softmax(scores).cpu().detach().numpy()
                    targets = targets.cpu().detach().numpy()
                    score_metrics[phase].extend(scores)
                    target_metrics[phase].extend(targets)

                    if training:
                        writer.add_scalar('Loss_iter/train_enc', loss.item(), batch_idx_train)
                        batch_idx_train += 1
                    else:
                        writer.add_scalar('Loss_iter/val_enc', loss.item(), batch_idx_test)
                        batch_idx_test += 1

                    if args.verbose:
                        print('[{:5s}] Epoch: {:2}  Iteration: {:3}  Loss: {:.5f}'.format(phase,
                                                                                          epoch,
                                                                                          batch_idx,
                                                                                          loss.item()))
        end = time.time()

        lr_sched.step(losses['test'] / len(data_loaders['test'].dataset))

        writer.add_scalars('Loss_epoch/train_val_enc',
                           {phase: losses[phase] / len(data_loaders[phase].dataset) for phase in args.phases},
                           epoch)

        result_file = {phase: 'phase-{}-epoch-{}.json'.format(phase, epoch) for phase in args.phases}
        result = {phase: utl.compute_result_multilabel(
            args.class_index,
            score_metrics[phase],
            target_metrics[phase],
            save_dir,
            result_file[phase],
            ignore_class=[0, 21],
            save=True,
            return_APs=True,
        ) for phase in args.phases}

        log = 'Epoch: ' + str(epoch)
        log += '\n[train] '
        for cls in range(args.num_classes):
            if cls == 0 or cls == 21:       # ignore background class and amibiguous class
                continue
            log += '| ' + args.class_index[cls] + ' AP: ' + str(result['train']['AP'][args.class_index[cls]] * 100)[:4] + ' %'
        log += '| mAP: ' + str(result['train']['mAP'] * 100)[:4] + ' %'
        log += '\n[test ] '
        for cls in range(args.num_classes):
            if cls == 0 or cls == 21:       # ignore background class and amibiguous class
                continue
            log += '| ' + args.class_index[cls] + ' AP: ' + str(result['test']['AP'][args.class_index[cls]] * 100)[:4] + ' %'
        log += '| mAP: ' + str(result['test']['mAP'] * 100)[:4] + ' %'
        log += '\n'
        logger_APs._write(str(log))

        mAP = {phase: result[phase]['mAP'] for phase in args.phases}
        writer.add_scalars('mAP_epoch/train_val_enc', {phase: mAP[phase] for phase in args.phases}, epoch)

        log = 'Epoch: {:2} | [train] loss: {:.5f}  mAP: {:.4f} |'
        log += ' [test] loss: {:.5f}  mAP: {:.4f}  |\n'
        log += 'running_time: {:.2f} sec'
        log = str(log).format(epoch,
                              losses['train'] / len(data_loaders['train'].dataset),
                              mAP['train'],
                              losses['test'] / len(data_loaders['test'].dataset),
                              mAP['test'],
                              end - start)
        print(log)
        logger._write(log)

        if best_test_mAP < mAP['test']:
            best_test_mAP = mAP['test']
            epoch_best_test_mAP = epoch

            # only the best validation map model is saved
            checkpoint_file = 'model-{}-features-{}.pth'.format(args.model, args.camera_feature)
            torch.save({
                'test_mAP': best_test_mAP,
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if args.distributed else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, osp.join(save_dir, checkpoint_file))
            torch.save({
                'test_mAP': best_test_mAP,
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if args.distributed else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, osp.join(writer.log_dir, checkpoint_file))

    log = '--- Best test mAP is {:.1f} % obtained at epoch {} ---'.format(best_test_mAP * 100, epoch_best_test_mAP)
    print(log)
    logger._write(log)

if __name__ == '__main__':
    main(parse_args())