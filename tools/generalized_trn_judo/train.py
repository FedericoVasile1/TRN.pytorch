import os.path as osp
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())
import _init_paths
import utils as utl
from configs.judo import parse_trn_args as parse_args
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

    logger_APs = utl.setup_logger(osp.join(writer.log_dir, 'APs_per_epoch-encoder.txt'))

    with torch.set_grad_enabled(False):
        temp = utl.build_data_loader(args, 'train')
        dataiter = iter(temp)
        camera_inputs, motion_inputs, _, _ = dataiter.next()
        writer.add_graph(model, (camera_inputs.to(device), motion_inputs.to(device)))
        writer.close()

    batch_idx_train = 1
    batch_idx_val = 1
    best_val_enc_mAP = -1
    epoch_best_val_enc_mAP = -1
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        if epoch == args.reduce_lr_epoch:
            print('=== Learning rate reduction planned for epoch ' + str(args.reduce_lr_epoch) + ' ===')
            args.lr = args.lr * 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr

        data_loaders = {phase: utl.build_data_loader(args, phase) for phase in args.phases}

        enc_losses = {phase: 0.0 for phase in args.phases}
        enc_score_metrics = {phase: [] for phase in args.phases}
        enc_target_metrics = {phase: [] for phase in args.phases}
        #enc_mAP = {phase: 0.0 for phase in args.phases}
        dec_losses = {phase: 0.0 for phase in args.phases}
        dec_score_metrics = {phase: [] for phase in args.phases}
        dec_target_metrics = {phase: [] for phase in args.phases}
        #dec_mAP = {phase: 0.0 for phase in args.phases}

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
                for batch_idx, (camera_inputs, motion_inputs, enc_targets, dec_targets) \
                        in enumerate(data_loaders[phase], start=1):
                    # camera_inputs.shape == (batch_size, enc_steps, feat_vect_dim [if starting from features])
                    # enc_targets.shape == (batch_size, enc_steps, num_classes)
                    # enc_targets.shape == (batch_size, enc_steps, dec_steps, num_classes)
                    batch_size = camera_inputs.shape[0]
                    camera_inputs = camera_inputs.to(device)
                    motion_inputs = motion_inputs.to(device)

                    if training:
                        optimizer.zero_grad()

                    enc_scores, dec_scores = model(camera_inputs, motion_inputs)
                    # enc_scores.shape == (batch_size, enc_steps, num_classes)
                    # dec_scores.shape == (batch_size, enc_steps, dec_steps, num_classes)

                    enc_scores = enc_scores.to(device)
                    enc_targets = enc_targets.to(device)
                    # sum losses along all timesteps for encoder
                    enc_loss = criterion(enc_scores[:, 0], enc_targets[:, 0].max(axis=1)[1])
                    for step in range(1, camera_inputs.shape[1]):
                        enc_loss += criterion(enc_scores[:, step], enc_targets[:, step].max(axis=1)[1])
                    enc_loss /= camera_inputs.shape[1]  # scale by enc_steps

                    dec_scores = dec_scores.to(device)
                    dec_targets = dec_targets.view(dec_targets.shape[0], args.enc_steps, args.dec_steps, dec_targets.shape[2]).to(device)
                    # sum losses along all timesteps for decoder
                    for enc_step in range(camera_inputs.shape[1]):
                        for dec_step in range(dec_targets.shape[2]):
                            if enc_step == dec_step == 0:
                                dec_loss = criterion(dec_scores[:, enc_step, dec_step], dec_targets[:, enc_step, dec_step].max(axis=1)[1])
                            dec_loss += criterion(dec_scores[:, enc_step, dec_step], dec_targets[:, enc_step, dec_step].max(axis=1)[1])
                    dec_loss /= (camera_inputs.shape[1] * dec_targets.shape[2])  # scale by enc_steps*dec_steps

                    enc_losses[phase] += enc_loss.item() * batch_size
                    dec_losses[phase] += dec_loss.item() * batch_size

                    if training:
                        loss = enc_loss + dec_loss
                        loss.backward()
                        optimizer.step()

                    # Prepare metrics for encoder
                    enc_scores = enc_scores.view(-1, args.num_classes)
                    enc_targets = enc_targets.view(-1, args.num_classes)
                    enc_scores = softmax(enc_scores).cpu().detach().numpy()
                    enc_targets = enc_targets.cpu().detach().numpy()
                    enc_score_metrics[phase].extend(enc_scores)
                    enc_target_metrics[phase].extend(enc_targets)
                    # Prepare metrics for decoder
                    dec_scores = dec_scores.view(-1, args.num_classes)
                    dec_targets = dec_targets.view(-1, args.num_classes)
                    dec_scores = softmax(dec_scores).cpu().detach().numpy()
                    dec_targets = dec_targets.cpu().detach().numpy()
                    dec_score_metrics[phase].extend(dec_scores)
                    dec_target_metrics[phase].extend(dec_targets)

                    if training:
                        writer.add_scalar('Loss_iter/train_enc', enc_loss.item(), batch_idx_train)
                        writer.add_scalar('Loss_iter/train_dec', dec_loss.item(), batch_idx_train)
                        batch_idx_train += 1
                    else:
                        writer.add_scalar('Loss_iter/val_enc', enc_loss.item(), batch_idx_val)
                        writer.add_scalar('Loss_iter/val_dec', dec_loss.item(), batch_idx_val)
                        batch_idx_val += 1

                    if args.verbose:
                        print('[{:5s}] Epoch: {:2}  Iteration: {:3}  Enc_Loss: {:.5f}  Dec_Loss: {:.5f}'.format(phase,
                                                                                                                epoch,
                                                                                                                batch_idx,
                                                                                                                enc_loss.item(),
                                                                                                                dec_loss.item()))
        end = time.time()

        lr_sched.step(enc_losses['val'] / len(data_loaders['val'].dataset))

        writer.add_scalars('Loss_epoch/train_val_enc',
                           {phase: enc_losses[phase] / len(data_loaders[phase].dataset) for phase in args.phases},
                           epoch)
        writer.add_scalars('Loss_epoch/train_val_dec',
                           {phase: dec_losses[phase] / len(data_loaders[phase].dataset) for phase in args.phases},
                           epoch)

        result_file = {phase: 'phase-{}-epoch-{}.json'.format(phase, epoch) for phase in args.phases}
        # Compute result for encoder
        enc_result = {phase: utl.compute_result_multilabel(
            args.class_index,
            enc_score_metrics[phase],
            enc_target_metrics[phase],
            save_dir,
            result_file[phase],
            ignore_class=[0],
            save=True,
            switch=False,
            return_APs=True,
        ) for phase in args.phases}

        log = 'Epoch: ' + str(epoch)
        log += '\n[train] '
        for cls in range(args.num_classes):
            if cls == 0:  # ignore background class
                continue
            log += '| ' + args.class_index[cls] + ' AP: ' + str(enc_result['train']['AP'][args.class_index[cls]] * 100)[:4] + ' %'
        log += '| mAP: ' + str(enc_result['train']['mAP'] * 100)[:4] + ' %'
        log += '\n[val ] '
        for cls in range(args.num_classes):
            if cls == 0:  # ignore background class
                continue
            log += '| ' + args.class_index[cls] + ' AP: ' + str(enc_result['val']['AP'][args.class_index[cls]] * 100)[:4] + ' %'
        log += '| mAP: ' + str(enc_result['val']['mAP'] * 100)[:4] + ' %'
        log += '\n'
        logger_APs._write(str(log))

        enc_mAP = {phase: enc_result[phase]['mAP'] for phase in args.phases}
        writer.add_scalars('mAP_epoch/train_val_enc', {phase: enc_mAP[phase] for phase in args.phases}, epoch)

        # Compute result for decoder
        dec_mAP = {phase: utl.compute_result_multilabel(
            args.class_index,
            dec_score_metrics[phase],
            dec_target_metrics[phase],
            save_dir,
            result_file,
            ignore_class=[0],
            save=False,
            switch=False,
            return_APs=False,
        ) for phase in args.phases}

        writer.add_scalars('mAP_epoch/train_val_dec', {phase: dec_mAP[phase] for phase in args.phases}, epoch)

        log = 'Epoch: {:2} | [train] enc_loss: {:.5f}  enc_mAP: {:.4f}  dec_loss: {:.5f}  dec_mAP: {:.4f} |'
        log += ' [val] enc_loss: {:.5f}  enc_mAP: {:.4f}  dec_loss: {:.5f}  dec_mAP: {:.4f} |\n'
        log += 'running_time: {:.2f} sec'
        log = str(log).format(epoch,
                              enc_losses['train'] / len(data_loaders['train'].dataset),
                              enc_mAP['train'],
                              dec_losses['train'] / len(data_loaders['train'].dataset),
                              dec_mAP['train'],
                              enc_losses['val'] / len(data_loaders['val'].dataset),
                              enc_mAP['val'],
                              dec_losses['val'] / len(data_loaders['val'].dataset),
                              dec_mAP['val'],
                              end - start)
        print(log)
        logger._write(log)

        if best_val_enc_mAP < enc_mAP['val']:
            best_val_enc_mAP = enc_mAP['val']
            epoch_best_val_enc_mAP = epoch

            # only the best validation map model is saved
            checkpoint_file = 'model-{}-features-{}.pth'.format(args.model, args.camera_feature)
            torch.save({
                'enc-val_mAP': best_val_enc_mAP,
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if args.distributed else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, osp.join(save_dir, checkpoint_file))
            torch.save({
                'enc-val_mAP': best_val_enc_mAP,
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if args.distributed else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, osp.join(writer.log_dir, checkpoint_file))

    log = '--- Best encoder validation mAP is {:.1f} % obtained at epoch {} ---'.format(best_val_enc_mAP * 100, epoch_best_val_enc_mAP)
    print(log)
    logger._write(log)

if __name__ == '__main__':
    main(parse_args())
