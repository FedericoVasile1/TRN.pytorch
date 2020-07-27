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
    criterion_discr = nn.CrossEntropyLoss().to(device)
    criterion_act = nn.CrossEntropyLoss(ignore_index=21).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if osp.isfile(args.checkpoint):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
        args.start_epoch += checkpoint['epoch']
    softmax = nn.Softmax(dim=1).to(device)

    writer = SummaryWriter()
    command = 'python ' + ' '.join(sys.argv)
    f = open(writer.log_dir + '/run_command.txt', 'w+')
    f.write(command)
    f.close()

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

        losses = {phase: 0.0 for phase in args.phases}
        losses_discr = {phase: 0.0 for phase in args.phases}
        losses_act = {phase: 0.0 for phase in args.phases}
        score_metrics = {phase: [] for phase in args.phases}
        score_metrics_discr = {phase: [] for phase in args.phases}
        score_metrics_act = {phase: [] for phase in args.phases}
        target_metrics = {phase: [] for phase in args.phases}
        target_metrics_discr = {phase: [] for phase in args.phases}
        target_metrics_act = {phase: [] for phase in args.phases}

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
                for batch_idx, (camera_inputs, _, enc_target, _) in enumerate(data_loaders[phase], start=1):
                    # camera_inputs.shape == (batch_size, enc_steps, feat_vect_dim)
                    # enc_target.shape == (batch_size, enc_steps, num_classes)
                    batch_size = camera_inputs.shape[0]
                    camera_inputs = camera_inputs.to(device)

                    # convert ground truth to only 0 and 1 values (0 means background, 1 means action)
                    #  (notice that target is a one-hot encodeing tensor, so at the end it should
                    #   be such)
                    target_discr = torch.max(enc_target, dim=2)[1]
                    target_discr[target_discr != 0] = 1  # convert all actions index classes to a single 'action class'
                    # re-convert tensor to one-hot encoding tensor
                    target_discr = torch.nn.functional.one_hot(target_discr, num_classes=2)

                    if training:
                        optimizer.zero_grad()

                    # forward pass
                    score, score_discr, score_act = model(camera_inputs)   # score.shape == (batch_size, enc_steps, num_classes)

                    score, score_discr, score_act = (score.to(device), score_discr.to(device), score_act.to(device))
                    target, target_discr = (enc_target.to(device), target_discr.to(device))
                    # sum losses along all timesteps for final loss
                    loss = criterion(score[:, 0], target[:, 0].max(axis=1)[1])
                    for step in range(1, camera_inputs.shape[1]):
                        loss += criterion(score[:, step], target[:, step].max(axis=1)[1])
                    loss /= camera_inputs.shape[1]      # scale by enc_steps

                    # sum losses along all timesteps for discriminator loss
                    loss_discr = criterion_discr(score_discr[:, 0], target_discr[:, 0].max(axis=1)[1])
                    for step in range(1, camera_inputs.shape[1]):
                        loss_discr += criterion_discr(score_discr[:, step], target_discr[:, step].max(axis=1)[1])
                    loss_discr /= camera_inputs.shape[1]  # scale by enc_steps

                    # sum losses along all timesteps for action loss
                    loss_act = criterion_act(score_act[:, 0], target[:, 0].max(axis=1)[1])
                    for step in range(1, camera_inputs.shape[1]):
                        loss_act += criterion_act(score_act[:, step], target[:, step].max(axis=1)[1])
                    loss_act /= camera_inputs.shape[1]  # scale by enc_steps

                    losses[phase] += loss.item() * batch_size
                    losses_discr[phase] += loss_discr.item() * batch_size
                    losses_act[phase] += loss_act.item() * batch_size

                    if training:
                        loss_discr.backward()
                        loss_act.backward()
                        loss.backward()
                        optimizer.step()

                    # Prepare metrics
                    score = score.view(-1, args.num_classes)
                    target = target.view(-1, args.num_classes)
                    score = softmax(score).cpu().detach().numpy()
                    target = target.cpu().detach().numpy()
                    score_metrics[phase].extend(score)
                    target_metrics[phase].extend(target)

                    # Prepare metrics
                    score_discr = score_discr.view(-1, 2)
                    target_discr = target_discr.view(-1, 2)
                    score_discr = softmax(score_discr).cpu().detach().numpy()
                    target_discr = target_discr.cpu().detach().numpy()
                    score_metrics_discr[phase].extend(score_discr)
                    target_metrics_discr[phase].extend(target_discr)

                    # Prepare metrics
                    score_act = score_act.view(-1, args.num_classes)
                    #target = target.view(-1, args.num_classes)
                    score_act = softmax(score_act).cpu().detach().numpy()
                    #target = target.cpu().detach().numpy()
                    score_metrics_act[phase].extend(score_act)
                    target_metrics_act[phase].extend(target)

                    if training:
                        writer.add_scalar('Loss_iter/train_enc', loss.item(), batch_idx_train)
                        batch_idx_train += 1
                    else:
                        writer.add_scalar('Loss_iter/val_enc', loss.item(), batch_idx_test)
                        batch_idx_test += 1

                    print('[{:5s}] Epoch: {:2}  Iteration: {:3}  Loss: {:.5f}'.format(phase,
                                                                                      epoch,
                                                                                      batch_idx,
                                                                                      loss.item()))
        end = time.time()

        writer.add_scalars('Loss_epoch/train_val_enc',
                           {phase: losses[phase] / len(data_loaders[phase].dataset) for phase in args.phases},
                           epoch)
        writer.add_scalars('Loss_Discr_epoch/train_val_enc',
                           {phase: losses_discr[phase] / len(data_loaders[phase].dataset) for phase in args.phases},
                           epoch)
        writer.add_scalars('Loss_Act_epoch/train_val_enc',
                           {phase: losses_act[phase] / len(data_loaders[phase].dataset) for phase in args.phases},
                           epoch)

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
        result_file = {phase: 'phase-{}-epoch-{}-discr.json'.format(phase, epoch) for phase in args.phases}
        mAP_discr = {phase: utl.compute_result_multilabel(
            ['Background', 'Action'],
            score_metrics_discr[phase],
            target_metrics_discr[phase],
            save_dir,
            result_file[phase],
            ignore_class=[],
            switch=False,
            save=True,
        ) for phase in args.phases}
        result_file = {phase: 'phase-{}-epoch-{}-act.json'.format(phase, epoch) for phase in args.phases}
        mAP_act = {phase: utl.compute_result_multilabel(
            args.class_index,
            score_metrics_act[phase],
            target_metrics_act[phase],
            save_dir,
            result_file[phase],
            ignore_class=[0, 21],
            save=True,
        ) for phase in args.phases}

        writer.add_scalars('mAP_epoch/train_val_enc', {phase: mAP[phase] for phase in args.phases}, epoch)
        writer.add_scalars('mAP_Discr_epoch/train_val_enc', {phase: mAP_discr[phase] for phase in args.phases}, epoch)
        writer.add_scalars('mAP_Act_epoch/train_val_enc', {phase: mAP_act[phase] for phase in args.phases}, epoch)


        log = 'Epoch: {:2} | [train] enc_avg_loss: {:.5f}  enc_mAP: {:.4f} |'
        log += ' [test] enc_avg_loss: {:.5f}  enc_mAP: {:.4f}  |\n'
        log += 'running_time: {:.2f} sec'
        log = str(log).format(epoch,
                              losses['train'] / len(data_loaders['train'].dataset),
                              mAP['train'],
                              losses['test'] / len(data_loaders['test'].dataset),
                              mAP['test'],
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