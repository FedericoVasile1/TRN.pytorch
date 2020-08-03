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

    args.device = device
    model = build_model(args)
    if osp.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.apply(utl.weights_init)
    if args.distributed:
        model = nn.DataParallel(model)
    model = model.to(device)

    criterion_let = nn.CrossEntropyLoss(ignore_index=21).to(device)
    criterion_le0 = nn.CrossEntropyLoss(ignore_index=21).to(device)
    criterion_lc = utl.ContrastiveLoss().to(device)
    criterion_la = nn.CrossEntropyLoss(ignore_index=21).to(device)
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
                for batch_idx, (camera_inputs, _, enc_target, _) in enumerate(data_loaders[phase], start=1):
                    # camera_inputs.shape == (batch_size, enc_steps, feat_vect_dim)
                    # enc_target.shape == (batch_size, enc_steps, num_classes)
                    batch_size = camera_inputs.shape[0]
                    camera_inputs = camera_inputs.to(device)

                    if training:
                        optimizer.zero_grad()

                    # forward pass
                    score, ptes, p0es, xtes, x0es = model(camera_inputs)            # score.shape == (batch_size, enc_steps, num_classes)

                    score = score.to(device)
                    target = enc_target.to(device)
                    # sum losses along all timesteps
                    loss_la = criterion_la(score[:, 0], target[:, 0].max(axis=1)[1])
                    for step in range(1, camera_inputs.shape[1]):
                        loss_la += criterion_la(score[:, step], target[:, step].max(axis=1)[1])
                    loss_la /= camera_inputs.shape[1]      # scale by enc_steps

                    ptes = ptes.to(device)
                    # sum losses along all timesteps
                    loss_let = criterion_let(ptes[:, 0], target[:, 0].max(axis=1)[1])
                    for step in range(1, camera_inputs.shape[1]):
                        loss_let += criterion_let(ptes[:, step], target[:, step].max(axis=1)[1])
                    loss_let /= camera_inputs.shape[1]  # scale by enc_steps

                    p0es = p0es.to(device)
                    # sum losses along all timesteps
                    loss_le0 = criterion_le0(p0es[:, 0], target[:, -1].max(axis=1)[1])
                    for step in range(1, camera_inputs.shape[1]):
                        loss_le0 += criterion_le0(p0es[:, step], target[:, -1].max(axis=1)[1])
                    loss_le0 /= camera_inputs.shape[1]  # scale by enc_steps

                    xtes = xtes.to(device)
                    x0es = x0es.to(device)
                    loss_lc = criterion_lc(xtes, x0es, target)

                    loss = loss_la + 1.0 * ((loss_let + loss_le0) + loss_lc)

                    losses[phase] += loss.item() * batch_size

                    if training:
                        loss.backward()
                        optimizer.step()

                    # Prepare metrics
                    score = score.view(-1, args.num_classes)
                    target = target.view(-1, args.num_classes)
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

                    print('[{:5s}] Epoch: {:2}  Iteration: {:3}  Loss: {:.5f}'.format(phase,
                                                                                      epoch,
                                                                                      batch_idx,
                                                                                      loss.item()))
        end = time.time()

        writer.add_scalars('Loss_epoch/train_val_enc',
                           {phase: losses[phase] / len(data_loaders[phase].dataset) for phase in args.phases},
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

        writer.add_scalars('mAP_epoch/train_val_enc', {phase: mAP[phase] for phase in args.phases}, epoch)

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