import os
import os.path as osp
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim

import _init_paths
from lib import utils as utl
from configs.thumos import parse_trn_args as parse_args
from lib.models import build_model

def main(args):
    this_dir = osp.join(osp.dirname(__file__), '.')
    save_dir = osp.join(this_dir, 'checkpoints')
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    command = 'python ' + ' '.join(sys.argv)
    logger = utl.setup_logger(osp.join(this_dir, 'log.txt'), args.phases, command=command)
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

    criterion = utl.MultiCrossEntropyLoss(ignore_index=21).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if osp.isfile(args.checkpoint):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
        args.start_epoch += checkpoint['epoch']
    softmax = nn.Softmax(dim=1).to(device)

    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        if epoch == 21:
            args.lr = args.lr * 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr

        # TODO: try to move it outside of the epochs loop. Now this is here
        #  due to the data augmentation made in TRNTHUMOSDataLayer.__init__ because
        #  data augmentation must be done at the start of every epoch.
        #  Try to move data_aug in __getitem__, in this way we could move these line
        #  outside of the loop
        data_loaders = {
            phase: utl.build_data_loader(args, phase)
            for phase in args.phases
        }

        enc_losses = {phase: 0.0 for phase in args.phases}
        enc_score_metrics = {phase: [] for phase in args.phases}
        enc_target_metrics = {phase: [] for phase in args.phases}
        enc_mAP = {phase: 0.0 for phase in args.phases}
        dec_losses = {phase: 0.0 for phase in args.phases}
        dec_score_metrics = {phase: [] for phase in args.phases}
        dec_target_metrics = {phase: [] for phase in args.phases}
        dec_mAP = {phase: 0.0 for phase in args.phases}

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
                for batch_idx, (camera_inputs, enc_target, dec_target) \
                        in enumerate(data_loaders[phase], start=1):
                    # camera_inputs.shape:(batch_size, num_frames_in_video, 3, H, W)
                    batch_size = camera_inputs.shape[0]
                    camera_inputs = camera_inputs.to(device)
                    enc_target = enc_target.to(device).view(-1, args.num_classes)
                    dec_target = dec_target.to(device).view(-1, args.num_classes)

                    enc_score, dec_score = model(camera_inputs)
                    enc_loss = criterion(enc_score, enc_target)
                    dec_loss = criterion(dec_score, dec_target)
                    enc_losses[phase] += enc_loss.item() * batch_size
                    dec_losses[phase] += dec_loss.item() * batch_size
                    if args.verbose:
                        print('[{}]Epoch: {:2} | iteration: {:3} | enc_loss: {:.5f} dec_loss: {:.5f}'.format(
                            phase, epoch, batch_idx, enc_loss.item(), dec_loss.item()
                        ))

                    if training:
                        optimizer.zero_grad()
                        loss = enc_loss + dec_loss
                        loss.backward()
                        optimizer.step()

                    # Prepare metrics for encoder
                    enc_score = softmax(enc_score).cpu().detach().numpy()
                    enc_target = enc_target.cpu().detach().numpy()
                    enc_score_metrics[phase].extend(enc_score)
                    enc_target_metrics[phase].extend(enc_target)
                    # Prepare metrics for decoder
                    dec_score = softmax(dec_score).cpu().detach().numpy()
                    dec_target = dec_target.cpu().detach().numpy()
                    dec_score_metrics[phase].extend(dec_score)
                    dec_target_metrics[phase].extend(dec_target)

        end = time.time()

        if args.debug:
            result_file = 'epoch-{}.json'.format(epoch)
            # Compute result for encoder
            enc_mAP = {phase: utl.compute_result_multilabel(
                args.class_index,
                enc_score_metrics[phase],
                enc_target_metrics[phase],
                save_dir,
                result_file,
                ignore_class=[0,21],
                save=True,
                verbose=False,
            ) for phase in args.phases}
            # Compute result for decoder
            dec_mAP = {phase: utl.compute_result_multilabel(
                args.class_index,
                dec_score_metrics[phase],
                dec_target_metrics[phase],
                save_dir,
                result_file,
                ignore_class=[0,21],
                save=False,
                verbose=False
            ) for phase in args.phases}

        # Output result
        logger.output(epoch, enc_losses, dec_losses,
                      {phase: len(data_loaders[phase].dataset) for phase in args.phases},
                      enc_mAP, dec_mAP, end - start, debug=args.debug)

        if not args.no_save:
            if args.save_last == False or epoch == (args.start_epoch + args.epochs - 1):
                # Save model
                checkpoint_file = 'epoch-{}.pth'.format(epoch)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if args.distributed else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, osp.join(save_dir, checkpoint_file))

if __name__ == '__main__':
    main(parse_args())