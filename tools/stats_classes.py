import argparse
import os.path as osp
import json
import numpy as np

def print_stats(class_to_count, tot_samples, class_index, phase):
    print('PHASE: ', phase)
    for idx_class, name_class in enumerate(class_index):
        if idx_class == 21:     # drop ambiguous class
            continue
        print(name_class, ': {:.2f}'.format(class_to_count[idx_class]/tot_samples))
    print()

def main(args):
    args.dataset = osp.basename(osp.normpath(args.data_root))
    with open(args.data_info, 'r') as f:
        data_info = json.load(f)[args.dataset]
    #args.train_session_set = data_info['train_session_set']
    #args.test_session_set = data_info['test_session_set']
    args.train_session_set = [i for i in data_info['train_session_set'] if 'video_validation_0000690' in i]
    args.test_session_set = [i for i in data_info['train_session_set'] if 'video_validation_0000690' in i]
    args.class_index = data_info['class_index']
    args.num_classes = len(args.class_index) - 1    # drop ambiguous class

    CHUNK_SIZE = args.chunk_size
    PHASES = ['train', 'test']
    for phase in PHASES:
        tot_samples = 0
        class_to_count = {}
        for i in range(args.num_classes):
            class_to_count[i] = 0

        vid_names = getattr(args, phase+'_session_set')
        for vid_name in vid_names:
            target = np.load(osp.join(args.data_root, 'target_frames_24fps', vid_name+'.npy'))
            # round to multiple of CHUNK_SIZE
            num_frames = target.shape[0]
            num_frames = num_frames - (num_frames % CHUNK_SIZE)
            target = target[:num_frames]
            # For each chunk, take only the central frame
            target = target[CHUNK_SIZE // 2::CHUNK_SIZE]

            target = np.argmax(target, axis=1)
            count = 0
            if args.downsample_backgr == -1:
                for idx_class in target:
                    if idx_class == 21:
                        continue
                    count += 1
                    class_to_count[idx_class] += 1
            else:
                for i in range(0, target.shape[0], args.downsample_backgr):
                    appo =  target[i:i+args.downsample_backgr]
                    zeros = np.zeros_like(appo)
                    if (appo == zeros).all():
                        continue
                    for idx_class in appo:
                        if idx_class == 21:
                            continue
                        count += 1
                        class_to_count[idx_class] += 1

            tot_samples += count

        print('CHUNK_SIZE: ', args.chunk_size, '  DOWNSAMPLE_BACKGR: ', args.downsample_backgr)
        print_stats(class_to_count, tot_samples, args.class_index, phase)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/THUMOS', type=str)
    parser.add_argument('--data_info', default='data/data_info.json', type=str)
    parser.add_argument('--chunk_size', default=6, type=int)
    parser.add_argument('--downsample_backgr', default=-1, type=int)
    main(parser.parse_args())