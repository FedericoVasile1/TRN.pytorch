import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from configs.build import build_data_info

def _candidates_plot_bar(classes, values, xlabel=None, ylabel=None, figsize=(8, 5), color='b', title=None, show_bar=False, save_bar=False):
    figure = plt.figure(figsize=figsize)
    rects = plt.bar(classes, values, color=color)
    plt.title(title, color='black')
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    # this function is to put the value on top of its corresponding column
    def autolabel(rects):
        # attach some text labels
        for ii, rect in enumerate(rects):
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2., 1.0 * height, '%s' % (values[ii]), ha='center', va='bottom')
    autolabel(rects)

    if show_bar:
        plt.show()
    if save_bar:
        plt.savefig('bar_stats_'+title+'.png')
        print('Bar saved at ' + os.path.join(os.getcwd(), 'bar_stats_'+title+'.png'))
    return figure

def _goodpoints_print_stats_classes(args):
    class_to_count = {}
    for i in range(args.num_classes):
        class_to_count[i] = 0

    valid_samples = None
    if args.phase != '':
        valid_samples = getattr(args, args.phase + '_session_set')

    TARGETS_BASE_DIR = os.path.join(args.data_root, args.target_labels_dir)
    tot_samples = 0
    for video_name in os.listdir(TARGETS_BASE_DIR):
        if '.npy' not in video_name:
            continue
        if valid_samples is not None and video_name.split('___')[1][:-4] not in valid_samples:
            continue

        target = np.load(os.path.join(TARGETS_BASE_DIR, video_name))
        num_frames = target.shape[0]
        num_frames = num_frames - (num_frames % args.chunk_size)
        target = target[:num_frames]
        # For each chunk, take only the central frame
        target = target[args.chunk_size // 2::args.chunk_size]

        target = target.argmax(axis=1)
        unique, counts = np.unique(target, return_counts=True)
        tot_samples += counts.sum()
        for idx, idx_class in enumerate(unique):
            class_to_count[idx_class] += counts[idx]

    if valid_samples is not None:
        print('=== PHASE: {} ==='.format(args.phase))
    else:
        print('=== ALL GOODPOINTS ===')
    for idx_class, count in class_to_count.items():
        class_name = args.class_index[idx_class]
        print('{:15s}=>  samples: {:8} ({:.1f} %)'.format(class_name, count, count / tot_samples * 100))
    if args.show_bar or args.save_bar:
        _candidates_plot_bar(args.class_index,
                 [round(i / tot_samples, 3) for i in list(class_to_count.values())],
                 'Class',
                 'Percentage',
                 title=args.dataset,
                 show_bar=args.show_bar,
                 save_bar=args.save_bar)

if __name__ == '__main__':
    base_dir = os.getcwd()
    base_dir = base_dir.split('/')[-1]
    CORRECT_LAUNCH_DIR = 'TRN.pytorch'
    if base_dir != CORRECT_LAUNCH_DIR:
        raise Exception('Wrong base dir, this file must be run from ' + CORRECT_LAUNCH_DIR + ' directory.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/JUDO', type=str)
    parser.add_argument('--data_info', default='data/data_info.json', type=str)
    parser.add_argument('--chunk_size', default=9, type=int)
    parser.add_argument('--target_labels_dir', default='goodpoints_4s_target_frames_25fps', type=str)
    parser.add_argument('--phase', default='', type=str)
    parser.add_argument('--show_bar', default=False, action='store_true')
    parser.add_argument('--save_bar', default=False, action='store_true')
    args = parser.parse_args()

    if not os.path.isdir(os.path.join(args.data_root)):
        raise Exception('{} not found'.format(os.path.join(args.data_root)))
    if not os.path.isfile(os.path.join(args.data_info)):
        raise Exception('{} not found'.format(os.path.join(args.data_info)))
    if args.phase not in ('train', 'val', 'test', ''):
        raise Exception('Wrong --phase argument. Expected one of: train|val|test')

    # do not modify
    args.use_untrimmed = True
    args.use_trimmed = False
    args.use_candidates = True
    args.eval_on_untrimmed = False
    args = build_data_info(args, basic_build=True)

    args.data_root = args.data_root + '/' + 'UNTRIMMED'
    args.train_session_set = args.train_session_set['UNTRIMMED']
    args.val_session_set = args.val_session_set['UNTRIMMED']
    args.test_session_set = args.test_session_set['UNTRIMMED']

    if not os.path.isdir(os.path.join(args.data_root, args.target_labels_dir)):
        raise Exception('{} not found'.format(os.path.join(args.data_root, args.target_labels_dir)))

    _goodpoints_print_stats_classes(args)