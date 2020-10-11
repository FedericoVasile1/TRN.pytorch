import torch
from PIL import Image
import cv2

import numpy as np
import os.path as osp
import os
import matplotlib.pyplot as plt
import io

def show_video_predictions(args,
                           video_name,
                           enc_target_metrics,
                           enc_score_metrics=None,
                           attn_weights=None,
                           frames_dir='video_frames_24fps',
                           fps=24,
                           transform=None):
    '''
    It shows the video_name together with its ground truth, its predictions(if provided), its attention
     weights(if provided)
    :param args: ParserArgument object containing main arguments
    :param video_name: string containing the name of the video
    :param enc_target_metrics: numpy array of shape(num_frames, num_classes) containing the ground truth of the video.
                                Notice that the that array that comes here is **already chunked**
    :param enc_score_metrics: numpy array of shape(num_frames, num_classes) containing the output scores of the model
    :param attn_weights:
    :param frames_dir: string containing the base folder name, i.e. under the base folder there will be
                        one folder(i.e. video_name) for each video, this will contains the frames of that video
    :param fps: int representing the fps at which the video is previously extracted(this number is needed in order
                    to correctly make the conversion from index of frame to its time in seconds)
    :return:
    '''
    if enc_score_metrics is not None:
        enc_pred_metrics = torch.argmax(torch.tensor(enc_score_metrics), dim=1)
    enc_target_metrics = torch.argmax(torch.tensor(enc_target_metrics), dim=1)

    speed = 1.0

    if args.save_video:
        print('Loading and saving video: ' + video_name + ' . . . .\n.\n.')
        frames = []
    num_frames = enc_target_metrics.shape[0]
    idx = 0
    while idx < num_frames:
        idx_frame = idx * args.chunk_size + args.chunk_size // 2
        pil_frame = Image.open(osp.join(args.data_root,
                                        frames_dir,
                                        video_name,
                                        str(idx_frame + 1) + '.jpg')).convert('RGB')
        if transform is not None:
            pil_frame = transform(pil_frame)

        open_cv_frame = np.array(pil_frame)

        # Convert RGB to BGR
        open_cv_frame = open_cv_frame[:, :, ::-1].copy()

        if args.dataset == 'JUDO':
            H, W, _ = open_cv_frame.shape
            open_cv_frame = cv2.resize(open_cv_frame, (W // 2, H // 2), interpolation=cv2.INTER_AREA)

        if attn_weights is not None:
            original_H, original_W, _ = open_cv_frame.shape
            if args.feature_extractor == 'VGG16':
                open_cv_frame = cv2.resize(open_cv_frame, (224, 224), interpolation=cv2.INTER_AREA)
            else:
                # RESNET2+1D feature extraction
                open_cv_frame = cv2.resize(open_cv_frame, (112, 112), interpolation=cv2.INTER_AREA)

            H, W, _ = open_cv_frame.shape

            attn_weights_t = attn_weights[idx]
            attn_weights_t = attn_weights_t.squeeze(0)
            attn_weights_t = cv2.resize(attn_weights_t.data.numpy().copy(), (W, H), interpolation=cv2.INTER_NEAREST)
            attn_weights_t = np.repeat(np.expand_dims(attn_weights_t, axis=2), 3, axis=2)
            attn_weights_t *= 255
            attn_weights_t = attn_weights_t.astype('uint8')
            # mask original image according to the attention weights
            open_cv_frame = cv2.addWeighted(attn_weights_t, 0.5, open_cv_frame, 0.5, 0)

            open_cv_frame = cv2.resize(open_cv_frame, (original_W, original_H), interpolation=cv2.INTER_AREA)

        open_cv_frame = cv2.copyMakeBorder(open_cv_frame, 60, 0, 30, 30, borderType=cv2.BORDER_CONSTANT, value=0)
        pred_label = args.class_index[enc_pred_metrics[idx]] if enc_score_metrics is not None else 'junk'
        target_label = args.class_index[enc_target_metrics[idx]]

        cv2.putText(open_cv_frame,
                    pred_label,
                    (0, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 0) if enc_score_metrics is None else (0, 255, 0) if pred_label == target_label else (0, 0, 255),
                    1)
        cv2.putText(open_cv_frame,
                    target_label,
                    (0, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255) if enc_score_metrics is not None else
                    (255, 255, 255) if target_label == 'Background' else (0, 0, 255),
                    1)
        cv2.putText(open_cv_frame,
                    'prob:{:.2f}'.format(
                        torch.tensor(enc_score_metrics)[idx, enc_pred_metrics[idx]].item()
                    ) if enc_score_metrics is not None else 'junk',
                    (210, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0) if enc_score_metrics is None else (0, 255, 0) if pred_label == target_label else (0, 0, 255),
                    1)

        # [ (idx_frame + 1) / 24 ]    => 24 because frames has been extracted at 24 fps
        cv2.putText(open_cv_frame,
                    '{:.2f}s'.format((idx_frame + 1) / fps),
                    (295, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 255, 255),
                    1)
        cv2.putText(open_cv_frame,
                    'speed: {}x'.format(speed),
                    (295, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 255, 255),
                    1)
        cv2.putText(open_cv_frame,
                    str(idx_frame + 1),
                    (295, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 255, 255),
                    1)

        if args.save_video:
            frames.append(open_cv_frame)
        else:
            # display the frame to screen
            cv2.imshow(video_name, open_cv_frame)
            # e.g. since video are extracted at 24 fps, we will display a frame every 1000[ms] / 24[f] = 41.6[ms]
            delay = 1000 / fps
            # since in our model we do not take all of the 24 frames, but only the central frame every chunk_size frames
            delay *= args.chunk_size
            # depending on the speed the video will be displayed faster(e.g. 2x if args.speed == 2.0) or slower
            delay /= speed
            key = cv2.waitKey(int(delay))  # time is in milliseconds
            if key == ord('q'):
                # quit
                cv2.destroyAllWindows()
                break
            if key == ord('p'):
                # pause
                cv2.waitKey(-1)  # wait until any key is pressed
            if key == ord('e'):
                # go faster
                delay /= 2
                speed *= 2
            if key == ord('w'):
                # go slower
                delay *= 2
                speed /= 2
            if key == ord('a'):
                # skip backward
                idx -= fps
                if idx < 0:
                    idx = 0
            if key == ord('s'):
                # skip forward
                idx += fps

        idx += 1

    if args.save_video:
        H, W, _ = open_cv_frame.shape
        out = cv2.VideoWriter(video_name+'.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps / args.chunk_size, (W, H))
        for frame in frames:
            out.write(frame)
        out.release()
        print('. . . video saved at ' + os.path.join(os.getcwd(), video_name+'.avi'))

def show_random_videos(args,
                       samples_list,
                       samples=1,
                       frames_dir='video_frames_24fps',
                       targets_dir='target_frames_24fps',
                       fps=24,
                       transform=None):
    '''
    It shows sampled videos from samples_list, randomly sampled. Furthermore, labels are attached to video.
    :param args: ParserArgument object containing main arguments
    :param video_name_list: a list containing the video names from which to sample a video
    :param samples: int representing the number of video to show
    :param frames_dir: string containing the base folder name, i.e. under the base folder there will be
                        one folder(i.e. video_name) for each video, this will contains the frames of that video
    :return:
    '''
    num_samples = len(samples_list)
    idx_samples = np.random.choice(num_samples, size=samples, replace=False)
    for i in idx_samples:
        video_name = samples_list[i]

        target = np.load(osp.join(args.data_root, targets_dir, video_name + '.npy'))
        num_frames = target.shape[0]
        num_frames = num_frames - (num_frames % args.chunk_size)
        target = target[:num_frames]
        target = target[args.chunk_size // 2::args.chunk_size]

        if not args.save_video:
            print('lib.utils.visualize.show_random_videos: showing video: ' + video_name)
        show_video_predictions(args, video_name, target, frames_dir=frames_dir, fps=fps, transform=transform)

def print_stats_classes(args):
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
        if valid_samples is not None and video_name[:-4] not in valid_samples:
            continue

        target = np.load(os.path.join(TARGETS_BASE_DIR, video_name))
        num_frames = target.shape[0]
        num_frames = num_frames - (num_frames % args.chunk_size)
        target = target[:num_frames]
        # For each chunk, take only the central frame
        target = target[args.chunk_size // 2::args.chunk_size]

        target_downsampled = []
        for idx in range(0, target.shape[0], args.enc_steps):
            if args.downsample_backgr:
                background_vect = np.zeros_like(target[idx:idx+args.enc_steps])
                background_vect[:, 0] = 1
                if (target[idx:idx+args.enc_steps] == background_vect).all():
                    continue
            target_downsampled.append(target[idx:idx+args.enc_steps])
        if target_downsampled == []:
            # we will enter here if and
            # only if args.downsample == True AND the current video_name has **all the frames** labeled as background
            # In this case, we skip to the next video since there is no action label to be counted
            continue
        target = np.concatenate(target_downsampled)

        target = target.argmax(axis=1)
        unique, counts = np.unique(target, return_counts=True)

        tot_samples += counts.sum()

        for idx, idx_class in enumerate(unique):
            class_to_count[idx_class] += counts[idx]

    if valid_samples is not None:
        print('=== PHASE: {} ==='.format(args.phase))
    else:
        print('=== ALL DATASET ===')
    for idx_class, count in class_to_count.items():
        class_name = args.class_index[idx_class]
        print('{:15s}=>  samples: {:8} ({:.1f} %)'.format(class_name, count, count / tot_samples * 100))
    if args.show_bar or args.save_bar:
        plot_bar(args.class_index,
                 [round(i / tot_samples, 3) for i in list(class_to_count.values())],
                 'Class',
                 'Percentage',
                 title=args.dataset,
                 show_bar=args.show_bar,
                 save_bar=args.save_bar)

def plot_bar(classes, values, xlabel=None, ylabel=None, figsize=(8, 5), color='b', title=None, show_bar=False, save_bar=False):
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

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to numpy array
    image = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()

    image = cv2.imdecode(image, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def print_stats_video(video_name, args):
    class_to_segmentdurations = {}
    for i in range(args.num_classes):
        class_to_segmentdurations[args.class_index[i]] = []
    segment_list = []

    video_targets = np.load(os.path.join(args.data_root, args.target_labels_dir, video_name))    # shape (num_frames, num_classes)
    video_targets = video_targets.argmax(axis=1)
    prev_idx_class = video_targets[0]
    duration = 1
    for idx_class in video_targets[1:]:
        if idx_class != prev_idx_class:
            class_to_segmentdurations[args.class_index[prev_idx_class]].append(duration)
            segment_list.append((args.class_index[prev_idx_class], round(duration / args.fps, 1)))
            duration = 0

        duration += 1
        prev_idx_class = idx_class

    for name_class, list_durations in class_to_segmentdurations.items():
        if len(list_durations) > 0:
            class_to_segmentdurations[name_class] = sum(list_durations) / len(list_durations)
        else:
            class_to_segmentdurations[name_class] = 0
        # convert the number of frames to number of seconds
        class_to_segmentdurations[name_class] = round(class_to_segmentdurations[name_class] / args.fps, 1)

    video_duration = round(len(video_targets) / args.fps, 1)

    return class_to_segmentdurations, segment_list, video_duration

def add_pr_curve_tensorboard(writer, class_name, class_index, labels, probs_predicted, global_step=0):
    '''
    Takes in a "class_index" and plots the corresponding
    precision-recall curve
    '''
    # Labels from all classes must be binarize to the only label of the current class
    class_labels = labels == class_index
    # For each sample, take only the probability of the current class
    class_probs_predicted = probs_predicted[:, class_index]

    writer.add_pr_curve(class_name,
                        class_labels,
                        class_probs_predicted,
                        global_step=global_step)

def get_segments(labels, class_index, fps, chunk_size):
    # labels: a list in which each element is a (num_classes,) tensor
    labels = np.argmax(np.array(labels), axis=1)
    segments_list = []

    prev_idx_class = labels[0]
    for idx_frame, idx_class in enumerate(labels[1:], start=1):
        if idx_class != prev_idx_class:
            real_idx_frame = idx_frame * chunk_size + chunk_size // 2
            time_seconds = '%.1f' % (real_idx_frame / fps)
            segments_list.append((time_seconds + ' s', class_index[idx_class]))

        prev_idx_class = idx_class
    return segments_list