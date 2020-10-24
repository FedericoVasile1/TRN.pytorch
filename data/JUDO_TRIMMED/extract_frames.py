import csv
import os
import argparse

def main(args):
    if not os.path.isdir(os.path.join(args.data_root)):
        raise Exception('{} not found'.format(os.path.join(args.data_root)))
    if not os.path.isdir(os.path.join(args.data_root, args.raw_videos_dir)):
        raise Exception('{} not found'.format(os.path.join(args.data_root, args.raw_videos_dir)))
    if not os.path.isdir(os.path.join(args.data_root, args.extracted_frames_dir)):
        os.makedir(os.path.join(args.data_root, args.extracted_frames_dir))
        print('{} folder created'.format(os.path.join(args.data_root, args.extracted_frames_dir)))

    SUBFOLDERS = ['Ashi', 'Koshi', 'MaSutemi', 'Te', 'YokoSutemi']
    for folder in SUBFOLDERS:
        filename_list = os.listdir(os.path.join(args.data_root, args.raw_videos_dir, folder))

        for filename in filename_list:
            print('Processing video {}'.format(filename))
            os.mkdir(os.path.join(args.data_root, args.extracted_frames_dir, folder, filename))

            filename = filename.replace(' ', '\ ')
            path_rawvideo = os.path.join(args.data_root, args.raw_videos_dir, folder, filename)
            path_savedir = os.path.join(args.data_root, args.extracted_frames_dir, folder, filename) + '/'
            os.system('ffmpeg -i '+path_rawvideo+' -r 25.0 '+path_savedir+'%d.jpg')

if __name__ == '__main__':
    base_dir = os.getcwd()
    base_dir = base_dir.split('/')[-1]
    CORRECT_LAUNCH_DIR = 'TRN.pytorch'
    if base_dir != CORRECT_LAUNCH_DIR:
        raise Exception('Wrong base dir, this file must be run from '+CORRECT_LAUNCH_DIR+' directory.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/JUDO_TRIMMED', type=str)
    # the directory where the raw videos are(i.e. .mp4 videos, we will extract the frames of these videos.)
    parser.add_argument('--raw_videos_dir', default='NageWaza', type=str)
    # the fps at which extract video frames
    parser.add_argument('--fps', default=25, type=int)
    # the directory where frames of video will be stored. Will be appended the fps information
    #  at the end of the directory name, e.g. {--extracted_frames_dir}_25fps
    parser.add_argument('--extracted_frames_dir', default='video_frames', type=str)
    args = parser.parse_args()

    args.extracted_frames_dir = args.extracted_frames_dir + '_{}fps'.format(str(args.fps))

    main(args)
