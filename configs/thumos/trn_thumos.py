from configs import parse_base_args, build_data_info

__all__ = ['parse_trn_args']

def parse_trn_args():
    parser = parse_base_args()
    parser.add_argument('--data_root', default='data/THUMOS', type=str)
    parser.add_argument('--model', default='TRN', type=str)
    parser.add_argument('--hidden_size', default=4096, type=int)
    # indicate the base directory where the frames are
    parser.add_argument('--camera_frames_root', default='video_frames_24fps', type=str)
    parser.add_argument('--camera_target_root', default='target_frames_24fps', type=str)
    # when starting from the frames, indicates the model that will act
    #  as feature extractor
    parser.add_argument('--feat_extr', default='vgg16', type=str)    # e.g. resnet18
    # True if we want the feature extractor model as trainable, false otherwise
    parser.add_argument('--feat_extr_trainable', action='store_true')
    parser.add_argument('--enc_steps', default=64, type=int)
    parser.add_argument('--dec_steps', default=8, type=int)
    parser.add_argument('--sample_frames', default=6, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--data_aug', action='store_true')
    return build_data_info(parser.parse_args())
