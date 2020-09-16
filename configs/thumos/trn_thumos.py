from configs import parse_base_args, build_data_info

__all__ = ['parse_trn_args']

def parse_trn_args():
    parser = parse_base_args()
    parser.add_argument('--data_root', default='data/THUMOS', type=str)
    parser.add_argument('--model', default='TRN', type=str)
    parser.add_argument('--inputs', default='camera', type=str)
    parser.add_argument('--hidden_size', default=1024, type=int)
    parser.add_argument('--camera_feature', default='resnet3d_112x112', type=str)
    parser.add_argument('--motion_feature', default='', type=str)
    parser.add_argument('--enc_steps', default=64, type=int)
    parser.add_argument('--dec_steps', default=8, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--downsample_backgr', action='store_true')
    parser.add_argument('--neurons', default=128, type=int)
    parser.add_argument('--feat_vect_dim', default=-1, type=int)
    parser.add_argument('--feature_extractor', default='', type=str)
    parser.add_argument('--put_linear', action='store_true')
    parser.add_argument('--show_predictions', action='store_true')
    parser.add_argument('--seed_show_predictions', default=-1, type=int)
    parser.add_argument('--chunk_size', default=6, type=int)
    parser.add_argument('--reduce_lr_epoch', default=-1, type=int)
    parser.add_argument('--reduce_lr_count', default=-1, type=int)
    return build_data_info(parser.parse_args())
