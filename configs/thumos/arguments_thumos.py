from configs import parse_base_args, build_data_info

__all__ = ['parse_model_args']

def parse_model_args():
    parser = parse_base_args()
    parser.add_argument('--data_root', default='data/THUMOS', type=str)
    parser.add_argument('--phases', default=['train', 'test'], type=list)

    parser.add_argument('--model', default='GRU', type=str)
    parser.add_argument('--hidden_size', default=1024, type=int)
    # number of unrolling steps for rnn
    parser.add_argument('--steps', default=64, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    # whether or not to put a linear later between feature extractor and temporal model
    parser.add_argument('--put_linear', action='store_true')
    # neurons of the linear layer above
    parser.add_argument('--neurons', default=128, type=int)

    # folder where the pre-extracted features(or raw frames) are
    parser.add_argument('--features_preextracted', default='i3d_224x224_chunk9', type=str)
    # in case of starting from pre-extracted features, this is the dimension of the feature vector
    parser.add_argument('--feat_vect_dim', default=-1, type=int)
    # in case of starting from raw frames, indicate here the feature extractor to use
    parser.add_argument('--feature_extractor', default='', type=str)
    parser.add_argument('--chunk_size', default=9, type=int)

    parser.add_argument('--show_predictions', action='store_true')
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--video_name', default='', type=str)

    parser.add_argument('--reduce_lr_epoch', default=-1, type=int)
    parser.add_argument('--reduce_lr_count', default=30, type=int)
    return build_data_info(parser.parse_args())
