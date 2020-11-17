from configs import parse_base_args, build_data_info

__all__ = ['parse_model_args']

def parse_model_args():
    parser = parse_base_args()
    parser.add_argument('--data_root', default='data/JUDO', type=str)
    parser.add_argument('--use_trimmed', action='store_true')
    parser.add_argument('--use_untrimmed', action='store_true')
    # if True, trimmed dataset will be all used for training(together with untrimmed training set), while
    #  validation and test will be done on val/test untrimmed sets
    parser.add_argument('--eval_on_untrimmed', action='store_true')
    parser.add_argument('--phases', default=['train', 'val'], type=list)

    parser.add_argument('--model', default='GRU', type=str)
    parser.add_argument('--hidden_size', default=1024, type=int)
    # number of unrolling steps for rnn
    parser.add_argument('--steps', default=16, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    # whether or not to put a linear layer between feature extractor and temporal model
    parser.add_argument('--put_linear', action='store_true')
    # neurons of the linear layer above
    parser.add_argument('--neurons', default=128, type=int)

    # folder where the pre-extracted features or raw frames are
    parser.add_argument('--model_input', default='i3d_224x224_chunk9', type=str)
    # in case of starting from pre-extracted features, must specify the dimension of the feature vector
    parser.add_argument('--feat_vect_dim', default=-1, type=int)
    # in case of starting from raw frames, must specify the feature extractor to use
    parser.add_argument('--feature_extractor', default='', type=str)
    # the feature extractor takes chunk_size frames in input and generate a feature vector.
    #  Common values: 9, 12, 16
    parser.add_argument('--chunk_size', default=-1, type=int)

    parser.add_argument('--show_predictions', action='store_true')
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--video_name', default='', type=str)

    parser.add_argument('--reduce_lr_epoch', default=-1, type=int)
    parser.add_argument('--reduce_lr_count', default=30, type=int)

    parser.add_argument('--downsample_backgr', action='store_true')

    parser.add_argument('--use_goodpoint', default=False, action='store_true')

    return build_data_info(parser.parse_args())
