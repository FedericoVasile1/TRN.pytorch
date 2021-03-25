from configs import parse_base_args, build_data_info

__all__ = ['parse_model_args']

def parse_model_args():
    parser = parse_base_args()
    # WARNING: will be used te UNTRIMMED JUDO dataset! see configs.base_configs.py line 15
    parser.add_argument('--data_root', default='data/JUDO', type=str)
    parser.add_argument('--phases', default=['train', 'val'], type=list)

    parser.add_argument('--model', default='GRU', type=str)
    parser.add_argument('--hidden_size', default=256, type=int)
    # number of unrolling steps for rnn
    parser.add_argument('--steps', default=32, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)

    # whether or not to put a linear layer between feature extractor and temporal model
    parser.add_argument('--put_linear', default=False, action='store_true')
    # neurons of the linear layer above
    parser.add_argument('--neurons', default=512, type=int)

    # folder where the pre-extracted features (or raw frames) are
    parser.add_argument('--model_input', default='i3d_224x224_chunk9', type=str)
    parser.add_argument('--model_target', default='4s_target_frames_25fps', type=str)

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

    # downsampling or not the action classes.
    # Specify as parameter the number of samples to which each class will be downsampled
    parser.add_argument('--downsampling', default=-1, type=int)

    parser.add_argument('--use_candidates', default=False, action='store_true')

    # only for transformer model
    parser.add_argument('--nhead', default=2, type=int)
    parser.add_argument('--num_layers_transformer', default=1, type=int)

    # only for dilated causal convolution model
    parser.add_argument('--num_layers_dcc', default=1, type=int)
    parser.add_argument('--kernel_sizes', default='2,2,2', type=str)
    parser.add_argument('--dilatation_rates', default='1,2,4', type=str)
    parser.add_argument('--num_filters', default='1024,512,512', type=str)

    return build_data_info(parser.parse_args())
