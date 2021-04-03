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
    parser.add_argument('--dropout', default=0.1, type=float)

    # whether or not to put a linear layer between feature extractor and temporal model
    parser.add_argument('--put_linear', default=False, action='store_true')
    # neurons of the linear layer above
    parser.add_argument('--neurons', default=512, type=int)

    # folder where the pre-extracted features (or raw frames) are
    parser.add_argument('--model_input', default='i3d_224x224_chunk6', type=str)
    parser.add_argument('--model_target', default='target_frames_24fps', type=str)

    # in case of starting from pre-extracted features, must specify the dimension of the feature vector
    parser.add_argument('--feat_vect_dim', default=-1, type=int)

    # in case of starting from raw frames, indicate here the feature extractor to use
    parser.add_argument('--feature_extractor', default='', type=str)
    # the feature extractor takes chunk_size frames in input and generate a feature vector.
    #  Common values: 9, 12, 16
    parser.add_argument('--chunk_size', default=-1, type=int)

    parser.add_argument('--show_predictions', action='store_true')
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--video_name', default='', type=str)

    parser.add_argument('--reduce_lr_epoch', default=-1, type=int)
    parser.add_argument('--reduce_lr_count', default=20, type=int)

    # DO NOT MODIFY. this is useless for thumos dataset
    parser.add_argument('--use_candidates', default=False, action='store_true')

    # only for trn model
    parser.add_argument('--enc_steps', default=64, type=int)
    parser.add_argument('--dec_steps', default=8, type=int)

    # only for transformer model
    parser.add_argument('--nhead', default=2, type=int)
    parser.add_argument('--num_layers_transformer', default=1, type=int)

    # only for dilated causal convolution model
    parser.add_argument('--num_layers_dcc', default=3, type=int)
    parser.add_argument('--kernel_sizes', default='2,2,2', type=str)
    parser.add_argument('--dilatation_rates', default='1,2,4', type=str)
    parser.add_argument('--num_filters', default='1024,512,512', type=str)

    return build_data_info(parser.parse_args())
