import os.path as osp
import json

__all__ = ['build_data_info']

def build_data_info(args, basic_build=False):
    args.dataset = osp.basename(osp.normpath(args.data_root))

    if args.dataset == 'THUMOS':
        with open(args.data_info, 'r') as f:
            data_info = json.load(f)[args.dataset]
        args.train_session_set = data_info['train_session_set']
        args.test_session_set = data_info['test_session_set']
    elif args.dataset == 'JUDO':
        # HARD-CODED: we will use JUDO/UNTRIMMED dataset
        args.data_root += '/UNTRIMMED'      # == 'data/JUDO/UNTRIMMED'

        with open(args.data_info, 'r') as f:
            data_info = json.load(f)[args.dataset]
            args.train_session_set = data_info['UNTRIMMED']['train_session_set']
            args.val_session_set = data_info['UNTRIMMED']['val_session_set']
            args.test_session_set = data_info['UNTRIMMED']['test_session_set']
    else:
        raise Exception('Wrong --data_root option. Unknow dataset')

    with open(args.data_info, 'r') as f:
        data_info = json.load(f)[args.dataset]
        args.class_index = data_info['class_index']
        args.num_classes = len(args.class_index)

    if basic_build:
        return args

    # the raw frames folder always starts(or contains) with this string
    #  e.g. video_frames_25fps/1.jpg
    #       video_frames_25fps/2.jpg
    #       .....
    BASE_FOLDER_RAW_FRAMES = 'video_frames_'
    # the features pre-extracted folder finishes with this string
    #  e.g. i3d_224x224_chunk9/video1.npy
    #       i3d_224x224_chunk9/video2.npy
    #       ....
    #       where videoN.npy.shape == (num_frames_in_video//chunk_size, feature_vector_dim)
    BASE_FOLDER_FEAT_EXTR = '_chunk'+str(args.chunk_size)


    if args.use_candidates:
        if 'candidates' not in args.model_input or 'candidates' not in args.model_target:
            raise Exception('With --use_candidates option you must provide input candidates features'
                            '(via --model_input option)  and target candidates(via --model_target option)')
        args.candidates = 'CANDIDATES'
    else:
        if 'candidates' in args.model_input or 'candidates' in args.model_target:
            raise Exception('You are providing candidates input without using --use_candidates option')
        args.candidates = ''

    if args.model == 'TRANSFORMER' and (args.nhead == -1 or args.num_layers == -1):
        raise Exception('You are usinga TRANSFORMER model without specifying the number of heads and layers')

    if args.model in ('CNN2D', 'CNN3D') and args.feature_extractor == '':
        raise Exception('You must specify a feature_extractor, i.e. the specific CNN model that you want.')
    if args.model not in ('CNN2D', 'CNN3D') and args.feature_extractor != '':
        raise Exception('--feature_extractor option is actually supported only for CNN2D and CNN3D model.')

    if BASE_FOLDER_RAW_FRAMES in args.model_input:
        if args.model not in ('CNN3D', 'CNN2D'):
            raise Exception('We currently support end to end mode only for CNN3D and CNN2D models.')

        if args.chunk_size == -1:
            raise Exception('Wrong --chunk_size option. Specify the number of consecutive frames that the feature '
                            'extractor will take as input at a time(e.g. --chunk_size 16)')
        # we do end to end learning(i.e. start from raw frames)
        args.E2E = 'E2E'
        # depending on the type of the feature extractor(i.e. 2D or 3D) we will have a different
        #  behavior for the Dataset class(e.g. check lib/datasets/judo_data_layer.__getitem__)
        args.is_3D = True if args.model=='CNN3D' else False
    #elif not args.model_input.endswith(BASE_FOLDER_FEAT_EXTR):
    elif 'resnext' not in args.model_input and not args.model_input.endswith(BASE_FOLDER_FEAT_EXTR):
        raise Exception('Wrong --model_input or --chunk_size option. They must indicate the same chunk size')
    elif args.feat_vect_dim == -1:
        raise Exception('Wrong --feat_vect_dim option. When starting from features pre-extracted, the dimension '
                        'of the feature vector must be specified')
    else:
        # no end to end learning because we are starting from features pre-extracted
        args.E2E = ''

    if args.use_candidates:
        if 'candidates' not in args.model_input or 'candidates' not in args.model_target:
            raise Exception('With --use_candidates option you must provide input candidates features'
                            '(via --model_input option)  and target candidates(via --model_target option)')
        args.candidates = 'CANDIDATES'
    else:
        args.candidates = ''

    return args
