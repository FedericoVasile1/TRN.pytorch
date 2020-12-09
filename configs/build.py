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
        if args.use_trimmed == args.use_untrimmed == False:
            raise Exception('At least one between --use_trimmed and --use_untrimmed must be used')
        if args.eval_on_untrimmed and (args.use_trimmed == False or args.use_untrimmed == False):
            raise Exception('Wrong --eval_on_untrimmed option. With --eval_on_untrimmed you'
                            'must have --use_trimmed and --use_untrimmed')

        args.train_session_set, args.val_session_set, args.test_session_set = ({}, {}, {})
        with open(args.data_info, 'r') as f:
            data_info = json.load(f)[args.dataset]
            if args.use_trimmed:
                args.train_session_set['TRIMMED'] = data_info['TRIMMED']['train_session_set']
                args.val_session_set['TRIMMED'] = data_info['TRIMMED']['val_session_set']
                args.test_session_set['TRIMMED'] = data_info['TRIMMED']['test_session_set']
            if args.use_untrimmed:
                args.train_session_set['UNTRIMMED'] = data_info['UNTRIMMED']['train_session_set']
                args.val_session_set['UNTRIMMED'] = data_info['UNTRIMMED']['val_session_set']
                args.test_session_set['UNTRIMMED'] = data_info['UNTRIMMED']['test_session_set']
    else:
        raise Exception('Wrong --data_root option. Unknow dataset')

    with open(args.data_info, 'r') as f:
        data_info = json.load(f)[args.dataset]
        args.class_index = data_info['class_index']
        args.num_classes = len(args.class_index)

    if basic_build:
        return args

    # the raw frames folder always starts with this string
    #  e.g. video_frames_25fps/1.jpg
    #       video_frames_25fps/2.jpg
    #       .....
    BASE_FOLDER_RAW_FRAMES = 'video_frames_'
    # the features pre-extracted folder finishes with this string
    #  e.g. i3d_224x224_chunk9/video1.npy
    #       i3d_224x224_chunk9/video2.npy
    #       ....
    #       where videoN.npy.shape == (num_frames_in_video/chunk_size, feature_vector_dim)
    BASE_FOLDER_FEAT_EXTR = '_chunk'+str(args.chunk_size)

    if args.model_input.startswith(BASE_FOLDER_RAW_FRAMES):
        if args.chunk_size == -1:
            raise Exception('Wrong --chunk_size option. Specify the number of consecutive frames that the feature '
                            'extractor will take as input at a time(e.g. --chunk_size 16)')
        # we do end to end learning(i.e. start from raw frames)
        args.E2E = 'E2E'
        # depending on the type of the feature extractor(i.e. 2D or 3D) we will have a different
        #  behavior for the Dataset class(e.g. check lib/datasets/judo_data_layer.__getitem__)
        args.is_3D = True if args.model in ('CNN3D', 'CONVLSTM') else False
    elif not args.model_input.endswith(BASE_FOLDER_FEAT_EXTR):
        raise Exception('Wrong --model_input or --chunk_size option. They must indicate the same chunk size')
    elif args.feat_vect_dim == -1:
        raise Exception('Wrong --feat_vect_dim option. When starting from features pre-extracted, the dimension '
                        'of the feature vector must be specified')
    else:
        # no end to end learning because we are starting from features pre-extracted
        args.E2E = ''

    if args.use_goodpoints and args.use_candidates:
        raise Exception('--use_goodpoints and --use_candidates can not be true together')
    if args.use_goodpoints:
        if 'goodpoints' not in args.model_input or 'goodpoints' not in args.model_target:
            raise Exception('With --use_goodpoints option you must provide input goodpoints features'
                            '(via --model_input option)  and target goodpoint(via --model_target option)')
        #if not args.use_untrimmed or args.use_trimmed:
        #    raise Exception('We actually have goodpoints features and targets only for untrimmed dataset')
        if args.E2E == 'E2E':
            raise Exception('We actually do not support goodpoints starting from raw frames')

        args.goodpoints = 'GOODPOINTS'
    else:
        args.goodpoints = ''
    if args.use_candidates:
        if 'candidates' not in args.model_input or 'candidates' not in args.model_target:
            raise Exception('With --use_candidates option you must provide input candidates features'
                            '(via --model_input option)  and target candidates(via --model_target option)')
        if not args.use_untrimmed or args.use_trimmed:
            raise Exception('We actually have candidates features and targets only for untrimmed dataset')
        if args.E2E == 'E2E':
            raise Exception('We actually do not support candidates starting from raw frames')

        args.candidates = 'CANDIDATES'
    else:
        args.candidates = ''

    return args
