import os.path as osp
import json

__all__ = ['build_data_info']

def build_data_info(args, basic_build=False):
    args.dataset = osp.basename(osp.normpath(args.data_root))
    with open(args.data_info, 'r') as f:
        data_info = json.load(f)[args.dataset]
    args.train_session_set = data_info['train_session_set']
    args.val_session_set = data_info['val_session_set'] if 'val_session_set' in data_info else None
    args.test_session_set = data_info['test_session_set']
    args.class_index = data_info['class_index']
    args.num_classes = len(args.class_index)

    if basic_build:
        return args

    if args.inputs == 'motion':
        raise Exception('Optical flow only is not supported')
    if args.inputs == 'multistream' and args.motion_feature == '':
        raise Exception('No --motion_features option provided; with --inputs == multistream a --motion_feature'
                        'option must be provided.')
    if args.camera_feature == 'resnet3d_featuremaps' and (args.inputs != 'camera' or args.motion_feature != ''):
        raise Exception('The current pipeline do not support rgb and flow fusion between feature maps.'
                        'Use rgb feature maps only or switch to feature vectors if you want to do '
                        'the fusion.')
    if 'video_frames_' in args.camera_feature and (args.inputs != 'camera' or args.motion_feature != ''):
        raise Exception('Currently the end to end mode is available for rgb input.')

    args.E2E = 'E2E' if 'video_frames_' in args.camera_feature else ''

    if args.feature_extractor == 'RESNET2+1D' \
            or args.feature_extractor == 'I3D' \
            or args.model == 'CNN3D' \
            or args.model == 'DISCRIMINATORCNN3D' \
            or (args.model == 'CONVLSTM' and args.feature_extractor == 'RESNET2+1D') \
            or (args.model == 'DISCRIMINATORCONVLSTM' and args.feature_extractor == 'RESNET2+1D'):
        args.is_3D = True
    else:
        args.is_3D = False

    if args.dataset == 'THUMOS' and args.camera_feature == 'i3d_224x224_chunk9' and args.chunk_size != 9:
        raise Exception('Wrong chunk_size option. With camera_feature == i3d_224x224_chunk9, you must '
                        'have chunk_size == 9')

    return args
