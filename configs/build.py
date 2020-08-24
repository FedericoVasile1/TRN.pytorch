import os.path as osp
import json

__all__ = ['build_data_info']

def build_data_info(args):
    args.dataset = osp.basename(osp.normpath(args.data_root))
    with open(args.data_info, 'r') as f:
        data_info = json.load(f)[args.dataset]
    args.train_session_set = data_info['train_session_set']
    args.test_session_set = data_info['test_session_set']
    args.class_index = data_info['class_index']
    args.num_classes = len(args.class_index)

    args.E2E = 'E2E' if args.camera_feature == 'video_frames_24fps' else ''

    if args.feature_extractor == 'RESNET2+1D' or args.model == 'CNN3D' or args.model == 'DISCRIMINATORCNN3D' \
            or (args.model == 'CONVLSTM' and args.feature_extractor == 'RESNET2+1D') \
            or (args.model == 'DISCRIMINATORCONVLSTM' and args.feature_extractor == 'RESNET2+1D'):
        args.is_3D = True
    else:
        args.is_3D = False

    return args
