from .thumos_data_layer import TRNTHUMOSDataLayer
from .thumos_data_layer_e2e import TRNTHUMOSDataLayerE2E

_DATA_LAYERS = {
    'TRNTHUMOS': TRNTHUMOSDataLayer,
    'TRNTHUMOSE2E': TRNTHUMOSDataLayerE2E,
    'LSTMTHUMOS': TRNTHUMOSDataLayer,
    'LSTMTHUMOSE2E': TRNTHUMOSDataLayerE2E,
    'ORACLELSTMTHUMOS': TRNTHUMOSDataLayer,
    'ORACLELSTMTHUMOSE2E': TRNTHUMOSDataLayerE2E,
    'MYTRNTHUMOS': TRNTHUMOSDataLayer,
    'MYTRNTHUMOSE2E': TRNTHUMOSDataLayerE2E,
    'RESNET2+1DTHUMOSE2E': TRNTHUMOSDataLayerE2E,
    'FC_ACTHUMOS': TRNTHUMOSDataLayer,
    'FUTURELSTMTHUMOS': TRNTHUMOSDataLayer,
}

def build_dataset(args, phase):
    data_layer = _DATA_LAYERS[args.model + args.dataset + ('E2E' if args.camera_feature == 'video_frames_24fps' else '')]
    return data_layer(args, phase)
