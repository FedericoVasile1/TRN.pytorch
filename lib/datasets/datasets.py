from .thumos_data_layer import TRNTHUMOSDataLayer
from .thumos_data_layer2 import TRNTHUMOSDataLayer2
from .thumos_data_layer_e2e import TRNTHUMOSDataLayerE2E
from .thumos_data_layer_triplelstm import TRNTHUMOSDataLayerTripleLSTM

_DATA_LAYERS = {
    'TRNTHUMOS': TRNTHUMOSDataLayer,
    'TRNTHUMOSE2E': TRNTHUMOSDataLayerE2E,
    'LSTMTHUMOS': TRNTHUMOSDataLayer,
    'LSTMTHUMOSE2E': TRNTHUMOSDataLayerE2E,
    #'MYTRNTHUMOS': TRNTHUMOSDataLayer,
    'MYTRNTHUMOS': TRNTHUMOSDataLayer2,
    'MYTRNTHUMOSE2E': TRNTHUMOSDataLayerE2E,
    'RESNET2+1DTHUMOSE2E': TRNTHUMOSDataLayerE2E,
    'DISCRIMINATORLSTMTHUMOS': TRNTHUMOSDataLayer,
    'DISCRIMINATORLSTMTHUMOSE2E': TRNTHUMOSDataLayerE2E,
    'DISCRIMINATORCNN3DTHUMOSE2E': TRNTHUMOSDataLayerE2E,
    'DISCRIMINATORCNNTHUMOSE2E': TRNTHUMOSDataLayerE2E,
    'CONVLSTMTHUMOSE2E': TRNTHUMOSDataLayerE2E,
    'DISCRIMINATORCONVLSTMTHUMOSE2E': TRNTHUMOSDataLayerE2E,
    'CNNTHUMOSE2E': TRNTHUMOSDataLayerE2E,
    'FCTHUMOS': TRNTHUMOSDataLayer,
    'DISCRACTCONVLSTM2E2E': TRNTHUMOSDataLayerE2E,
    'DISCRACTLSTMTHUMOS': TRNTHUMOSDataLayer,
    'IDUTHUMOS': TRNTHUMOSDataLayer,
    'STARTENDLSTMTHUMOS': TRNTHUMOSDataLayer,
    'TRIPLELSTMTHUMOS': TRNTHUMOSDataLayerTripleLSTM,
    'RULSTMTHUMOS': TRNTHUMOSDataLayer,
}

def build_dataset(args, phase):
    data_layer = _DATA_LAYERS[args.model + args.dataset + ('E2E' if args.camera_feature == 'video_frames_24fps' else '')]
    return data_layer(args, phase)
