from .hdd_data_layer import TRNHDDDataLayer
from .thumos_data_layer import TRNTHUMOSDataLayer
from .thumos_data_layer_e2e import TRNTHUMOSDataLayerE2E, TRNTHUMOSDataLayerVGG
from .thumos_data_layer_cnn3d import CNN3DTHUMOSDataLayer

_DATA_LAYERS = {
    'TRNHDD': TRNHDDDataLayer,
    'TRNTHUMOS': TRNTHUMOSDataLayer,
    'LSTMTHUMOS': TRNTHUMOSDataLayer,
    'LSTMV2THUMOS': TRNTHUMOSDataLayer,
    'TRN2V2THUMOS': TRNTHUMOSDataLayer,
    'TRN2V2THUMOSE2E': TRNTHUMOSDataLayerE2E,
    'CNN3DTHUMOS': CNN3DTHUMOSDataLayer,
    'TRNTHUMOSVGG': TRNTHUMOSDataLayerVGG,
}

def build_dataset(args, phase):
    data_layer = _DATA_LAYERS[args.model + args.dataset + ('E2E' if args.E2E else '') +
                              ('VGG' if args.camera_feature == 'video_frames_24fps' else '')]
    return data_layer(args, phase)
