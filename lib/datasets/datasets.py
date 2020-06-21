from .hdd_data_layer import TRNHDDDataLayer
from .thumos_data_layer import TRNTHUMOSDataLayer

_DATA_LAYERS = {
    'TRNHDD': TRNHDDDataLayer,
    'TRNTHUMOS': TRNTHUMOSDataLayer,
    'LSTMTHUMOS': TRNTHUMOSDataLayer,
    'LSTMV2THUMOS': TRNTHUMOSDataLayer,
    'TRN2V2THUMOS': TRNTHUMOSDataLayer,
}

def build_dataset(args, phase):
    data_layer = _DATA_LAYERS[args.model + args.dataset]
    return data_layer(args, phase)
