from .generalized_trn import GeneralizedTRN
from .lstm import LSTMmodel, LSTMmodelV2
from .trn2 import TRN2V2

_META_ARCHITECTURES = {
    'TRN': GeneralizedTRN,
    'LSTM': LSTMmodel,
    'LSTMV2': LSTMmodelV2,
    'TRN2V2': TRN2V2,
}

def build_model(args):
    meta_arch = _META_ARCHITECTURES[args.model]
    return meta_arch(args)
