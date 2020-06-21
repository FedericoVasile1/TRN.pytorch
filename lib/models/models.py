from .generalized_trn import GeneralizedTRN
from .lstm import LSTMmodel, LSTMmodelV2

_META_ARCHITECTURES = {
    'TRN': GeneralizedTRN,
    'LSTM': LSTMmodel,
    'LSTMV2': LSTMmodelV2,
}

def build_model(args):
    meta_arch = _META_ARCHITECTURES[args.model]
    return meta_arch(args)
