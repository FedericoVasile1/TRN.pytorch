from .generalized_trn import GeneralizedTRN
from .lstm import LSTMmodel, LSTMmodelV2
from .trn2 import TRN2V2, TRN2V2E2E

_META_ARCHITECTURES = {
    'TRN': GeneralizedTRN,
    'LSTM': LSTMmodel,
    'LSTMV2': LSTMmodelV2,
    'TRN2V2': TRN2V2,
    'TRN2V2E2E': TRN2V2E2E,
}

def build_model(args):
    meta_arch = _META_ARCHITECTURES[args.model + ('E2E' if args.E2E else '')]
    return meta_arch(args)
