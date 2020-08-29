from .generalized_trn import GeneralizedTRN
from .lstm import RNNmodel
from .cnn3d import CNN3D
from .convlstm import ConvLSTM
from .cnn import CNN
from .idu import IDU
from .rulstm import RULSTM
from .lstm_attention import LSTMAttention
from .dcc_lstm import DCCLSTM

_META_ARCHITECTURES = {
    'TRN': GeneralizedTRN,
    'LSTM': RNNmodel,
    'GRU': RNNmodel,
    'CNN3D': CNN3D,
    'CONVLSTM': ConvLSTM,
    'DISCRIMINATORCONVLSTM': ConvLSTM,
    'CNN': CNN,
    'IDU': IDU,
    'STARTENDLSTM': RNNmodel,
    'STARTENDGRU': RNNmodel,
    'RULSTM': RULSTM,
    'LSTMATTENTION': LSTMAttention,
    'DCCLSTM': DCCLSTM,
}

def build_model(args):
    meta_arch = _META_ARCHITECTURES[args.model]
    return meta_arch(args)
