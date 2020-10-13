from .generalized_trn import GeneralizedTRN
from .rnn import RNNmodel
from .cnn3d import CNN3D
from .convlstm import ConvLSTM
from .cnn import CNN
from .idu import IDU
from .rulstm import RULSTM
from .rnn_attention import RNNAttention
from .dcc_rnn import DCCRNN

_META_ARCHITECTURES = {
    'TRN': GeneralizedTRN,
    'LSTM': RNNmodel,
    'GRU': RNNmodel,
    'GRUMULTITASK': RNNmodel,
    'CNN3D': CNN3D,
    'CONVLSTM': ConvLSTM,
    'DISCRIMINATORCONVLSTM': ConvLSTM,
    'CNN': CNN,
    'IDU': IDU,
    'STARTENDLSTM': RNNmodel,
    'STARTENDGRU': RNNmodel,
    'RULSTM': RULSTM,
    'LSTMATTENTION': RNNAttention,
    'GRUATTENTION': RNNAttention,
    'DCCLSTM': DCCRNN,
    'DCCGRU': DCCRNN,
}

def build_model(args):
    meta_arch = _META_ARCHITECTURES[args.model]
    return meta_arch(args)
