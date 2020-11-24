from .rnn import RNNmodel
from .cnn3d import CNN3D
from .cnn import CNN
from .rnn_attention import RNNAttention
from .bidirectionalrnn import BIDIRECTIONALRNN
from .encdec import EncDec

_META_ARCHITECTURES = {
    'LSTM': RNNmodel,
    'GRU': RNNmodel,
    'CNN3D': CNN3D,
    'CNN': CNN,
    'LSTMATTENTION': RNNAttention,
    'GRUATTENTION': RNNAttention,
    'BIDIRECTIONALLSTM': BIDIRECTIONALRNN,
    'BIDIRECTIONALGRU': BIDIRECTIONALRNN,

    'ENCDECLSTM': EncDec,
}

def build_model(args):
    meta_arch = _META_ARCHITECTURES[args.model]
    return meta_arch(args)
