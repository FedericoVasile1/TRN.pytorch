from .rnn import RNNmodel
from .cnn3d import CNN3D
from .cnn import CNN
from .rnn_attention import RNNAttention
from .bidirectionalgru import BIDIRECTIONALGRU

_META_ARCHITECTURES = {
    'LSTM': RNNmodel,
    'GRU': RNNmodel,
    'CNN3D': CNN3D,
    'CNN': CNN,
    'LSTMATTENTION': RNNAttention,
    'GRUATTENTION': RNNAttention,
    'BIDIRECTIONALGRU': BIDIRECTIONALGRU,
}

def build_model(args):
    meta_arch = _META_ARCHITECTURES[args.model]
    return meta_arch(args)
