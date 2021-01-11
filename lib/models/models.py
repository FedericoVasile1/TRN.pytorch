from .rnn import RNNmodel
from .cnn3d import CNN3D
from .cnn2d import CNN2D
from .rnn_attention import RNNAttention
from .bidirectionalrnn import BIDIRECTIONALRNN
from .transformer import Transformer

_META_ARCHITECTURES = {
    'LSTM': RNNmodel,
    'GRU': RNNmodel,
    'CNN3D': CNN3D,
    'CNN2D': CNN2D,
    'LSTMATTENTION': RNNAttention,
    'GRUATTENTION': RNNAttention,
    'BIDIRECTIONALLSTM': BIDIRECTIONALRNN,
    'BIDIRECTIONALGRU': BIDIRECTIONALRNN,
    'TRANSFORMER': Transformer,
}

def build_model(args):
    meta_arch = _META_ARCHITECTURES[args.model]
    return meta_arch(args)
