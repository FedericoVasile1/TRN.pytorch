from .rnn import RNNmodel
from .rnn_attention import RNNAttention
from .bidirectionalrnn import BIDIRECTIONALRNN
from .transformer import Transformer
from .cnn3d import CNN3D
from .onlyact import OnlyAct

_META_ARCHITECTURES = {
    'LSTM': RNNmodel,
    'GRU': RNNmodel,
    'LSTMATTENTION': RNNAttention,
    'GRUATTENTION': RNNAttention,
    'BIDIRECTIONALLSTM': BIDIRECTIONALRNN,
    'BIDIRECTIONALGRU': BIDIRECTIONALRNN,
    'TRANSFORMER': Transformer,
    'CNN3D': CNN3D,
    'ONLYACT': OnlyAct,
}

def build_model(args):
    meta_arch = _META_ARCHITECTURES[args.model]
    return meta_arch(args)
