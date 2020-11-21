from .rnn import RNNmodel
from .cnn3d import CNN3D
from .cnn import CNN
from .rnn_attention import RNNAttention
from .bidirectionalrnn import BIDIRECTIONALRNN
from .rnnback import RNNBackmodel
from .autoencoder.video_CAE import  VideoAutoencoderLSTM

_META_ARCHITECTURES = {
    'LSTM': RNNmodel,
    'GRU': RNNmodel,
    'CNN3D': CNN3D,
    'CNN': CNN,
    'LSTMATTENTION': RNNAttention,
    'GRUATTENTION': RNNAttention,
    'BIDIRECTIONALLSTM': BIDIRECTIONALRNN,
    'BIDIRECTIONALGRU': BIDIRECTIONALRNN,

    'GRUBACK': RNNBackmodel,

    'CAE': VideoAutoencoderLSTM,
}

def build_model(args):
    meta_arch = _META_ARCHITECTURES[args.model]
    return meta_arch(args)
