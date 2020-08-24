from .generalized_trn import GeneralizedTRN
from .lstm import LSTMmodel
from .cnn3d import CNN3D
from .discriminator_lstm import DiscriminatorLSTM
from .discriminator_cnn3d import DiscriminatorCNN3D
from .discriminator_cnn import DiscriminatorCNN
from .convlstm import ConvLSTM
from .cnn import CNN
from .idu import IDU
from .rulstm import RULSTM
from .lstm_attention import LSTMAttention
from .tc_lstm import TCLSTM

_META_ARCHITECTURES = {
    'TRN': GeneralizedTRN,
    'LSTM': LSTMmodel,
    'RESNET2+1D': CNN3D,
    'DISCRIMINATORLSTM': DiscriminatorLSTM,
    'DISCRIMINATORCNN3D': DiscriminatorCNN3D,
    'DISCRIMINATORCNN': DiscriminatorCNN,
    'CONVLSTM': ConvLSTM,
    'DISCRIMINATORCONVLSTM': ConvLSTM,
    'CNN': CNN,
    'IDU': IDU,
    'STARTENDLSTM': LSTMmodel,
    'RULSTM': RULSTM,
    'LSTMATTENTION': LSTMAttention,
    'TCLSTM': TCLSTM,
}

def build_model(args):
    meta_arch = _META_ARCHITECTURES[args.model]
    return meta_arch(args)
