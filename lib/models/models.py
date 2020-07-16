from .generalized_trn import GeneralizedTRN
from .lstm import LSTMmodel
from .mytrn import MyTRN
from .cnn3d import CNN3D
from .discriminator_lstm import DiscriminatorLSTM
from .discriminator_cnn3d import DiscriminatorCNN3D
from .discr_act_lstm import DiscrActLSTM
from .discriminator_cnn import DiscriminatorCNN
from .convlstm import ConvLSTM

_META_ARCHITECTURES = {
    'TRN': GeneralizedTRN,
    'LSTM': LSTMmodel,
    'MYTRN': MyTRN,
    'RESNET2+1D': CNN3D,
    'DISCRIMINATORLSTM': DiscriminatorLSTM,
    'DISCRIMINATORCNN3D': DiscriminatorCNN3D,
    'DISCRACTLSTM': DiscrActLSTM,
    'DISCRIMINATORCNN': DiscriminatorCNN,
    'CONVLSTM': ConvLSTM,
    'DISCRIMINATORCONVLSTM': ConvLSTM,
}

def build_model(args):
    meta_arch = _META_ARCHITECTURES[args.model]
    return meta_arch(args)
