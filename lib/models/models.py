from .generalized_trn import GeneralizedTRN
from .lstm import LSTMmodel
from .mytrn import MyTRN
from .cnn3d import CNN3D
from .discriminator_lstm import DiscriminatorLSTM
from .discriminator_cnn3d import DiscriminatorCNN3D
from .discr_act_lstm import DiscrActLSTM
from .discr_act_2_lstm import DiscrActLSTM2
from .discriminator_cnn import DiscriminatorCNN
from .convlstm import ConvLSTM
from .cnn import CNN
from .fc import FC
from .discr_act_2_convlstm import DiscrActConvLSTM2
from .idu import IDU

_META_ARCHITECTURES = {
    'TRN': GeneralizedTRN,
    'LSTM': LSTMmodel,
    'MYTRN': MyTRN,
    'RESNET2+1D': CNN3D,
    'DISCRIMINATORLSTM': DiscriminatorLSTM,
    'DISCRIMINATORCNN3D': DiscriminatorCNN3D,
    'DISCRACTLSTM': DiscrActLSTM,
    'DISCRACTLSTM2': DiscrActLSTM2,
    'DISCRIMINATORCNN': DiscriminatorCNN,
    'CONVLSTM': ConvLSTM,
    'DISCRIMINATORCONVLSTM': ConvLSTM,
    'CNN': CNN,
    'FC': FC,
    'DISCRACTCONVLSTM2': DiscrActConvLSTM2,
    'IDU': IDU,
}

def build_model(args):
    meta_arch = _META_ARCHITECTURES[args.model]
    return meta_arch(args)
