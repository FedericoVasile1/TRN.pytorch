from .generalized_trn import GeneralizedTRN
from .lstm import LSTMmodel
from .mytrn import MyTRN
from .cnn3d import CNN3D
from .fc_actiondetector import FC_AC
from .futurelstm import FutureLSTM
from .disciminator_lstm import DiscriminatorLSTM

_META_ARCHITECTURES = {
    'TRN': GeneralizedTRN,
    'LSTM': LSTMmodel,
    'ORACLELSTM': LSTMmodel,
    'MYTRN': MyTRN,
    'RESNET2+1D': CNN3D,
    'FUTURELSTM': FutureLSTM,
    'DISCRIMINATORLSTM': DiscriminatorLSTM,
}

def build_model(args):
    meta_arch = _META_ARCHITECTURES[args.model]
    return meta_arch(args)
