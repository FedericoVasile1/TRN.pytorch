from .hdd_data_layer import TRNHDDDataLayer
from .thumos_data_layer import TRNTHUMOSDataLayer

from torchvision import transforms

_DATA_LAYERS = {
    'TRNHDD': TRNHDDDataLayer,
    'TRNTHUMOS': TRNTHUMOSDataLayer,
}

def build_dataset(args, phase):
    data_layer = _DATA_LAYERS[args.model + args.dataset]
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return data_layer(args, transform, phase)
