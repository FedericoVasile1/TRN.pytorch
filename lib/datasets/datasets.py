from .thumos_data_layer import THUMOSDataLayer
from .thumos_data_layer_e2e import THUMOSDataLayerE2E
from .judo_data_layer import JUDODataLayer
from .judo_data_layer_e2e import JUDODataLayerE2E
from .goodpoints_judo_data_layer import Goodpoints_JUDODataLayer

_DATA_LAYERS = {
    'LSTMTHUMOS': THUMOSDataLayer,
    'LSTMTHUMOSE2E': THUMOSDataLayerE2E,
    'GRUTHUMOS': THUMOSDataLayer,
    'GRUTHUMOSE2E': THUMOSDataLayerE2E,
    'CNN3DTHUMOSE2E': THUMOSDataLayerE2E,
    'CNNTHUMOSE2E': THUMOSDataLayerE2E,

    'LSTMJUDO': JUDODataLayer,
    'LSTMJUDOE2E': JUDODataLayerE2E,
    'GRUJUDO': JUDODataLayer,
    'GRUJUDOE2E': JUDODataLayerE2E,
    'CNN3DJUDOE2E': JUDODataLayerE2E,
    'CNNJUDOE2E': JUDODataLayerE2E,
    'BIDIRECTIONALLSTMJUDO': JUDODataLayer,
    'BIDIRECTIONALGRUJUDO': JUDODataLayer,

    'ENCDECLSTMJUDO': JUDODataLayer,
    'ENCDECBIDIRECTIONALLSTMJUDO': JUDODataLayer,
    'ENCDECGRUJUDO': JUDODataLayer,
    'ENCDECBIDIRECTIONALGRUJUDO': JUDODataLayer,

    'GRUJUDOGOODPOINTS': Goodpoints_JUDODataLayer,

    'CONVLSTMJUDOE2E': JUDODataLayerE2E,
}

def build_dataset(args, phase):
    data_layer = _DATA_LAYERS[args.model + args.dataset + args.E2E + args.goodpoints]
    return data_layer(args, phase)
