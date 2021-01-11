from .thumos_data_layer import THUMOSDataLayer
from .thumos_data_layer_e2e import THUMOSDataLayerE2E
from .judo_data_layer import JUDODataLayer
from .judo_data_layer_e2e import JUDODataLayerE2E
from .candidates_judo_data_layer import Candidates_JUDODataLayer
from .candidates_judo_data_layer_e2e import Candidates_JUDODataLayerE2E

_DATA_LAYERS = {
    'LSTMTHUMOS': THUMOSDataLayer,
    'LSTMTHUMOSE2E': THUMOSDataLayerE2E,
    'GRUTHUMOS': THUMOSDataLayer,
    'GRUTHUMOSE2E': THUMOSDataLayerE2E,
    'CNN3DTHUMOSE2E': THUMOSDataLayerE2E,
    'CNN2DTHUMOSE2E': THUMOSDataLayerE2E,

    'CNN3DJUDOE2E': JUDODataLayerE2E,
    'CNN2DJUDOE2E': JUDODataLayerE2E,
    'LSTMJUDO': JUDODataLayer,
    'GRUJUDO': JUDODataLayer,
    'BIDIRECTIONALLSTMJUDO': JUDODataLayer,
    'BIDIRECTIONALGRUJUDO': JUDODataLayer,
    'LSTMATTENTIONJUDO': JUDODataLayer,
    'GRUATTENTIONJUDO': JUDODataLayer,
    'TRANSFORMERJUDO': JUDODataLayer,

    'CNN3DJUDOE2ECANDIDATES': Candidates_JUDODataLayerE2E,
    'CNN2DJUDOE2ECANDIDATES': Candidates_JUDODataLayerE2E,
    'LSTMJUDOCANDIDATES': Candidates_JUDODataLayer,
    'GRUJUDOCANDIDATES': Candidates_JUDODataLayer,
    'BIDIRECTIONALLSTMJUDOCANDIDATES': Candidates_JUDODataLayer,
    'BIDIRECTIONALGRUJUDOCANDIDATES': Candidates_JUDODataLayer,
    'LSTMATTENTIONJUDOCANDIDATES': Candidates_JUDODataLayer,
    'GRUATTENTIONJUDOCANDIDATES': Candidates_JUDODataLayer,
    'TRANSFORMERJUDOCANDIDATES': Candidates_JUDODataLayer,
}

def build_dataset(args, phase):
    data_layer = _DATA_LAYERS[args.model + args.dataset + args.E2E + args.candidates]
    return data_layer(args, phase)
