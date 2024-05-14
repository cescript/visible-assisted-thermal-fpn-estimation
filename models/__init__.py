from models.SAFTAModel import SAFTAModel
from models.EMPTYModel import EMPTYModel
from models.BESTModel import BESTModel
from models.DLSNUCModel import DLSNUCModel
from models.D1WLSModel import D1WLSModel
from models.MULTIVIEWModel import MULTIVIEWModel

# generate a FPN detector model with the given options
def GenerateModel(modelName, modelOpt, dataOpt):
    if modelName == 'SAFTAModel':
        model = SAFTAModel(modelOpt, dataOpt)
    elif modelName == 'EMPTYModel':
        model = EMPTYModel(modelOpt, dataOpt)
    elif modelName == 'BESTModel':
        model = BESTModel(modelOpt, dataOpt)
    elif modelName == 'DLSNUCModel':
        model = DLSNUCModel(modelOpt, dataOpt)
    elif modelName == 'D1WLSModel':
        model = D1WLSModel(modelOpt, dataOpt)
    elif modelName == 'MULTIVIEWModel':
        model = MULTIVIEWModel(modelOpt, dataOpt)
    else:
        raise ValueError("Model {} not found." % modelName)
    # return the model
    return model