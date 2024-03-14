import torch.nn as nn
import torchvision
from methods.SimpleModel import SimpleModel
from methods.ETFModel import ETFModel
from methods.DERModel import DERModel
from methods.EWCModel import EWCModel
from methods.MEMOModel import MEMOModel
from methods.SCRModel import SCRModel
from methods.MIRModel import MIRModel
from methods.XDERModel import XDERModel
from methods.ODDLModel import ODDLModel
from methods.NCFSCIL import NCFSCIL
from methods.REMINDModel import REMINDModel
from methods.POLRSModel import POLRS


def get_backbone(
    name: str = "resnet18",
    image_size: int = 224,
    neck: str = "default",
):
    """
    return a nn.Module instance containing the following attributes:
    - num_features
    """
    net = torchvision.models.__dict__[name]()
    net.num_features = net.fc.weight.shape[1]
    net.fc = nn.Identity()
    net.neck = neck

    if image_size == 32 or image_size == 64:
        net.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        net.maxpool = nn.Identity()

    return net


def get_model(
    method: str = "memory_only",
    backbone: str = "resnet18",
    neck: str = "default",
    image_size: int = 224,
    **kwargs,
):
    """
    return a Model (nn.Module) instance that outputs
    - loss = model(batch) when model.train()
    - acc  = model(batch) when model.eval()
    """

    if method in ["memory_only", "er"]:
        return SimpleModel(get_backbone(name=backbone, image_size=image_size, neck=neck), num_classes=kwargs.get("num_classes", 10))
    
    elif method == "der":
        return DERModel(get_backbone(backbone, image_size=image_size, neck=neck), num_classes=kwargs.get("num_classes", 10))
    
    elif method == "ewc":
        return EWCModel(get_backbone(backbone, image_size=image_size, neck=neck), num_classes=kwargs.get("num_classes", 10))
    
    elif method == "oddl":
        return ODDLModel(get_backbone(backbone, image_size=image_size, neck=neck), num_classes=kwargs.get("num_classes", 10), device=kwargs.get("device"), lr=kwargs.get("lr"), dataset=kwargs.get("dataset_name"))
    
    elif method == "memo":
        return MEMOModel(get_backbone(backbone, image_size=image_size, neck=neck), num_classes=kwargs.get("num_classes", 10))

    elif method == "etf":
        return ETFModel(get_backbone(backbone, image_size=image_size, neck=neck), num_classes=kwargs.get("num_classes", 10), residual_addition=kwargs.get("residual_addition", True), residual_num=kwargs.get("residual_num", 50), knn_top_k=kwargs.get("knn_top_k", 50))
    
    elif method == "scr":
        return SCRModel(get_backbone(backbone, image_size=image_size, neck=neck), num_classes=kwargs.get("num_classes", 10))
    
    elif method == "mir":
        return MIRModel(get_backbone(backbone, image_size=image_size, neck=neck), num_classes=kwargs.get("num_classes", 10))

    elif method == "ncfscil":
        return NCFSCIL(get_backbone(backbone, image_size=image_size, neck=neck), num_classes=kwargs.get("num_classes", 10), spatial_feat_dim=kwargs.get("feat_dim", 8), num_codebooks=kwargs.get("num_codebooks", 32), codebook_size=kwargs.get("codebook_size", 256))
    
    elif method == "remind":
        return REMINDModel(get_backbone(backbone, image_size=image_size, neck=neck), num_classes=kwargs.get("num_classes", 10), spatial_feat_dim=kwargs.get("feat_dim", 8), num_codebooks=kwargs.get("num_codebooks", 32), codebook_size=kwargs.get("codebook_size", 256))

    elif method == "xder":
        return XDERModel(get_backbone(backbone, image_size=image_size, neck=neck), num_classes=kwargs.get("num_classes", 10), device=kwargs.get("device"), dataset=kwargs.get("dataset_name"))
    
    elif method == "polrs":
        return POLRS(get_backbone(backbone, image_size=image_size, neck=neck), num_classes=kwargs.get("num_classes", 10), init_lr=kwargs.get("lr"), train_num=kwargs.get("train_num"))
