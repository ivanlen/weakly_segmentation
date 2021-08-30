from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


def create_model(output_classes):
    """
    https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/

    :param output_classes: segmentation model output classes number
    :return: torch deeplabv3_resnet101 model ready to be trained
    """
    model = deeplabv3_resnet101(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    # backbone resnet101 has output shape 2048
    model.classifier = DeepLabHead(2048, output_classes)
    model.train()
    return model
