import torch
from torchvision import models
from torchvision.models import resnet50, ResNet50_Weights, VGG16_Weights
import timm


def initialize_model(args):
    """
    Initializes a specified model for image classification.

    Args:
        model_name (str): The name of the model to initialize ('resnet50', 'vgg16', 'efficientnet_b0', 'vit_base_patch16_224').
        input_shape (tuple): The shape of the input image (channels, height, width).
        num_classes (int): The number of classes for the classification task.

    Returns:
        A PyTorch model initialized with the specified number of output classes.
    """
    model_name = args["architecture"]
    num_classes = args["num_classes"]
    model = None
    if model_name == 'resnet50':
        weights = ResNet50_Weights.IMAGENET1K_V1  # or ResNet50_Weights.DEFAULT for the most up-to-date
        model = resnet50(weights=weights)
    elif model_name == 'vgg16':
        model = models.vgg16(weights=VGG16_Weights.DEFAULT)
    elif model_name == 'efficientnet_b0':
        # Check if efficientnet_b0 is available in torchvision, otherwise load from timm
        model = models.efficientnet_b0(pretrained=True) if hasattr(models, 'efficientnet_b0') else timm.create_model('efficientnet_b0', pretrained=True)
    elif model_name == 'vit_base_patch16_224':
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
    else:
        raise ValueError(f"Model {model_name} not recognized.")

    # Adapt the model's classifier to the number of classes, if necessary
    if model_name in ['resnet50', 'efficientnet_b0', 'vit_base_patch16_224']:
        in_features = model.fc.in_features if model_name != 'vit_base_patch16_224' else model.head.in_features
        if model_name == 'vit_base_patch16_224':
            model.head = torch.nn.Linear(in_features, num_classes)
        else:
            model.fc = torch.nn.Linear(in_features, num_classes)
    elif model_name == 'vgg16':
        in_features = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(in_features, num_classes)

    # Ensure the model is in evaluation mode (you would switch to .train() mode during training)
    model.eval()

    return model


if __name__=="main__":
    # Example usage
    model_name = 'resnet50'  # Choose from 'resnet50', 'vgg16', 'efficientnet_b0', 'vit_base_patch16_224'
    input_shape = (3, 224, 224)  # Example input shape
    num_classes = 10  # Example number of classes

    model = initialize_model(model_name, input_shape, num_classes)
    print(model)
