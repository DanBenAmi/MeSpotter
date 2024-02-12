import argparse
import yaml
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from datetime import datetime

from train_utils.initialize_lr_scheduler import initialize_lr_scheduler
from train_utils.initialize_optimizer import initialize_optimizer
from CustomImageDataset import CustomImageDataset
from models import get_model
from augmentations import get_augmentations
from utils import load_data, save_checkpoint


def train(config):
    # extract training parameters from config file
    epochs = config["train"]["epochs"]
    batch_size = config["train"]["batch_size"]
    train_augmentations = get_augmentations(config["train"]["augmentations"])

    # Prepare device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # create datasets
    train_set = CustomImageDataset(labels_table_path=config["train"]["labels_table_path"],
                       img_dir=config[image_dir_path],
                       transform=train_augmentations,
                       num_classes=config["model"]["num_classes"])

    valid_set = CustomImageDataset(labels_table_path=config["valid"]["labels_table_path"],
                                   img_dir=config[image_dir_path],
                                   num_classes=config["model"]["num_classes"])

    # Load data
    train_loader = DataLoader(train_set, batch_size=config['train']['batch_size'], shuffle=True, num_workers=config['train']["num_workers"])
    valid_loader = DataLoader(valid_set, batch_size=config['valid']['batch_size'], shuffle=False, num_workers=config['valid']["num_workers"])

    # Initialize model
    net = get_model(config).to(device)

    # Optimizer, lr-scheduler and loss
    optimizer = initialize_optimizer(config["train"]["optimizer"], net.parameters(), config["train"]["optimizer_params"])
    lr_scheduler = initialize_lr_scheduler(config["train"]["lr_scheduler"], optimizer, config["train"]["lr_scheduler_params"])
    criterion = nn.CrossEntropyLoss()

    # TensorBoard
    writer = SummaryWriter()

    for epoch in range(config['training']['epochs']):
        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:  # log every 10 mini-batches
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 10:.3f}")
                writer.add_scalar('training loss', running_loss / 10, epoch * len(train_loader) + i)
                running_loss = 0.0

    print('Finished Training')
    save_path = f"{config['model']['architecture']}__{datetime.now().strftime('%Y%m%d%H%M%S')}"
    os.makedirs(save_path, exist_ok=True)
    torch.save(net.state_dict(), f"{save_path}/model_final.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument('--config', type=str, required=True, help='Path to config.yaml')
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    train(config)
