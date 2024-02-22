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
from tqdm import tqdm
import shutil

from train_utils.initialize_lr_scheduler import initialize_lr_scheduler
from train_utils.initialize_optimizer import initialize_optimizer
from CustomImageDataset import CustomImageDataset
from models.initialize_model import initialize_model
from augmentations import get_augmentations
from evaluate import evaluate

DEBUG = True    # tqdm progress bar and pytorch do not work with Mac M1 gpu while debugging.


def train(config):
    # extract training parameters from config file
    epochs = config["train"]["epochs"]
    batch_size = config["train"]["batch_size"]
    train_augmentations = get_augmentations(config["train"]["augmentations"], config["model"]["input_size"])
    valid_augmentations = get_augmentations(config["valid"]["augmentations"], config["model"]["input_size"])


    # Prepare device
    device = "cpu" if DEBUG else torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # create datasets
    train_set = CustomImageDataset(labels_table_path=config["train"]["labels_table_path"],
                       img_dir=config["image_dir_path"],
                       transform=train_augmentations,
                       num_classes=config["model"]["num_classes"])

    valid_set = CustomImageDataset(labels_table_path=config["valid"]["labels_table_path"],
                                   img_dir=config["image_dir_path"],
                                   transform=valid_augmentations,
                                   num_classes=config["model"]["num_classes"])

    # Load data
    train_loader = DataLoader(train_set, batch_size=config['train']['batch_size'], shuffle=True, num_workers=config['train']["num_workers"])
    valid_loader = DataLoader(valid_set, batch_size=config['valid']['batch_size'], shuffle=False, num_workers=config['valid']["num_workers"])

    # Initialize model
    net = initialize_model(config["model"])
    net = net.to(device)

    # Optimizer, lr-scheduler and loss
    optimizer = initialize_optimizer(config["train"]["optimizer"], net.parameters(), config["train"]["optimizer_params"])
    lr_scheduler = initialize_lr_scheduler(config["train"]["lr_scheduler"], optimizer, config["train"]["lr_scheduler_params"])
    criterion = nn.CrossEntropyLoss()


    # run results directory
    run_dir_name = f"{config['model']['architecture']}__{datetime.now().strftime('%Y%m%d%H%M%S')}"
    run_dir_path = os.path.join("runs", run_dir_name)
    os.makedirs(run_dir_path, exist_ok=True)
    shutil.copy(config["train"]["labels_table_path"], os.path(run_dir_path, "train_data.xlsx"))
    shutil.copy(config["valid"]["labels_table_path"], os.path(run_dir_path, "valid_data.xlsx"))
    tb_dir = os.path.join(run_dir_path, "tb")
    os.makedirs(tb_dir, exist_ok=True)
    ckpts_dir = os.path.join(run_dir_path, "ckpts")
    os.makedirs(ckpts_dir, exist_ok=True)

    # TensorBoard
    tb_writer = SummaryWriter(tb_dir)


    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = train_loader if DEBUG else tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for batch in train_bar:

            inputs, labels = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            running_loss += loss.item()

        lr_scheduler.step()

        # evaluate
        if epoch % config["train"]["epochs_between_eval"] == 0:
            evaluate(net, valid_loader, epoch, tb_writer, DEBUG=False)

        # save checkpoint
        if epoch % config["train"]["epochs_between_ckpt"] == 0:
            torch.save(net.state_dict(), os.path.join(ckpts_dir ,f"ckpt_epoch_{epoch}.pth"))


    torch.save(net.state_dict(), os.path.join(ckpts_dir, f"final_ckpt.pth"))
    tb_writer.close()
    print('Finished Training')




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument('--config', type=str, required=True, help='Path to config.yaml')
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    train(config)
