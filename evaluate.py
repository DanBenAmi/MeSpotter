
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


def evaluate(net, valid_loader, epoch, tb_writer, DEBUG=False):
        net.eval()
        val_running_loss = 0.0
        correct = 0
        total = 0
        val_bar = valid_loader if DEBUG else tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{epochs} [Eval]")
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                _, true_labels = torch.max(labels, 1)
                total += labels.size(0)
                correct += (predicted == true_labels).sum().item()

        val_loss = val_running_loss / len(valid_loader)
        val_accuracy = correct / total
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}')