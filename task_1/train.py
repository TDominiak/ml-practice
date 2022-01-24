import copy
import time
from typing import List

import torch
import torchvision
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import models
from tqdm import tqdm

from task_1.dataset import DatasetConfig
from task_1.log import logger


def _run_training_steps(model: torchvision.models, criterion: nn.modules, optimizer: torch.optim,
                        scheduler: torch.optim, num_epochs: int, device: torch.device,
                        dataset_config: DatasetConfig) -> torchvision.models:
    epoch_counter_train: List = []
    epoch_counter_val: List = []
    train_loss: List = []
    val_loss: List = []
    train_acc: List = []
    val_acc: List = []
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    dataset_sizes = {x: len(dataset_config.image_datasets[x]) for x in ['train', 'val']}
    for epoch in range(num_epochs):
        logger.info('Epoch {}/{}'.format(epoch + 1, num_epochs))
        logger.info('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            with tqdm(dataset_config.data_loaders[phase], unit="batch") as t_epoch:
                for idx, (inputs, labels) in enumerate(t_epoch):
                    t_epoch.set_description(f"Epoch {epoch}")
                    # setup a timer for the train
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    t_epoch.set_postfix(loss=loss.item())

            # For graph generation
            if phase == "train":
                train_loss.append(running_loss / dataset_sizes[phase])
                train_acc.append(running_corrects.double() / dataset_sizes[phase])
                epoch_counter_train.append(epoch)
            if phase == "val":
                val_loss.append(running_loss / dataset_sizes[phase])
                val_acc.append(running_corrects.double() / dataset_sizes[phase])
                epoch_counter_val.append(epoch)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == "train":
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == "val":
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
            logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logger.info('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)

    return model


def get_model(length_class_names: int) -> torchvision.models:
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, length_class_names)

    return model_ft


def train_model(device: torch.device, num_epochs: int, dataset_config: DatasetConfig) -> torchvision.models:
    length_class_names = len(dataset_config.image_datasets['train'].classes)
    model_ft = get_model(length_class_names)
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    # NOTE: For a production solution, I would keep each hardcoded in config.yaml and I would change the next one into
    # a dataclass
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001, betas=(0.9, 0.999))
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    return _run_training_steps(
        model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs, device, dataset_config)
