from typing import List

import torch
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
from torch.utils.data import DataLoader
import torchvision

from task_1.log import logger


# NOTE: I would use more metrics when developing a real solution
def test_model(model: torchvision.models, device, data_loader: DataLoader):
    # Test the accuracy with test data
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    logger.info('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))


