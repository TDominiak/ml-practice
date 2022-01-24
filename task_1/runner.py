import argparse
import os
from argparse import Namespace

import torch

from task_1.dataset import get_data_transformation, get_image_dataset, DatasetConfig
from task_1.log import logger
from task_1.test_model import test_model
from task_1.train import train_model, get_model


def __parse_config() -> Namespace:
    parser = argparse.ArgumentParser(
        description='ArgumentParser')
    parser.add_argument(
        '-d', '--data_dir', help='Directory of input data',
        required=True)
    parser.add_argument(
        '-m', '--trained_model_path', help='Path to trained model',
        required=False)
    parser.add_argument(
        '-e', '--num_epochs', help='Number of maximum epochs', default=2,
        required=False)
    args = parser.parse_args()

    return args


def main() -> None:
    config = __parse_config()
    if not os.path.exists(os.path.join(config.data_dir, 'train')) or not \
            os.listdir(os.path.join(config.data_dir, 'train')):
        raise Exception("Folder doesn't exist or is empty! Yse 'splitfolders --ratio .8 .1 .1 -- data/input' "
                        "to create appropriate folder structure")
    data_transformation = get_data_transformation()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if not config.trained_model_path:
        sets = ['train', 'val']
        dataset_config = DatasetConfig(
            sets=sets,
            data_transformation=data_transformation,
            image_datasets=get_image_dataset(data_transformation, config.data_dir, sets))
        model = train_model(device, config.num_epochs, dataset_config)
        logger.info('\nSaving the model...')
        model_path = './task_1/result/trained_model.pth'
        torch.save(model.state_dict(), model_path)
        config.trained_model_path = model_path
    sets = ['test']
    dataset_config = DatasetConfig(
        sets=sets,
        data_transformation=data_transformation,
        image_datasets=get_image_dataset(data_transformation, config.data_dir, sets))
    model = get_model(len(dataset_config.image_datasets['test'].classes))
    model.load_state_dict(torch.load(config.trained_model_path))
    model.eval()
    test_model(model, device, dataset_config.data_loaders['test'])


if __name__ == '__main__':
    main()
