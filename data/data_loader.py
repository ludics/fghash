import torch
import os

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from PIL import Image, ImageFile

import data.cifar10 as cifar10
import data.nus_wide as nuswide
import data.flickr25k as flickr25k
import data.imagenet as imagenet
import data.cub_2011 as cub2011
import data.nabirds as nabirds

from data.transform import train_transform, encode_onehot

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_data(dataset, root, num_query, num_train, batch_size, num_workers, sampler=None):
    """
    Load dataset.

    Args
        dataset(str): Dataset name.
        root(str): Path of dataset.
        num_query(int): Number of query data points.
        num_train(int): Number of training data points.
        num_workers(int): Number of loading data threads.

    Returns
        query_dataloader, train_dataloader, retrieval_dataloader(torch.utils.data.DataLoader): Data loader.
    """
    if dataset == 'cifar-10':
        query_dataloader, train_dataloader, retrieval_dataloader = cifar10.load_data(root,
                                     num_query,
                                     num_train,
                                     batch_size,
                                     num_workers,
                                     )
    elif dataset == 'nus-wide-tc10':
        query_dataloader, train_dataloader, retrieval_dataloader = nuswide.load_data(10,
                                     root,
                                     num_query,
                                     num_train,
                                     batch_size,
                                     num_workers,
                                     )
    elif dataset == 'nus-wide-tc21':
        query_dataloader, train_dataloader, retrieval_dataloader = nuswide.load_data(21,
                                     root,
                                     num_query,
                                     num_train,
                                     batch_size,
                                     num_workers
                                     )
    elif dataset == 'flickr25k':
        query_dataloader, train_dataloader, retrieval_dataloader = flickr25k.load_data(root,
                                       num_query,
                                       num_train,
                                       batch_size,
                                       num_workers,
                                       )
    elif dataset == 'imagenet':
        query_dataloader, train_dataloader, retrieval_dataloader = imagenet.load_data(root,
                                      batch_size,
                                      num_workers,
                                      )
    elif dataset == 'cub-2011':
        if sampler == 'PK':
            query_dataloader, train_dataloader, retrieval_dataloader = cub2011.load_data(root,
                                        80,
                                        num_workers,
                                        'PK'
                                        )
        else:
            query_dataloader, train_dataloader, retrieval_dataloader = cub2011.load_data(root,
                                        batch_size,
                                        num_workers,
                                        )
    elif dataset == 'nabirds':
        query_dataloader, train_dataloader, retrieval_dataloader = nabirds.load_data(root,
                                      batch_size,
                                      num_workers,
                                      )
    else:
        raise ValueError("Invalid dataset name!")

    return query_dataloader, train_dataloader, retrieval_dataloader


def sample_dataloader(dataloader, num_samples, batch_size, root, dataset):
    """
    Sample data from dataloder.

    Args
        dataloader (torch.utils.data.DataLoader): Dataloader.
        num_samples (int): Number of samples.
        batch_size (int): Batch size.
        root (str): Path of dataset.
        sample_index (int): Sample index.
        dataset(str): Dataset name.

    Returns
        sample_dataloader (torch.utils.data.DataLoader): Sample dataloader.
    """
    data = dataloader.dataset.data
    targets = dataloader.dataset.targets

    sample_index = torch.randperm(data.shape[0])[:num_samples]
    data = data[sample_index]
    targets = targets[sample_index]
    sample = wrap_data(data, targets, batch_size, root, dataset)

    return sample, sample_index


def wrap_data(data, targets, batch_size, root, dataset):
    """
    Wrap data into dataloader.

    Args
        data (np.ndarray): Data.
        targets (np.ndarray): Targets.
        batch_size (int): Batch size.
        root (str): Path of dataset.
        dataset(str): Dataset name.

    Returns
        dataloader (torch.utils.data.dataloader): Data loader.
    """
    class MyDataset(Dataset):
        def __init__(self, data, targets, root, dataset):
            self.data = data
            self.targets = targets
            self.root = root
            self.transform = train_transform()
            self.dataset = dataset
            if dataset == 'cifar-10':
                self.onehot_targets = encode_onehot(self.targets, 10)
            else:
                self.onehot_targets = self.targets

        def __getitem__(self, index):
            if self.dataset == 'cifar-10':
                img = Image.fromarray(self.data[index])
                if self.transform is not None:
                    img = self.transform(img)
            else:
                img = Image.open(os.path.join(self.root, self.data[index])).convert('RGB')
                img = self.transform(img)
            return img, self.targets[index], index

        def __len__(self):
            return self.data.shape[0]

        def get_onehot_targets(self):
            """
            Return one-hot encoding targets.
            """
            return torch.from_numpy(self.onehot_targets).float()

    dataset = MyDataset(data, targets, root, dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    return dataloader
