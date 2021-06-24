# dataloader

import argparse
import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name',
                        type=str,
                        help='name of a dataset to train/test on',
                        default='cifar10',)
    parser.add_argument('--data_root',
                        type=str,
                        help='root directory where data is stored',
                        default='../data/',)
    parser.add_argument('--resolution',
                        type=int,
                        help='resolution of a dataset to be used',
                        default=32,)
    args = parser.parse_args()
    
    return args


# define sampler
def get_sampler(dataset, classes=[0]):
    if isinstance(classes, int): classes = [classes]
    targets = torch.tensor(dataset.targets)
    mask = torch.zeros_like(targets)
    for c in classes:
        mask += targets==c
    target_idx = mask.nonzero()
    
    sampler = torch.utils.data.sampler.SubsetRandomSampler(target_idx)
    
    return sampler
    
    
# define transform
def get_transforms(args):
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(args.resolution),
        transforms.CenterCrop(args.resolution)
    ])
    
    return data_transforms




if __name__ == '__main__':
    # Load image dataset
    # dataset_name = 'cifar10'
    # data_root = '../data/'
    args = get_args()

    dataset_classes = [ds for ds in dir(torchvision.datasets) 
                       if ds.lower() == args.dataset_name.lower()]
    assert dataset_classes, f'Dataset name {args.dataset_name} does not exist'
    dataset_class = getattr(torchvision.datasets, dataset_classes[0])

    train_dataset = dataset_class(root=args.data_root, train=True, download=True, transform=get_transforms(args))
    test_dataset = dataset_class(root=args.data_root, train=False, download=True, transform=get_transforms(args))
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=4,
        sampler=get_sampler(train_dataset, [0, 1]),
        drop_last=True,
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=4,
        sampler=get_sampler(test_dataset, [0, 1]),
        drop_last=True,
    )
    
    def plot_images(imgs):
        import matplotlib.pyplot as plt
        imgs = torchvision.utils.make_grid(imgs)
        plt.imshow(imgs.permute(1, 2, 0))
        

    imgs, labels = next(iter(train_dataloader))
    print(labels)
    plot_images(imgs)