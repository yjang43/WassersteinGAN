# train
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import wandb


import models


from torchvision.models.inception import inception_v3
from tqdm import tqdm
from dataloader import get_dataloader


### SET ARGS
from easydict import EasyDict
args = EasyDict({
    'model_name': 'wgan',
    'dataset_name': 'cifar10',
    'data_root': '../data/',
    'resolution': 32,
    'classes_to_include': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'batch_size': 4,
    'evaluation_size': 100,
    'device': 'cpu',
    'nz': 100,
    'ngf': 64,
    'ndf': 64,
    'nc': 3,
    'lr':0.0002,
    'total_iteration': 10
})

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',
                        type=str,
                        help='Name of the model',
                        default='wgan')
    parser.add_argument('--dataset_name',
                        type=str,
                        help='Name of the dataset to train with',
                        default='cifar10')
    parser.add_argument('--data_root',
                        type=str,
                        help='Root directory of where dataset exists',
                        default='../data/')
    parser.add_argument('--resolution',
                        type=int,
                        help='Resolution of an image',
                        default=32)
    parser.add_argument('--classes_to_include',
                        type=int,
                        nargs='+',
                        help='Classes to include in training',
                        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument('--batch_size',
                        type=int,
                        help='Size of a batch',
                        default=128)
    parser.add_argument('--evaluation_size',
                        type=int,
                        help='The number of data to evaluate image quality on',
                        default=1024)
    parser.add_argument('--device',
                        type=str,
                        help='Device to use in training',
                        default='cuda')
    parser.add_argument('--nz',
                        type=int,
                        help='The size of dimension of noise vector',
                        default=100)
    parser.add_argument('--ngf',
                        type=int,
                        help='The size of feature map for generator',
                        default=64)
    parser.add_argument('--ndf',
                        type=int,
                        help='The size of feature map for discriminator',
                        default=64)
    parser.add_argument('--nc',
                        type=int,
                        help='The number of channels, CIFAR-10 -> 3, MNIST -> 1',
                        default=3)
    parser.add_argument('--lr',
                        type=float,
                        help='Learning rate',
                        default=0.0002)
    parser.add_argument('--total_iteration',
                        type=int,
                        help='Total number of iteration for training',
                        default=5000)
    args = parser.parse_args()
    
    return args
    

# eval code here
@torch.no_grad()
def inception_score(generator, inception_model, args):
    generator.eval()
    
    upsample = nn.Upsample((399, 399), mode='bilinear')
    eval_sz = args.evaluation_size
    batch_sz = args.batch_size
    scores = []
    
    while eval_sz >=0: 
        bs = min(eval_sz, batch_sz)
        noise = torch.randn(bs, args.nz, 1, 1).to(args.device)
        imgs = generator(noise)
        imgs = upsample(imgs)
        x = inception_model(imgs)
        
        conditional_dist = F.softmax(x, dim=-1)
        marginal_dist = conditional_dist.mean(dim=0)

        for i in range(conditional_dist.size()[0]):
            score = F.kl_div(conditional_dist[i].log(),
                             marginal_dist, reduction='sum')
            scores.append(score.item())
            
        eval_sz = eval_sz - batch_sz
    
    inception_score = torch.tensor(scores).mean(0).exp().item()
    generator.train()
    
    return inception_score




def train(generator, discriminator, criterion, optimizer_g, optimizer_d,
          inception_model, train_loader, args):
    losses_d, losses_g = [], []
    
    best_generator_score = 0
    pbar = tqdm(range(args.total_iteration))
    train_iter = iter(train_loader)
    fixed_noise = torch.randn(16, args.nz, 1, 1).to(args.device)

    for itr in range(args.total_iteration):
        try:
            imgs, _ = next(train_iter)
        except StopIteration:
            train_loader = get_dataloader(args, train=True)
            train_iter = iter(train_loader)
            imgs, _ = next(train_iter)
        imgs = imgs.to(args.device)

        # 1. Update Discriminator

        optimizer_d.zero_grad()
        real_pred = discriminator(imgs)

        noise = torch.randn(args.batch_size, args.nz, 1, 1).to(args.device)
        fake_imgs = generator(noise)
        fake_pred = discriminator(fake_imgs.detach())
        loss_d = criterion((fake_pred, real_pred), mode='discriminator_loss')
        losses_d.append(loss_d.item())

        loss_d.backward()
        optimizer_d.step()

        # 2. Update Generator
        optimizer_g.zero_grad()
        fake_pred = discriminator(fake_imgs)
        loss_g = criterion((fake_pred, real_pred), mode='generator_loss')
        losses_g.append(loss_g.item())

        loss_g.backward()
        optimizer_g.step()

        wandb.log({'generator loss': loss_g.item(),
                   'discriminator loss': loss_d.item()})
        
        pbar.set_description(f"G: {round(loss_g.item(), 3)} | D: {round(loss_d.item(), 3)}")
        pbar.refresh()
        pbar.update()

        if (itr + 1) % int(args.total_iteration / 10) == 0:
            # evaluation
            generator_score = inception_score(generator, inception_model, args)

            print('-' * 20)
            print(f'Iteration: {itr + 1}')
            print(f'Discriminator Loss: {loss_d.item()}')
            print(f'Generator Loss: {loss_g.item()}')
            print(f'Inception Score: {generator_score}')
            
            # save model
            if generator_score > best_generator_score:
                save_ckpt(generator, discriminator, score, args)
            
            generator.eval()
            with torch.no_grad():
                sample_img = generator(fixed_noise)
            generator.train()
            sample_img = torchvision.utils.make_grid(sample_img.cpu(), normalize=True, nrow=4)
            wandb.log({f'sample{itr + 1}': wandb.Image(sample_img)})

def save_ckpt(generator, discriminator, score, args):
    generator.cpu()
    discriminator.cpu()
    
    ckpt = {
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'score': score,
    }
    
    os.makedirs('checkpoints/', exist_ok=True)
    torch.save(ckpt, f'checkpoints/{args.model_name}.{args.dataset_name}.pt')
    generator.to(args.device)
    discriminator.to(args.device)

if __name__ == '__main__':

    # init logger
    wandb.init(project='wgan', config=args)
    
    args = get_args()
    print(args)
    model_classes = [m for m in dir(models) 
                       if m.lower() == args.model_name.lower()]
    assert model_classes, f'Model name {args.model_name} does not exist'
    components = getattr(models, model_classes[0]).components

    generator = components['generator'](args).train().to(args.device)
    discriminator = components['discriminator'](args).train().to(args.device)
    criterion = components['criterion'](args)
    optimizer_g = components['optimizer'](args, params=generator.parameters())
    optimizer_d = components['optimizer'](args, params=discriminator.parameters())
    
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model.to(args.device)
    inception_model.eval()
    
    train_loader = get_dataloader(args, train=True)

    wandb.watch(generator)
    wandb.watch(discriminator)
    
    train(generator, discriminator, criterion, optimizer_g, optimizer_d,
          inception_model, train_loader, args)
    
    # close logger
    wandb.finish()