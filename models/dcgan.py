# DCGAN
import math
import torch
import torch.nn as nn



def weights_init(m):
    classname = m.__class__.__name__
    if classname == 'ConvTranspose2d' or classname == 'Conv2d':
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname == 'BatchNorm2d':
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class ConvBlock(nn.Module):
    def __init__(self, conv_kwargs, activation='leaky_relu', normalization='batch_normalization'):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(**conv_kwargs)
        
        if normalization == 'batch_normalization': self.norm = nn.BatchNorm2d(conv_kwargs['out_channels'])
        elif normalization == 'instance_normalization': self.norm = nn.InstanceNorm2d(conv_kwargs['out_channels'])
        elif normalization is None: self.norm = None
        
        if activation == 'leaky_relu': self.actv = nn.LeakyReLU(0.2)
        elif activation == 'sigmoid': self.actv = nn.Sigmoid()
    
    def forward(self, x):
        out = self.conv(x)
        if self.norm is not None:
            out = self.norm(out)
        out = self.actv(out)
        return out

    
class DCGANDiscriminator(nn.Module):
    def __init__(self, args):
        super(DCGANDiscriminator, self).__init__()
        depth = int(math.log2(args.resolution)) -1

        net = [ConvBlock(
            {
                'in_channels': args.nc,
                'out_channels': args.ngf,
                'kernel_size': 4,
                'stride': 2,
                'padding': 1
            },
            normalization=None
        )]
        
        mult = 1
        for _ in range(depth - 2):
            net.append(ConvBlock(
                {
                    'in_channels': args.ngf * mult,
                    'out_channels': args.ngf * mult * 2,
                    'kernel_size': 4,
                    'stride': 2,
                    'padding': 1
                }
            ))
            mult *= 2
        net.append(ConvBlock(
            {
                'in_channels': args.ngf * mult,
                'out_channels': 1,
                'kernel_size': 4,
                'stride': 1,
                'padding': 0
            },
            activation='sigmoid',
            normalization=None
        ))
        self.net = nn.Sequential(*net)
        self.apply(weights_init)
            
    def forward(self, x):
        return self.net(x)
        

class ConvTransposeBlock(nn.Module):
    
    def __init__(self, deconv_kwargs, activation='relu', normalization='batch_normalization'):
        super(ConvTransposeBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(**deconv_kwargs)
        
        if normalization == 'batch_normalization': self.norm = nn.BatchNorm2d(deconv_kwargs['out_channels'])
        elif normalization == 'instance_normalization': self.norm = nn.InstanceNorm2d(deconv_kwargs['out_channels'])
        elif normalization is None: self.norm = nn.Sequential()
        
        if activation == 'relu': self.actv = nn.ReLU()
        elif activation == 'tanh': self.actv = nn.Tanh()
    
    def forward(self, x):
        out = self.deconv(x)
        out = self.norm(out)
        out = self.actv(out)
        return out
    
    

class DCGANGenerator(nn.Module):
    def __init__(self, args):
        super(DCGANGenerator, self).__init__()
        depth = int(math.log2(args.resolution)) -1
        
        mult = 2 ** (depth - 2)
        net = [ConvTransposeBlock(
            {
                'in_channels': args.nz,
                'out_channels': args.ngf * mult,
                'kernel_size': 4,
                'stride': 1,
                'padding': 0
            },
        )]
        for _ in range(depth - 2):
            mult = int(mult * 0.5)
            net.append(ConvTransposeBlock(
                {
                    'in_channels': args.ngf * mult * 2,
                    'out_channels': args.ngf * mult,
                    'kernel_size': 4,
                    'stride': 2,
                    'padding': 1
                },
            ))
        net.append(ConvTransposeBlock(
            {
                'in_channels': args.ngf * mult,
                'out_channels': args.nc,
                'kernel_size': 4,
                'stride': 2,
                'padding': 1
            },
            activation='tanh',
            normalization=None
        ))
        
        self.net = nn.Sequential(*net)
        self.apply(weights_init)
        
    def forward(self, x):
        return self.net(x)


class DCGANLoss(nn.Module):
    def __init__(self, args):
        super(DCGANLoss, self).__init__()
        self.device = args.device
        self.bce = nn.BCELoss()
        
    def forward(self, x, mode='discriminator_loss'):
        if mode == 'discriminator_loss':
            fake_pred, real_pred = x
            real_loss = self.bce(real_pred, torch.tensor(1.0).expand_as(real_pred).to(self.device))
            fake_loss = self.bce(fake_pred, torch.tensor(0.0).expand_as(fake_pred).to(self.device))
            loss = (real_loss + fake_loss) * 0.5
            
        elif mode == 'generator_loss':
            fake_pred, _ = x
            loss = self.bce(fake_pred, torch.tensor(1.0).expand_as(fake_pred).to(self.device))
        
        return loss

components = {
    'generator': DCGANGenerator,
    'discriminator': DCGANDiscriminator,
    'criterion': DCGANLoss
}