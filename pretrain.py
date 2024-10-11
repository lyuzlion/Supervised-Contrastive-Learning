import os
import argparse
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
from models.resnet import SupConResNet
from utils.util import *
from loss import SupConLoss


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.9,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'], help='dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    opt = parser.parse_args()


    opt.data_folder = '../data/' + opt.dataset + '/'
    opt.model_path = './checkpoints/{}_models'.format(opt.dataset)
    opt.model_name = '{}'.format(opt.model)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_loader(opt):

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                        transform=TwoCropTransform(train_transform),
                        download=True)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                        transform=TwoCropTransform(train_transform),
                        download=True)
    else:
        raise ValueError(opt.dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)

    return train_loader

def train(train_loader, model, criterion, optimizer, opt):
    """one epoch training"""
    model.train()
    losses = AverageMeter()
    print()
    for idx, (images, labels) in enumerate(train_loader):
        # print(images[0].shape) # torch.Size([256, 3, 32, 32])

        images = torch.cat([images[0], images[1]], dim=0)
        images = images.cuda()
        labels = labels.cuda()
        bsz = labels.shape[0]

        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = criterion(features, labels)

        # update metric
        losses.update(loss.item(), bsz)
        if idx % 10 == 0:
            print('Loss: {:.4f}'.format(losses.avg))
            
        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg


def main():
    opt = parse_option()

    train_loader = set_loader(opt)
    model = SupConResNet(name=opt.model).cuda()
    criterion = SupConLoss(temperature=opt.temp).cuda()
    optimizer = optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)
    loss_min = np.inf
    
    model.train()

    for epoch in tqdm(range(1, opt.epochs + 1)):
        loss = train(train_loader, model, criterion, optimizer, opt)

        if loss < loss_min:
            loss_min = loss
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{}.pth'.format(epoch))
            save_model(model, optimizer, opt, epoch, save_file)
        
        if epoch % 1 == 0:
            adjust_learning_rate(opt, optimizer)


if __name__ == '__main__':
    main()