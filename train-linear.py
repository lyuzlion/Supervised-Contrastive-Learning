import argparse
import torchvision.datasets as datasets
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torch
from models.resnet import *
from tqdm import tqdm
from utils.util import AverageMeter, adjust_learning_rate, compute_accuracy
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=800,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.08,
                        help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.9,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--projection', type=str, default='Linear', choices=['Linear', 'MLP', 'Transformer'],help='projection head')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'], help='dataset')

    parser.add_argument('--checkpoint', type=str, default='',
                        help='path to pre-trained model')

    opt = parser.parse_args()

    opt.data_folder = '../data/' + opt.dataset + '/'

    opt.model_name = '{}'.format(opt.model)


    if opt.dataset == 'cifar10':
        opt.num_class = 10
    elif opt.dataset == 'cifar100':
        opt.num_class = 100
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt


def train(train_loader, model, classifier, criterion, optimizer):
    """one epoch training"""
    model.eval()
    classifier.train()

    losses = AverageMeter()
    top1 = AverageMeter()

    for idx, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()

        # compute loss
        with torch.no_grad():
            features = model.encoder(images) # 这里丢掉模型的projection head
        output = classifier(features.detach())
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), labels.shape[0])

        # print(labels.shape) # torch.Size([256])
        # print(output.shape) # torch.Size([256, 10])

        acc1, acc5 = compute_accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], labels.shape[0])

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg, top1.avg


def validate(valid_loader, model, classifier, criterion):
    """validation"""
    model.eval()
    classifier.eval()

    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        for idx, (images, labels) in enumerate(valid_loader):
            images = images.cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(model.encoder(images))
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = compute_accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)


    print('Validation Accuracy: {top1.avg:.3f}%'.format(top1=top1))
    sys.stdout.flush()
    return losses.avg, top1.avg


def main():
    best_acc = 0
    opt = parse_option()    

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder, transform=train_transform, download=True)
        val_dataset = datasets.CIFAR10(root=opt.data_folder, train=False, transform=valid_transform)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder, transform=train_transform, download=True)
        val_dataset = datasets.CIFAR100(root=opt.data_folder, train=False, transform=valid_transform)
    else:
        raise ValueError(opt.dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=True)


    # build model and criterion
    model = SupConResNet(name=opt.model).cuda()
    model.load_state_dict(torch.load(opt.checkpoint, weights_only=False)['model'])

    criterion = torch.nn.CrossEntropyLoss().cuda()

    if opt.projection == 'Linear':
        classifier = LinearClassifier(name=opt.model, num_classes=opt.num_class).cuda()
    elif opt.projection == 'MLP':
        classifier = MLPClassifier(name=opt.model, num_classes=opt.num_class).cuda()
    else:
        classifier = TransformerClassifier(name=opt.model, num_classes=opt.num_class).cuda()
    
    optimizer = optim.SGD(classifier.parameters(), lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)

    for epoch in tqdm(range(1, opt.epochs + 1)):
        loss, acc = train(train_loader, model, classifier, criterion, optimizer)
        # print('Train epoch {}, accuracy:{:.2f}'.format(epoch, acc))

        loss, val_acc = validate(valid_loader, model, classifier, criterion)
        if val_acc > best_acc:
            best_acc = val_acc
            print('saving model...')
            sys.stdout.flush()
            torch.save(classifier.state_dict(), 'projection-checkpoints/' + opt.projection + '/{}_{}.pth'.format(opt.dataset, opt.model))

        if epoch % 10 == 0:
            adjust_learning_rate(opt, optimizer)

    print('best accuracy: {:.2f}'.format(best_acc))


if __name__ == '__main__':
    main()
