import argparse
import torchvision.datasets as datasets
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torch
from models.resnet import *
from tqdm import tqdm
from utils.util import AverageMeter, compute_accuracy
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'



def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--projection', type=str, default='Linear',choices=['Linear', 'MLP', 'Transformer'], help='projection head')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='dataset')
    parser.add_argument('--checkpoint', type=str, default='', help='path to pre-trained model')

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



def test(test_loader, model, classifier):
    """test"""
    model.eval()
    classifier.eval()

    top1 = AverageMeter()
    top3 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images = images.cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(model.encoder(images))

            # update metric
            acc1, acc3, acc5 = compute_accuracy(output, labels, topk=(1, 3, 5))
            top1.update(acc1[0], bsz)
            top3.update(acc3[0], bsz)
            top5.update(acc5[0], bsz)


    print('Top1 Accuracy: {top1.avg:.3f}%'.format(top1=top1))
    print('Top3 Accuracy: {top3.avg:.3f}%'.format(top3=top3))
    print('Top5 Accuracy: {top5.avg:.3f}%'.format(top5=top5))
    return


def main():
    opt = parse_option()    

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    if opt.dataset == 'cifar10':
        val_dataset = datasets.CIFAR10(root=opt.data_folder, train=False, transform=test_transform)
    elif opt.dataset == 'cifar100':
        val_dataset = datasets.CIFAR100(root=opt.data_folder, train=False, transform=test_transform)
    else:
        raise ValueError(opt.dataset)

    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=True)

    model = SupConResNet(name=opt.model).cuda()
    model.load_state_dict(torch.load(opt.checkpoint, weights_only=False)['model'])

    if opt.projection == 'Linear':
        classifier = LinearClassifier(name=opt.model, num_classes=opt.num_class).cuda()
    elif opt.projection == 'MLP':
        classifier = MLPClassifier(name=opt.model, num_classes=opt.num_class).cuda()
    else:
        classifier = TransformerClassifier(name=opt.model, num_classes=opt.num_class).cuda()

 
    classifier.load_state_dict(torch.load('./projection-checkpoints/' + opt.projection + '/{}_{}.pth'.format(opt.dataset, opt.model), weights_only=False))    
    test(test_loader, model, classifier)
        

if __name__ == '__main__':
    main()
