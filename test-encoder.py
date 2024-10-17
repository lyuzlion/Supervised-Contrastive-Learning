import os
import argparse
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
from models.resnet import SupConResNet
from utils.util import *
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

def parse_option():
    parser = argparse.ArgumentParser('argument for testing encoder')

    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet50'], help='model')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')
    parser.add_argument('--checkpoint', type=str, default='', help='path to pre-trained model')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07, help='temperature for loss function')

    opt = parser.parse_args()

    opt.data_folder = '../data/' + opt.dataset + '/'
    if opt.dataset == 'cifar10':
        opt.num_class = 10
    elif opt.dataset == 'cifar100':
        opt.num_class = 100
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    return opt


def set_loader(opt):

    test_transform = transforms.Compose([
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
                        transform=TwoCropTransform(test_transform),
                        train=False,
                        download=True)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                        transform=TwoCropTransform(test_transform),
                        train=False,
                        download=True)
    else:
        raise ValueError(opt.dataset)

    test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

    return test_loader

def test(test_loader, model):
    '''
    test function
    '''
    features = []
    labels = []
    model.eval()
    for idx, (image, label) in tqdm(enumerate(test_loader)):
        # print(images[0].shape) # torch.Size([256, 3, 32, 32])
        with torch.no_grad():
            image = torch.cat([image[0], image[1]], dim=0)
            image = image.cuda()
            label = label.cuda()
            bsz = label.shape[0]

            feature = model.encoder(image)

            features.append(feature.cpu().detach().numpy()[0].tolist())
            labels.append(label.cpu().detach().numpy()[0].tolist())
            # features.append(f1.cpu().detach().numpy())
            # labels.append(label.cpu().detach().numpy())
    features = np.array(features)   
    labels = np.array(labels)
    return features, labels

def plot_embedding(data, label, title, opt):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]), color=plt.cm.Set1(label[i] / opt.num_class), fontdict={'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()
    return

def main():
    opt = parse_option()

    test_loader = set_loader(opt)
    model = SupConResNet(name=opt.model).cuda()
    model.load_state_dict(torch.load(opt.checkpoint, weights_only=False)['model'])

    features, labels = test(test_loader, model)
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(features)
    print('result.shape',result.shape)
    plot_embedding(result, labels, 't-SNE embedding of the CIFAR100', opt.num_class)
    


if __name__ == '__main__':
    main()