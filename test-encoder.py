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
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

def parse_option():
    parser = argparse.ArgumentParser('argument for testing encoder')

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
        if idx > 100:
            break
        with torch.no_grad():
            image = torch.cat([image[0], image[1]], dim=0)
            image = image.cuda()
            label = label.cuda()
            bsz = label.shape[0]

            feature = model(image)
            f1, f2 = torch.split(feature, [bsz, bsz], dim=0)

            features.append(f1.cpu().detach().numpy()[0].tolist())
            labels.append(label.cpu().detach().numpy()[0].tolist())
            # features.append(f1.cpu().detach().numpy())
            # labels.append(label.cpu().detach().numpy())
    features = np.array(features)   
    labels = np.array(labels)
    return features, labels

def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]), color=plt.cm.Set1(label[i] / 10.), fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()
    return

def main():
    opt = parse_option()

    test_loader = set_loader(opt)
    model = SupConResNet(name=opt.model).cuda()
    model.load_state_dict(torch.load('checkpoints/cifar10_models/resnet18/ckpt_epoch_197.pth', weights_only=False)['model'])

    features, labels = test(test_loader, model)
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(features)
    print('result.shape',result.shape)
    plot_embedding(result, labels, 't-SNE embedding of the digits')
    


if __name__ == '__main__':
    main()