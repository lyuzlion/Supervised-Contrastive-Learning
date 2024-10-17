# Supervised-Contrastive-Learning
An implementation of Supervised Contrastive Learning via pytorch.

pretrain resnet(18) on cifar-10:

`python pretrain.py`


train the projection head:

`python train-linear.py --checkpoint ./checkpoints/cifar10_models/resnet18/ckpt_epoch_197.pth`

`python test.py --checkpoint ./checkpoints/cifar10_models/resnet18/ckpt_epoch_197.pth --projection Transformer`

`nohup python pretrain.py --dataset cifar100 &`


`python test.py --checkpoint ./checkpoints/cifar100_models/resnet18/ckpt_epoch_399.pth --dataset cifar100`

`python test-encoder.py --dataset cifar100 --model resnet18 --checkpoint ./checkpoints/cifar100_models/resnet18/ckpt_epoch_399.pth`

`python test-encoder.py --dataset cifar100 --model resnet50 --checkpoint ./checkpoints/cifar100_models/resnet50/ckpt_epoch_396.pth`

`python test.py --checkpoint ./checkpoints/cifar100_models/resnet50/ckpt_epoch_396.pth --dataset cifar100 --model resnet50`

`python test-encoder.py --model resnet50 --checkpoint ./checkpoints/cifar10_models/resnet50/ckpt_epoch_397.pth`