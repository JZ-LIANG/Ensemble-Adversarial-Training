import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
import torchvision
import numpy as np
import os
import argparse
import pathlib
from tensorboardX import SummaryWriter

import sys
from ens_adv_train import ens_adv_train, validate

# import models
from models.cifar10.resnet import ResNet34, ResNet101, ResNet18, ResNet50
from models.cifar10.mobilenetv2_2 import MobileNetV2
from models.cifar10.inception import GoogLeNet


parser = argparse.ArgumentParser(description='Adv Training')

parser.add_argument('--dataset', default='cifar10', type=str,
                    help='select the training dataset')

parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--eps', default = 2, type=float, metavar='M',
                    help='option1: random epsilon distribution')

parser.add_argument('--attacker', default='stepll', type=str,
                    help='option2: attacker for generating adv input')

parser.add_argument('--loss_schema', default='averaged', type=str,
                    help='option3: loss schema')


# reproducible 
torch.manual_seed(66)
np.random.seed(66)


######################################### modify accordingly ##################################################
# adv models: the static model used to generate adv input images
# fixed to memory for all the trainings to speed up.
adv_resnet18 = ResNet18()
adv_resnet50 = ResNet50()
adv_mobilenet_125 = MobileNetV2(width_mult=1.25)
adv_googlenet = GoogLeNet()


adv_models = [adv_resnet18, adv_resnet50, adv_mobilenet_125, adv_googlenet]
adv_model_names = ['resnet18', 'resnet50', 'mobilenet_125', 'googlenet']

# models: models for be adv training
# loaded only on its training to save memory.
model_classes = [ ResNet34, ResNet101, MobileNetV2, MobileNetV2]
model_names = [ 'resnet34', 'resnet101', 'mobilenet_1', 'mobilenet_075']
params = {
    'mobilenet_1': 1.0,
    'mobilenet_075': 0.75,
}


# path
trial_name = 'adv_models:'
for adv_model_name in adv_model_names:
    trial_name = trial_name + '-' + adv_model_name
# path to pre-trained models checkpoints
adv_checkpoint_path = 'checkpoints/cifar10/'
output_path = 'checkpoints/adv_train/cifar10/' + trial_name +'/'
tensorboard_path = 'tensorboard/cifar10/adv_train/' + trial_name +'/'
######################################### modify accordingly ##################################################



if not os.path.isdir(output_path):
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
if not os.path.isdir(tensorboard_path):
    pathlib.Path(tensorboard_path).mkdir(parents=True, exist_ok=True)

def main(model_class, model_name, model_path, adv_models, writer, args):
    dataset = args.dataset
    epochs = args.epochs

    best_acc = 0

    # prepare data loader 
    trainloader, testloader = get_data_loader(dataset)

    # create model
    if model_name in params.keys():
        model = model_class(params[model_name])
    else:
        model = model_class()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        model = model.cuda()

    # optimizer 
    criterion = nn.CrossEntropyLoss(reduction = 'mean')
    # paper use RMSProp but author's github use adam, here we follow the author's github
    optimizer = optim.Adam(model.parameters(), lr= 0.001, weight_decay=5e-4)

    # training
    for epoch in range(epochs): 
        ens_adv_train(trainloader, criterion, optimizer, model, adv_models, writer, epoch, args)
        acc = validate(testloader, model, criterion, writer, epoch)

        if acc > best_acc :
            best_acc = acc
            save_checkpoint(model, model_path, optimizer, best_acc, epoch)



# save model
def save_checkpoint(model, model_path, optimizer, best_acc, epoch):
    state = {
        'state_dict': model.state_dict(),
        'acc': best_acc,
        'epoch': epoch,
        'optimizer' : optimizer.state_dict(),

    }
    torch.save(state, model_path)



def get_data_loader(dataset):
    if dataset == 'cifar10':
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # mean subtract 
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        trainset = torchvision.datasets.CIFAR10(root='/home/deliangj/data/', train=True, download=False, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

        testset = torchvision.datasets.CIFAR10(root='/home/deliangj/data/', train=False, download=False, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=4)

        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    elif dataset == "cinic10":
        cinic_directory = '/home/deliangj/data/cinic10'
        cinic_mean = [0, 0, 0]
        cinic_std = [0, 0, 0]

        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # mean subtract 
        transforms.Normalize(mean=cinic_mean,std=cinic_std)
        ,])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cinic_mean,std=cinic_std),
        ])

        trainset = torchvision.datasets.ImageFolder(cinic_directory + '/train', transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

        testset = torchvision.datasets.ImageFolder(cinic_directory + '/test',  transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=4)
    else:
        print('not such dataset !')
        return 

    return trainloader, testloader


if __name__ == '__main__':

    # training parameters
    args = parser.parse_args()

    # checkpoint paths
    model_save_paths = [output_path + model_name + '.pth.tar' for model_name in model_names]
    adv_model_paths = [adv_checkpoint_path + adv_model_name + '.pth.tar' for adv_model_name in adv_model_names]

    # load adv models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        for i in range(len(adv_models)):
            adv_models[i] = torch.nn.DataParallel(adv_models[i])
            adv_models[i] = adv_models[i].cuda()
            # pre-trained static models !
            adv_models[i] = adv_models[i].eval()
    else:
        print('gpu not avaible please check !')
        sys.exit()

    # adv pre-trained static models
    for i in range(len(adv_model_paths)):
        checkpoint = torch.load(adv_model_paths[i])
        if 'state_dict' in checkpoint.keys():
            state = 'state_dict'
        elif 'net' in checkpoint.keys():
            state = 'net'
        adv_models[i].load_state_dict(checkpoint[state])

    # starting training each model
    for i in range(len(model_classes)):
        print('adv training model: ' + model_names[i])
        writer = SummaryWriter(tensorboard_path + model_names[i])
        main(model_classes[i], model_names[i], model_save_paths[i], adv_models, writer, args)

