import torch
import numpy as np
import os
import sys
from scipy.stats import truncnorm

from Attacks.Gradient_based.least_likely_class_method import least_likely_class_method
from Attacks.Gradient_based.fast_gradient_method import fast_gradient_method

def ens_adv_train(trainloader, criterion, optimizer, model, adv_models, writer, epoch, args):

    losses_combine = AverageMeter()
    top1_combine = AverageMeter()
    losses_clean = AverageMeter()
    top1_clean = AverageMeter()
    losses_adv = AverageMeter()
    top1_adv = AverageMeter()
   
    # training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for i, (inputs_clean, targets_clean) in enumerate(trainloader):

        # inputs
        inputs_clean, targets_clean = inputs_clean.to(device), targets_clean.to(device)
        # generate adv images
        # in paper, clean and adv images are half to half in each batch, 
        # but in author's github, clean and adv image are using entire batch and then the loss is averaged from loss of these two batch 
        # when selected == len(adv_models), select the currunt state of the model
        # otherwise choose the corresponding static model 
        selected = np.random.randint(len(adv_models) + 1)
        if selected == len(adv_models):
            adv_generating_model = model
        else:
            adv_generating_model = adv_models[selected]
        # the model generate adv should be in eval() model
        adv_generating_model.eval()


        # setting epsilon, normal it to range: [0, 1]
        if 0 < args.eps and  args.eps < 1:
            # fixed epsilon
            eps = args.eps 
        elif args.eps == 1 :
            # paper: <adversarial machine learning at scale>, arXiv:1611.01236
            # favor small epsilon
            # a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
            eps = truncnorm.rvs(a = 0, b = 2, loc = 0, scale = 8) / 255.0
        elif args.eps == 2 :
            # uniform distribution, even the possibility for large and small eps, range [2/255, 16/255]
            eps = np.random.randint(low = 2, high =17) / 255.0
 

        # generate adv images
        if args.attacker == 'stepll':
            # Step.L.L adv 
            inputs_adv = least_likely_class_method(adv_generating_model, inputs_clean, eps, clip_min= 0, clip_max= 1)
        elif args.attacker =='fgsm':
            # Step.L.L adv 
            inputs_adv = fast_gradient_method(adv_generating_model, inputs_clean, eps, clip_min= 0, clip_max= 1)            


        # training
        ## in case that the adv_generating_model is the training model itself, clean the gradient and swith the model
        model.zero_grad()
        model.train()

        # clean image
        logits_clean = model(inputs_clean)
        loss1 = criterion(logits_clean, targets_clean)


        # adv image 
        logits_adv = model(inputs_adv)
        loss2 = criterion(logits_adv, targets_clean)


        # combine the loss1 and loss2
        if args.loss_schema == 'averaged':
            # loss on multiple outputs
            # https://discuss.pytorch.org/t/a-model-with-multiple-outputs/10440
            loss = 0.5*(loss1 + loss2)
        elif args.loss_schema == 'weighted':
            # paper: <adversarial machine learning at scale>, arXiv:1611.01236
            # favor for clean input
            loss = (1 / 1.3) (loss1 + 0.3* loss2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # print log and tensorboard
        # clean
        acc1, _  = accuracy(logits_clean, targets_clean, topk=(1,5))
        losses_clean.update(loss1.item(), inputs_clean.size(0))
        top1_clean.update(acc1[0], inputs_clean.size(0))

        # adv
        acc2, _ = accuracy(logits_adv, targets_clean, topk=(1, 5))
        losses_adv.update(loss2.item(), inputs_clean.size(0))
        top1_adv.update(acc2[0], inputs_clean.size(0))

        # combine
        acc = 0.5*(acc1[0] + acc2[0])
        losses_combine.update(loss.item(), inputs_clean.size(0))
        top1_combine.update(acc, inputs_clean.size(0))

        # return losses_clean, top1_clean, losses_adv, top1_adv, losses_combine, top1_combine

        # progress_bar(i, len(trainloader), 'Epoch: %d | clean: %.3f | Top1: %.3f | Top5: %.3f '
        # % (epoch, losses.avg, top1.avg, top5.avg))
        
        if i % 20 == 0:
            n_iter = epoch * len(trainloader) + i
            writer.add_scalar('Train/Loss_clean', losses_clean.val, n_iter)
            writer.add_scalar('Train/Loss_adv', losses_adv.val, n_iter)
            writer.add_scalar('Train/Losses_combine', losses_combine.val, n_iter)
            writer.add_scalar('Train/Prec@1_clean', top1_clean.val, n_iter)
            writer.add_scalar('Train/Prec@1_adv', top1_adv.val, n_iter)
            writer.add_scalar('Train/Prec@1_combine', top1_combine.val, n_iter)
            writer.add_scalar('Train/epsilon', eps, n_iter)
            writer.add_scalar('Train/selected', selected, n_iter)



def validate(testloader, model, criterion, writer, epoch):

    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(testloader):
            # inputs
            inputs, targets = inputs.to(device), targets.to(device)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            acc1, _ = accuracy(outputs, targets, topk=(1,5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))

            n_iter_val = epoch * len(testloader) + i
            writer.add_scalar('Test/Loss_clean', losses.val, n_iter_val)
            writer.add_scalar('Test/Prec@1_clean', top1.val, n_iter_val)
          
    return top1.avg            

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def eps_truncnorm():
    # paper: <adversarial machine learning at scale>
    # eps drawn from a truncated normal schema in interval [0, 16] with [mean=0, std=8]: 
    # a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
    a = 0
    b = 2
    loc = 0
    scale = 8

    return truncnorm.rvs(a = 0, b = 2, loc = 0, scale = 8)