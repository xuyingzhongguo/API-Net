import torch
import shutil
import os
import torchvision
from torch.utils.data.dataloader import default_collate
import sys


def save_checkpoint(save_path, state, is_best, saved_file):
    # torch.save(state, saved_file)
    if is_best:
        torch.save(state, saved_file)
        # shutil.copyfile(saved_file, os.path.join(save_path, 'model_best.pth.tar'))


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

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



def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def accuracy_test(scores, targets):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = torch.topk(scores, 1)

    tn = tp = fn = fp = 0
    if ind == 0 and targets.view(-1) == 0:
        tn = tn + 1
    elif ind != 0 and targets != 0:
        tp = tp + 1
    elif ind != 0 and targets.view(-1) == 0:
        fp = fp + 1
    elif ind == 0 and targets != 0:
        fn = fn + 1
    else:
        sys.exit('wrong acc calculation')

    correct = ind.eq(targets.view(-1))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size), tn, tp, fn, fp


def accuracy_test_open_set(scores, targets):
    tn = tp = fn = fp = 0
    batch_size = targets.size(0)
    _, ind = torch.topk(scores, 1)
    if ind == 0 and targets.view(-1) == 0:
        correct = 1
        tn = tn + 1
    elif ind != 0 and targets != 0:
        correct = 1
        tp = tp + 1
    elif ind != 0 and targets.view(-1) == 0:
        correct = 0
        fp = fp + 1
    elif ind == 0 and targets != 0:
        correct = 0
        fn = fn + 1
    else:
        sys.exit('wrong acc calculation')

    return correct * (100.0 / batch_size), tn, tp, fn, fp


def init_weights_zero(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.zeros_(m.weight)


def init_weights_xavier_normal(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight)


def init_weights_xavier_uniform(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)


def init_weights_kaiming_normal(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight)


def init_weights_kaiming_uniform(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.kaiming_uniform_(m.weight)


def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)




