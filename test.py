import argparse
import os
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
from models import API_Net
from datasets import RandomDataset_test
from utils import accuracy_test, AverageMeter

np.set_printoptions(suppress=True,
                    formatter={'float_kind': '{:0.4f}'.format})

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--exp_name', default=None, type=str,
                    help='name of experiment')
parser.add_argument('--data', metavar='DIR', default='',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--evaluate-freq', default=10, type=int,
                    help='the evaluation frequence')
parser.add_argument('--resume', default='./checkpoint.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--model_load_path', default='./save', type=str,
                    help='model path you want to test')
parser.add_argument('--output_path', default='./output/test.txt', type=str,
                    help='test result output path')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--n_classes', default=2, type=int,
                    help='the number of classes')
parser.add_argument('--n_classes_total', default=5, type=int,
                    help='the overall number of classes')
parser.add_argument('--n_samples', default=8, type=int,
                    help='the number of samples per class')
parser.add_argument('--test_list', default='data_list/trycode.txt', type=str,
                    help='test list')
parser.add_argument('--model_name', default='res101', type=str)
parser.add_argument('--dist_type', default='euclidean', type=str)
parser.add_argument('--image_loader', default='default_loader', type=str)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def main(args):
    model_path = args.model_load_path
    result_write_path = args.output_path
    test_list = args.test_list
    batch_size = args.batch_size
    n_classes_total = args.n_classes_total
    model_name = args.model_name
    dist_type = args.dist_type
    image_loader = args.image_loader

    # create model
    model = API_Net(num_classes=n_classes_total, model_name=model_name)
    model = model.to(device)
    model.conv = nn.DataParallel(model.conv)

    if os.path.isfile(args.resume):
        print('loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        print('loaded checkpoint {}(epoch {})'.format(args.resume, checkpoint['epoch']))
    else:
        print('no checkpoint found at {}'.format(args.resume))

    transform_3 = transforms.Compose([
        transforms.Resize([512, 512]),
        transforms.RandomCrop([448, 448]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )])

    transform_9 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([512, 512]),
        transforms.RandomCrop([448, 448]),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406, 0.485, 0.456, 0.406, 0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225, 0.229, 0.224, 0.225, 0.229, 0.224, 0.225)
        )])

    if image_loader == 'nine_channels':
        transform_picked = transform_9
    else:
        transform_picked = transform_3

    test_dataset = RandomDataset_test(val_list=test_list,
                                      loader=image_loader,
                                      transform=transform_picked
                                      )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    _, tns, tps, fns, fps = test(test_loader, model, batch_size, dist_type, image_loader, result_write_path)
    print(f'true negative: {tns}, true positive: {tps}, false negative: {fns}, false positive: {fps}')
    print('-----------------------------------------------------------------------------------------')


def test(test_loader, model, bs, dist_type, image_loader, output_file='output-predictions.txt'):
    batch_time = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    # print(model)

    result_file = open(output_file, 'a')
    tns = tps = fns = fps = 0

    with torch.no_grad():
        for i, (input, target, image_name) in enumerate(test_loader):
            input_val = input.to(device)
            target_val = target.to(device)
            result_file.write(image_name[0] + '\n')

            # compute output
            logits = model(input_val, targets=None, flag='test', dist_type=dist_type, loader=image_loader)
            logits_str = str(logits.view(-1).cpu())
            result_file.write(logits_str + '\n')

            class_label, score_value = logits.max(0)
            labels_scores_predictions = '{0} {1} {2}'.format(class_label, target[0][0].numpy(), score_value)
            result_file.write(labels_scores_predictions + '\n')

            prec1, tn, tp, fn, fp = accuracy_test(logits, target_val)
            tns = tns + tn
            tps = tps + tp
            fns = fns + fn
            fps = fps + fp

            top1.update(prec1, logits.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Validation results: \t Time: {time}\nTest: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(test_loader), batch_time=batch_time,
                    top1=top1, time=time.asctime(time.localtime(time.time()))))
        print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    # result_file.close()
    return top1.avg, tns, tps, fns, fps


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
