from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import os.path as osp

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from vehiclereid.utils.loggers import Logger, RankLogger
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image

from args import argument_parser, dataset_kwargs, optimizer_kwargs, lr_scheduler_kwargs
from vehiclereid.data_manager import ImageDataManager
from vehiclereid.utils.avgmeter import AverageMeter
from vehiclereid.utils.iotools import check_isfile
from vehiclereid.losses import DeepSupervision, VAE
from vehiclereid.utils.torchtools import count_num_param, accuracy, \
    load_pretrained_weights, save_checkpoint, resume_from_checkpoint
from vehiclereid.utils.generaltools import set_random_seed
from vehiclereid.optimizers import init_optimizer
from vehiclereid.lr_schedulers import init_lr_scheduler

# global variables
parser = argument_parser()
args = parser.parse_args()

def main():
    global args

    set_random_seed(args.seed)

    if not args.use_avai_gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False
    log_name = 'log_test.txt' if args.evaluate else 'log_train.txt'
    sys.stdout = Logger(osp.join(args.save_dir, log_name))
    print('==========\nArgs:{}\n=========='.format(args))

    if use_gpu:
        print('Currently using GPU {}'.format(args.gpu_devices))
        cudnn.benchmark = True

    # Data Loader (Input Pipeline)
    dm = ImageDataManager(use_gpu, **dataset_kwargs(args))
    trainloader, testloader_dict = dm.return_dataloaders()

    model = VAE(x_dim=args.height*args.width, h_dim1= 512, h_dim2=256, z_dim=2)

    if args.load_weights and check_isfile(args.load_weights):
        load_pretrained_weights(model, args.load_weights)

    optimizer = init_optimizer(model, **optimizer_kwargs(args))
    scheduler = init_lr_scheduler(optimizer, **lr_scheduler_kwargs(args))

    if args.resume and check_isfile(args.resume):
        args.start_epoch = resume_from_checkpoint(args.resume, model, optimizer=optimizer)

    time_start = time.time()
    ranklogger = RankLogger(args.source_names, args.target_names)
    print('=> Start training')
    '''
    if args.fixbase_epoch > 0:
        print('Train {} for {} epochs while keeping other layers frozen'.format(args.open_layers, args.fixbase_epoch))
        initial_optim_state = optimizer.state_dict()

        for epoch in range(args.fixbase_epoch):
            train(epoch, model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu, fixbase=True)

        print('Done. All layers are open to train for {} epochs'.format(args.max_epoch))
        optimizer.load_state_dict(initial_optim_state)
    '''
    for epoch in range(args.start_epoch, args.max_epoch):
        train(epoch, model, optimizer, trainloader, use_gpu)

        scheduler.step()

        if (epoch + 1) > args.start_eval and args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0 or (
                epoch + 1) == args.max_epoch:
            print('=> Test')

            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'optimizer': optimizer.state_dict(),
            }, args.save_dir)

    elapsed = round(time.time() - time_start)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print('Elapsed {}'.format(elapsed))
    ranklogger.show_summary()

def train(epoch, model, optimizer, trainloader, use_gpu):
    accs = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()

    end = time.time()
    for batch_idx, (imgs, pids, _, _) in enumerate(trainloader):
        data_time.update(time.time() - end)

        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()

        recon_batch, mu, log_var = model(imgs)
        # VAE loss function
        BCE = F.binary_cross_entropy(recon_batch, imgs.view(-1, 784), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = BCE + KLD

        if isinstance(recon_batch, (tuple, list)):
            loss = DeepSupervision(loss, recon_batch, pids)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        accs.update(accuracy(recon_batch, pids)[0])

        if (batch_idx + 1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.4f} ({data_time.avg:.4f})\t'
                  'Vae {vae.val:.4f} ({vae.avg:.4f})\t'
                  'Acc {acc.val:.2f} ({acc.avg:.2f})\t'.format(
                epoch + 1, batch_idx + 1, len(trainloader),
                batch_time=batch_time,
                data_time=data_time,
                vae=loss,
                acc=accs
            ))

        end = time.time()

def test(model, queryloader, use_gpu):
    batch_time = AverageMeter()

    model.eval()
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (imgs, pids, _, _) in enumerate(queryloader):
            if use_gpu:
                imgs = imgs.cuda()
            recon, mu, log_var = model(imgs)
            # sum up batch loss
            BCE = F.binary_cross_entropy(recon, imgs.view(-1, 784), reduction='sum')
            KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = BCE + KLD
            test_loss += loss

    test_loss /= len(queryloader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == '__main__':
    main()