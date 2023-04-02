import torch
import time
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import parser
import argparse
import torch.backends.cudnn as cudnn
import torchvision
from functools import reduce
from visdom_logger import VisdomLogger
from src.utils import warp_tqdm, save_checkpoint, load_cfg_from_cfg_file, merge_cfg_from_list, Logger, get_log_file
from src.utils import warp_tqdm, get_metric, AverageMeter,save_checkpoint
from src.utils_mstar import get_training_dataloader,get_val_dataloader,get_test_dataloader,compute_mean_std,get_network
from src.mstar import MstarTrain,MstarTest
from src.models.ingredient import get_model
from src.optim import get_optimizer, get_scheduler
mstar={}
mstar['train_mean']=[0.2312,0.2689,0.2351]
mstar['train_std']=[0.0669,0.0451,0.0643]
mstar['val_mean']=[0.2221,0.2614,0.2286]
mstar['val_std']=[0.0657,0.0445,0.0635]
class Trainer:
    def __init__(self,device,args):
        self.device=device
        self.args=args
        self.train_mean,self.train_std=mstar['train_mean'],mstar['train_std']
        self.train_loader=get_training_dataloader(mean=self.train_mean,std=self.train_std,batch_size=args.batch_size,
                                                  num_workers=args.num_workers,shuffle=True)
        self.val_mean,self.val_std=mstar['val_mean'],mstar['val_std']
        self.val_loader=get_val_dataloader(mean=self.val_mean,std=self.val_std,batch_size=args.batch_size,
                                                  num_workers=args.num_workers,shuffle=True)
        self.num_classes=args.num_classes
    def cross_entropy(self, logits, one_hot_targets, reduction='batchmean'):
        logsoftmax_fn = nn.LogSoftmax(dim=1)
        logsoftmax = logsoftmax_fn(logits)
        return - (one_hot_targets * logsoftmax).sum(1).mean()
    def do_epoch(self, epoch, scheduler, print_freq, disable_tqdm, callback, model,
                 alpha, optimizer):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        # switch to train mode
        model.train()
        steps_per_epoch = len(self.train_loader)
        end = time.time()
        tqdm_train_loader = warp_tqdm(self.train_loader, disable_tqdm)
        for i,(input,target) in enumerate(tqdm_train_loader):

            input, target = input.to(self.device), target.to(self.device, non_blocking=True)

            smoothed_targets = self.smooth_one_hot(target)
            assert (smoothed_targets.argmax(1) == target).float().mean() == 1.0
            # Forward pass
            if alpha > 0:  # Mixup augmentation
                # generate mixed sample and targets
                lam = np.random.beta(alpha, alpha)
                rand_index = torch.randperm(input.size()[0]).cuda()
                target_a = smoothed_targets
                target_b = smoothed_targets[rand_index]
                mixed_input = lam * input + (1 - lam) * input[rand_index]

                output = model(mixed_input)
                loss = self.cross_entropy(output, target_a) * lam + self.cross_entropy(output, target_b) * (1. - lam)
            else:
                output = model(input)
                loss = self.cross_entropy(output, smoothed_targets)
                

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prec1 = (output.argmax(1) == target).float().mean()
            top1.update(prec1.item(), input.size(0))
            if not disable_tqdm:
                tqdm_train_loader.set_description('Acc {:.2f}'.format(top1.avg))

            # Measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                       epoch, i, len(self.train_loader), batch_time=batch_time,
                       loss=losses, top1=top1))
                if callback is not None:
                    callback.scalar('train_loss', i / steps_per_epoch + epoch, losses.avg, title='Train loss')
                    callback.scalar('@1', i / steps_per_epoch + epoch, top1.avg, title='Train Accuracy')
        scheduler.step()
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
        if callback is not None:
            callback.scalar('lr', epoch, current_lr, title='Learning rate')

    def smooth_one_hot(self, targets):
        assert 0 <= self.args.label_smoothing < 1
        with torch.no_grad():
            new_targets = torch.empty(size=(targets.size(0), self.num_classes), device=self.device)
            new_targets.fill_(self.args.label_smoothing / (self.num_classes-1))
            new_targets.scatter_(1, targets.unsqueeze(1), 1. - self.args.label_smoothing)
        return new_targets

    def meta_val(self, model, disable_tqdm, callback, epoch):
        top1 = AverageMeter()
        model.eval()

        with torch.no_grad():
            tqdm_test_loader = warp_tqdm(self.val_loader, disable_tqdm)
            for i, (inputs, target, _) in enumerate(tqdm_test_loader):
                inputs, target = inputs.to(self.device), target.to(self.device, non_blocking=True)
                #output = model(inputs, feature=True)[0].cuda(0)
                output = model(inputs, feature=True)[0]
                train_out = output[:self.args.meta_val_way * self.args.meta_val_shot]
                train_label = target[:self.args.meta_val_way * self.args.meta_val_shot]
                test_out = output[self.args.meta_val_way * self.args.meta_val_shot:]
                test_label = target[self.args.meta_val_way * self.args.meta_val_shot:]
                train_out = train_out.reshape(self.args.meta_val_way, self.args.meta_val_shot, -1).mean(1)
                train_label = train_label[::self.args.meta_val_shot]
                prediction = self.metric_prediction(train_out, test_out, train_label)
                acc = (prediction == test_label).float().mean()
                top1.update(acc.item())
                if not disable_tqdm:
                    tqdm_test_loader.set_description('Acc {:.2f}'.format(top1.avg * 100))

        if callback is not None:
            callback.scalar('val_acc', epoch + 1, top1.avg, title='Val acc')
        return top1.avg

    def metric_prediction(self, support, query, train_label):
        support = support.view(support.shape[0], -1)
        query = query.view(query.shape[0], -1)
        distance = get_metric(self.args.meta_val_metric)(support, query)
        predict = torch.argmin(distance, dim=1)
        predict = torch.take(train_label, predict)
        return predict
    def getValAcc(self,model):
        acc=0
        batch_acc=[]
        for i,(inputs,targets,_) in enumerate(self.val_loader):
            inputs, target = inputs.to(self.device), target.to(self.device, non_blocking=True)
            output = model(inputs, feature=True)[0].cuda(0)
            output = torch.max(torch.softmax(output,dim=-1),dim=-1)[1]
            batch_acc.append(torch.sum(target.reshape(-1)==output.reshape(-1)).div(self.args.batch_size))
        acc=reduce(lambda x,y:x+y,batch_acc)
        return acc

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Main')
    parser.add_argument('--base_config', default='config/dirichlet/base_config/resnet18/mini/base_config.yaml',type=str, required=True, help='Base config file')
    parser.add_argument('--method_config', default='config/dirichlet/methods_config/alpha_tim.yaml',type=str, required=True, help='Method config file')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.base_config is not None
    cfg = load_cfg_from_cfg_file(args.base_config)
    cfg.update(load_cfg_from_cfg_file(args.method_config))
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    return cfg
def main():
    args = parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    callback = None if args.visdom_port is None else VisdomLogger(port=args.visdom_port)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    torch.cuda.set_device(0)
    # init logger
    log_file = get_log_file(log_path=args.log_path, dataset=args.dataset,
                            backbone=args.arch, method=args.method)
    logger = Logger(__name__, log_file)

    # create model
    logger.info("=> Creating model '{}'".format(args.arch))
    model = torch.nn.DataParallel(get_network(args).cuda())
    logger.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    optimizer = get_optimizer(args=args, model=model)
    
    start_epoch = 0
    best_prec1 = -1

    cudnn.benchmark = True
     # If this line is reached, then training the model
    trainer = Trainer(device=device, args=args)
    scheduler = get_scheduler(optimizer=optimizer,
                              num_batches=len(trainer.train_loader),
                              epochs=args.epochs,
                              args=args)
    tqdm_loop = warp_tqdm(list(range(start_epoch, args.epochs)),
                          disable_tqdm=False)
    for epoch in tqdm_loop:
        # Do one epoch
        trainer.do_epoch(model=model, optimizer=optimizer, epoch=args.epochs,
                         scheduler=scheduler, disable_tqdm=False,
                         callback=callback,print_freq=args.print_freq,alpha=args.alpha)
        #此处有改动，我们不再采用原方法的场景式训练方法，而是采用基于迁移学习的小样本学习方法
        prec1 = trainer.getValAcc(model)
        logger.info('Meta Val {}: {}'.format(epoch, prec1))
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        tqdm_loop.set_description('Best Acc {:.2f}'.format(best_prec1 * 100.))

        # Save checkpoint
        save_checkpoint(state={'epoch': epoch + 1,
                               'arch': args.arch,
                               'state_dict': model.state_dict(),
                               'best_prec1': best_prec1,
                               'optimizer': optimizer.state_dict()},
                        is_best=is_best,
                        folder=args.ckpt_path)
        if scheduler is not None:
            scheduler.step()

    # Final evaluation on test set
    logger.info("=> Creating model '{}'".format(args.arch))
    model = torch.nn.DataParallel(get_model(args)).cuda()

    logger.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    
if __name__ == "__main__":
    import pdb
    pdb.set_trace()
    main()