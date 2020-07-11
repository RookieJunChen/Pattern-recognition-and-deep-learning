import os
import argparse
import logging
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

sys.path.append('./')
from dataset.backbone_dataset import ImageNetDataset
from model.Backbone import ResNet
from dataset.transform import RandomResize

def get_args():
    # parse options
    parser = argparse.ArgumentParser(description='Resnet Imagenet Training')
    parser.add_argument('--net-name', type=str, default='ResNet101', help='ResNet101|ResNet101_SE|Resnet50|ResNet50_SE')
    parser.add_argument('--gpu-ids', type=str, default='', help='gpu ids: e.g. 0  0,1,2, 0,2')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer [sgd|adam|adamW]')
    parser.add_argument('--lr', default=1e-1, type=float, help='learning rate')
    parser.add_argument('--batch-size', type=int, default=256, help='input batch size')
    parser.add_argument('--nThreads', default=0, type=int, help='# threads for loading data')
    parser.add_argument('--max-epochs', type=int, default=120, help='Epochs of training')
    parser.add_argument('--data-dir', type=str, default='/share/data/zhangyabo2/ImageNet/', help='root directory of dataset, no need to specify train or val dir')
    parser.add_argument('--log-dir', type=str, default='./logs/backbone/ResNet101', help='log directory')
    parser.add_argument('--tensor-dir', type=str, default='./tensor_logs/backbone/ResNet101', help='tensorboard log directory') 
    parser.add_argument('-sd', '--save-dir', type=str, default='./models/backbone', help='saving model directory')
    parser.add_argument('--start', type=int, default=0, help='start epoch number')
    return parser.parse_args()


def init_logger(log_file=None, log_dir=None, log_level=logging.INFO, mode='w', stdout=True):
    """
    """
    import datetime
    def get_date_str():
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M-%S')

    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    if log_dir is None:
        log_dir = '~/temp/log/'
    if log_file is None:
        log_file = 'log_' + get_date_str() + '.txt'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, log_file)
    print('log file path:' + log_file)

    logging.basicConfig(level=logging.DEBUG,
                        format=fmt,
                        filename=log_file,
                        filemode=mode)

    if stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(log_level)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    return logging


def main():
    args = get_args()
    writer = SummaryWriter(args.tensor_dir)
    logger = init_logger(log_dir=args.log_dir)
    
    # specify cuda devices
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device('cuda') if args.gpu_ids else torch.device('cpu')
    logger.info(f'using device {device}')
    
    # make checkpoints dir
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    
    # Data argumentation
    transform_train = transforms.Compose([
        RandomResize([256, 480]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_set = ImageNetDataset(args.data_dir, transform=transform_train, mode='train')
    val_set = ImageNetDataset(args.data_dir, transform=transform_test, mode='val')
    
    #print(len(train_set)) 
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.nThreads)
    test_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.nThreads)
    
    
    # model construct
    if args.net_name == 'ResNet101':
        model = ResNet(name=101, is_SE=False)
    elif args.net_name == 'ResNet101_SE':
        model = ResNet(name=101, is_SE=True)
    elif args.net_name == 'ResNet50':
        model = ResNet(name=50, is_SE=False)
    elif args.net_name == 'ResNet50_SE':
        model = ResNet(name=50, is_SE=True)
    else:
        raise NotImplementedError('Unrecognized model: '+args.net_name)
    
    if os.path.exists(os.path.join(args.save_dir, args.net_name+'_best.pth')):
        model.load_state_dict(torch.load(os.path.join(args.save_dir, args.net_name+'_best.pth'), map_location=device))

    # model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    print("cuda number:",torch.cuda.device_count())
#    if os.path.exists(os.path.join(args.save_dir, args.net_name+'_best.pth')):
#        model.load_state_dict(torch.load(os.path.join(args.save_dir, args.net_name+'_best.pth'), map_location=device))
    model.to(device)
    # specify loss function
    criterion = nn.CrossEntropyLoss()
    # specify optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam([{'params': model.parameters(), 'initial_lr': args.lr}], lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD([{'params': model.parameters(), 'initial_lr': args.lr*0.0001}], lr=args.lr*0.0001, momentum=0.9, weight_decay=1e-4)
    else:
        raise NotImplementedError('Unrecognized optimizer: '+args.optimizer)
    
    milestones = [20, 40, 60, 80, 100]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1, last_epoch=args.start-1)
    
    print(optimizer.param_groups[0]['lr'])
    print(scheduler.get_lr())
    n_epochs = args.max_epochs
    best_top1_acc = 0
    best_top5_acc = 0
    
    for epoch in range(args.start, n_epochs):
        train_loss = 0.0
        model.train()
        for train_data, train_gt in train_loader:
            train_data = train_data.to(device)
            train_gt = train_gt.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(train_data)
            # calculate the loss
            loss = criterion(output, train_gt)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(train_loader.dataset)
        
        writer.add_scalar('training_loss', loss.item(), epoch)
        logger.info('Epoch:  {} , Training Loss: {}'.format(epoch, train_loss))
        scheduler.step()
        
        # eval
        model.eval()  # prep model for evaluation
    
        # test process does not require back propagation
        top1_acc = 0
        top5_acc = 0
        with torch.no_grad():
            for test_data, test_gt in test_loader:
                test_data = test_data.to(device)
                #test_gt = test_gt.to(device)
                test_gt = test_gt.cpu().numpy()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(test_data)
                output = output.cpu().numpy()
                # convert output probabilities to predicted class
                top1_acc += (output.argmax(axis=-1)==test_gt).mean()
                top5_ids = output.argsort()[:, -5:][:, ::-1]
                top5_num = 0
                # calculate top5 accuracy
                for ids, gt in zip(top5_ids, test_gt):
                    if gt in ids:
                        top5_num += 1
                top5_acc += top5_num / len(test_gt)
        top1_acc /= len(test_loader)
        top5_acc /= len(test_loader)
        
        # save best model
        if (top1_acc+top5_acc) >= (best_top1_acc+best_top5_acc):
            best_top1_acc = top1_acc
            best_top5_acc = top5_acc
            logger.info(f'EPOCH: {epoch}\nBEST TOP1 ACC: {top1_acc}, BEST TOP5 ACC: {top5_acc}')
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), os.path.join(args.save_dir, args.net_name+'_best.pth'))
#                print("SUCCESS save")
#                break
            else:
                torch.save(model.state_dict(), os.path.join(args.save_dir, args.net_name+'_best.pth'))
        writer.add_scalar('top1_acc', top1_acc, epoch)
        writer.add_scalar('top5_acc', top5_acc, epoch)
        logger.info(f'EPOCH: {epoch}\nTOP1 ACC: {top1_acc}, TOP5 ACC: {top5_acc}\nBEST TOP1 ACC: {best_top1_acc}, BEST TOP5 ACC: {best_top5_acc}')
    

if __name__ == '__main__':
    main()
