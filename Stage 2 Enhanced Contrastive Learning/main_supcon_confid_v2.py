from __future__ import print_function

import os
import sys
import argparse
import time
import math
import random
from sklearn import metrics
# import tensorboard_logger as tb_logger
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
# from networks.resnet_big import SupConResNet
# from efficientnet_pytorch.model import SupConEfficientNet
from resnet_big import SupConResNet
from loss import SupConLoss, SimCLRLoss
from model import SupConXception
from InceptionV3 import SupConInceptionV3
import scipy.spatial
from dataset_customed import FolderDataset, select_FolderDataset
try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=3000,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=100,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.5,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='xception')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=299, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    opt.data_folder = '/home/xsc/experiment/Our_DeepfakeUCL/pesudo_label_experiment/FF++_deepfake_raw_all_jpg_small_face_video_acc71%'
    opt.val_data_folder = '/home/xsc/experiment/Our_DeepfakeUCL/pesudo_label_experiment/FF++_deepfake_raw_all/val'
    opt.select_data_folder = '/home/xsc/experiment/Our_DeepfakeUCL/pesudo_label_experiment/FF++_deepfake_raw_all/train'
    dataset_name = opt.data_folder.split('/')[-1]
    opt.model_path = './save/SupCon/FF++'
    if not os.path.isdir(opt.model_path):
        os.makedirs(opt.model_path)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(dataset_name)

    # iterations = opt.lr_decay_epochs.split(',')
    # opt.lr_decay_epochs = list([])
    # for it in iterations:
        # opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, dataset_name, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_loader(opt):
    # construct data loader
    train_transform = transforms.Compose([
        # transforms.Resize((224,224)),
        # transforms.RandomErasing(p=0.8,scale=(0.02, 0.20), ratio=(0.5, 2.0),inplace=True),
        transforms.RandomResizedCrop(size=opt.size, scale=(0.5, 1.)),
        # transforms.CenterCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
          transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.8,scale=(0.02, 0.20), ratio=(0.5, 2.0),inplace=True),
        # transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])  
    test_transform = transforms.Compose([transforms.Resize((299, 299)),
                                                               transforms.ToTensor(),
                                                               transforms.Normalize(std=(0.5, 0.5, 0.5),
                                                                                    mean=(0.5, 0.5, 0.5))])
    select_dataset = FolderDataset(img_folder_root=opt.select_data_folder, transform=test_transform)
    select_loader = torch.utils.data.DataLoader(select_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, drop_last=False, shuffle=False)                                                                                
    
    train_dataset = FolderDataset(img_folder_root=opt.data_folder, transform=TwoCropTransform(train_transform))
    
    # train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, drop_last=False)
    

    val_dataset = datasets.ImageFolder(root=opt.val_data_folder, transform=test_transform)
    val_loader= torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True, drop_last=False)

    return train_loader, select_loader, val_loader

def get_select_loder(select_img_path_list, new_assign_label):
    train_transform = transforms.Compose([
        # transforms.Resize((224,224)),
        # transforms.RandomErasing(p=0.8,scale=(0.02, 0.20), ratio=(0.5, 2.0),inplace=True),
        transforms.RandomResizedCrop(size=299, scale=(0.5, 1.)),
        # transforms.CenterCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.8,scale=(0.02, 0.20), ratio=(0.5, 2.0),inplace=True),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    select_train_dataset = select_FolderDataset(select_img_path_list=select_img_path_list, new_assign_label=new_assign_label, transform=TwoCropTransform(train_transform))
    select_train_loader = torch.utils.data.DataLoader(select_train_dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True, drop_last=False)
    return select_train_loader

#计算样本间距离
def distance(x, y, p=2):
    '''
    input:x(ndarray):第一个样本的坐标
          y(ndarray):第二个样本的坐标
          p(int):等于1时为曼哈顿距离,等于2时为欧氏距离
    output:distance(float):x到y的距离      
    '''   
    dis2 = np.sum(np.abs(x-y)**p) # 计算
    dis = np.power(dis2,1/p)
    return dis

#计算每个样本到质心的距离，并按照从小到大的顺序排列
def sorted_list(data,Cmass):
    '''
    input:data(ndarray):数据样本
          Cmass(ndarray):数据样本质心
    output:dis_list(list):排好序的样本到质心距离
    '''
    dis_list = []
    for i in range(len(data)):       # 遍历data数据，与质心cmass求距离
        dis_list.append(distance(Cmass,data[i][:]))
    dis_list = sorted(dis_list)      # 排序
    return dis_list

def select_confid_data(model, dataloader):
    labels = []
    result = torch.rand(0)
    img_path_list = []
    model.eval()
    with torch.no_grad():
        for i, (img_path, img, targets) in enumerate(dataloader):
            images = img.cuda(non_blocking=True)
            targets = targets.numpy()
            labels.extend(targets)
            feat_contrast= model(images)
            result = result.cuda(non_blocking=True)
            result = torch.cat([result, feat_contrast], dim=0)
            img_path_list += img_path
    
    result = result.cpu()
    result = result.detach().numpy()
    from sklearn.cluster import KMeans,SpectralClustering, DBSCAN
    from sklearn import metrics
    clf = KMeans(n_clusters=2, init='k-means++')
    fit_clf=clf.fit(result)
    clf.predict(result)
    # print(clf.cluster_centers_)
    # print(clf.cluster_centers_[0])
    center_0 = clf.cluster_centers_[0]
    center_0 = np.array(center_0)
    center_0 = center_0[None] # 扩充维度
    # print("----------")
    # print(clf.cluster_centers_[1])
    center_1 = clf.cluster_centers_[1]
    center_1 = np.array(center_1)
    center_1 = center_1[None] # 扩充维度
    dis_list_0 = []
    dis_list_1 = []
    img_path_list_0 = []
    img_path_list_1 = []
    y_pred = clf.fit_predict(result)
    for i, pred in enumerate(y_pred):
        result_ = np.array(result[i][:])
        result_ = result_[None]
        if pred == 0:
            # dis = distance(center_0, result[i][:])
            # print(result)
            dis = scipy.spatial.distance.cdist(center_0, result_,'cosine')    
            dis_list_0.append(dis)
            img_path_list_0.append(img_path_list[i])
        else:
            # dis = distance(center_1, result[i][:])
            dis = scipy.spatial.distance.cdist(center_1, result_,'cosine')
            dis_list_1.append(dis)
            img_path_list_1.append(img_path_list[i])
    # print(img_path_list)
    # print(img_path_list_0)
    # print(img_path_list_1)
    # sys.exit()
    sorted_id_0 = sorted(range(len(dis_list_0)), key=lambda k: dis_list_0[k])
    sorted_id_1 = sorted(range(len(dis_list_1)), key=lambda k: dis_list_1[k])

    nn_id_0 = sorted_id_0[:15000]
    nn_id_1 = sorted_id_1[:15000]
    # print(nn_id_0)
    # print(nn_id_1)
    # sys.exit()
    # nn_id = nn_id_0 + nn_id_1

    confid_img = []
    for i in nn_id_0:
        confid_img.append(img_path_list_0[i])
    for j in nn_id_1:
        confid_img.append(img_path_list_1[j])
    # print(confid_img)
    # sys.exit()
    y_trues = []
    # y_preds = []
    new_assign_label = [0]*len(nn_id_0) +  [1]*len(nn_id_1)
    right_count = 0
    for i in confid_img:
        if i.split('/')[-1].split('_')[0] == '0':
            y_trues.append(0)
        else:
            y_trues.append(1)
        # if i.split('/')[-2].split('_')[0] == '1':
        #     y_preds.append(1)
        # else:
        #     y_preds.append(0)
    test_acc = metrics.accuracy_score(y_trues, new_assign_label)
    # test_acc = right_count/len(confid_img)
    print(f'test_acc:{test_acc}')
    
    return test_acc, confid_img, new_assign_label


def set_model(opt):
    model = SupConInceptionV3()
   
    print("Model load success")
    criterion_contrast = SupConLoss(temperature=opt.temp)
    # criterion_contrast = SimCLRLoss(temperature=opt.temp)
    # criterion_classify = torch.nn.CrossEntropyLoss()

    # enable synchronized Batch Normalization
    # if opt.syncBN:
    #     model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
           model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion_contrast = criterion_contrast.cuda()
        # criterion_classify = criterion_classify.cuda()
        cudnn.benchmark = True

    return model, criterion_contrast


def train(train_loader, model, criterion_contrast, optimizer, scheduler, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    
    for idx, (_, images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        # print("image size:")
        # print(images[0].size())
        images = torch.cat([images[0], images[1]], dim=0)
        # print(images[0])
        # print("--------")
        # print(images[1])
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # print(labels)
        bsz = labels.shape[0]
        # print("batch_size:")
        # print(bsz)
        # warm-up learning rate
        # warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)
        # compute loss
        feat_contrast = model(images)
        # print(features)
        # print("features size")
        # print(features.size())
        # print(images)
        f1, f2 = torch.split(feat_contrast, [bsz, bsz], dim=0)
        feat_contrast = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        # if opt.method == 'SupCon':
        loss_contrast = criterion_contrast(feat_contrast, labels)
 
        loss = loss_contrast
        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses)) 

    return losses.avg

def val(model, dataloader):
    labels = []
    result = torch.rand(0)
    model.eval()
    with torch.no_grad():
        for i, (img, targets) in enumerate(dataloader):
            images = img.cuda(non_blocking=True)
            targets = targets.numpy()
            labels.extend(targets)
            feat_contrast= model(images)
            result = result.cuda(non_blocking=True)
            result = torch.cat([result, feat_contrast], dim=0)
    result = result.cpu()
    result = result.detach().numpy()
    from sklearn.cluster import KMeans,SpectralClustering
    from sklearn import metrics
    kmeans = KMeans(n_clusters=2, init='k-means++')
    # print(kmeans)
    y_pred = kmeans.fit_predict(result)

    ACC = metrics.accuracy_score(labels, y_pred)
    print("ACC")
    print(ACC)
    ARI = metrics.adjusted_rand_score(labels, y_pred)
    print("ARI(调整兰德指数)")
    print(ARI)

    NMI = metrics.normalized_mutual_info_score(labels, y_pred)
    print("NMI(标准化互信息)")
    print(NMI)

    return ACC
def main():
    opt = parse_option()

    # build data loader
    train_loader, select_loader, val_loader= set_loader(opt)

    # build model and criterion
    model, criterion_contrast = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)
    loss_list =  []
    max_acc = 0
    val_acc_list = []
    test_confid_data_acc_list = []
    count = 0
    best_val_acc = 0
    # training routine
    for epoch in range(1, opt.epochs + 1):
       #adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        
        if epoch %  100 == 0:
            new_train_data_path = os.path.join(os.getcwd(), 'use_for_DC_epoch_1000.csv')
            train_path_list = []
            new_assign_label = []
            with open(new_train_data_path) as f:
                reader = csv.reader(f)
                column=[row for row in  reader]
                for i in range(2, len(column)):
                    path = column[i][0]
                    label = column[i][1]
                    train_path_list.append(path)
                    new_assign_label.append(label)
        
            train_loader = get_select_loder(select_img_path_list=train_path_list, new_assign_label=new_assign_label)
            # flag = model.init_weights()
            optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)
            # print("model.init_weights:")
            # print(flag)
    
        loss = train(train_loader, model, criterion_contrast, optimizer, scheduler, epoch, opt)
        val_acc = val(model, val_loader)
        test_confid_data_acc, confid_img, new_assign_label = select_confid_data(model, select_loader)
        if test_confid_data_acc > max_acc:
            import pandas as pd 
            import csv
            def pd_tocsv(file_path, new_assign_label, sava_path=None):  # pandas库储存数据到excel
                dfData = {  # 用字典设置DataFrame所需数据
                    '文件路径': file_path,
                    '聚类分配新标签':new_assign_label
                }
                df = pd.DataFrame(dfData)  # 创建DataFrame
                csv_file_Name = os.path.join(os.getcwd(), sava_path)
                df.to_csv(csv_file_Name, index=False)  # 存表，去除原始索引列（0,1,2...）

            sava_path = 'use_for_DC_epoch_1000.csv'
            pd_tocsv(file_path=confid_img, new_assign_label=new_assign_label, sava_path=sava_path)
            max_acc = test_confid_data_acc
        print(f'max_acc:{max_acc}')
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        loss_list.append(loss)
        val_acc_list.append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        print(f'best_val_acc:{best_val_acc}')
        test_confid_data_acc_list.append(test_confid_data_acc)

        if (epoch+1) % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

    # draw loss 
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')
    plt.figure(1)
    plt.plot(loss_list,'b',label = 'Supcon_loss')
    plt.ylabel('loss')
    plt.xlabel('iter_num')
    plt.legend()
    plt.savefig("SupCon_loss.jpg")

    plt.switch_backend('Agg')
    plt.figure(2)
    plt.plot(val_acc_list,'b',label = 'val_acc_list')
    plt.ylabel('val_acc_list')
    plt.xlabel('iter_num')
    plt.legend()
    plt.savefig("SupCon_val_acc_list.jpg")

    plt.switch_backend('Agg')
    plt.figure(3)
    plt.plot(test_confid_data_acc_list,'b',label = 'test_confid_data_acc_list')
    plt.ylabel('test_confid_data_acc_list')
    plt.xlabel('iter_num')
    plt.legend()
    plt.savefig("SupCon_test_confid_data_acc_list.jpg")

if __name__ == '__main__':
    main()


