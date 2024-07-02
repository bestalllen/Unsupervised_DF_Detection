#!/usr/bin/env python3

import os
import sys
import argparse
import time
import numpy as np
import datetime
import json
import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances
from lib.train_util import AverageMeter, save_model
from data.transform import TwoCropTransform, get_transforms
from data.dataset import CustomImageDataset
from loss import ECLoss
from model import ECL


def args_func():
    parser = argparse.ArgumentParser('argument for enhanced contrastive learner training')

    parser.add_argument('--print_freq', type=int, default=100,
                        help='print frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--select_confidence_sample', type=int, default=None,
                        help='epoch of select confidence sample')
    parser.add_argument('--k', type=float, default=None,
                        help='select top k confidence samples')
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')
    parser.add_argument('--backbone', type=str, default='Xception')
    parser.add_argument('--data_folder', type=str, default=None, help='path to DF dataset')
    parser.add_argument('--pseudo_label_file', type=str, default=None, help='path to pseudo_label.json')
    parser.add_argument('--image size', type=int, default=299, help='parameter for RandomResizedCrop')

    args = parser.parse_args()
    return args


def set_loader(pseudo_label_dict, args):
    # construct data loader

    train_transform = get_transforms(name="train")
    train_dataset = CustomImageDataset(pseudo_label_dict, args.data_folder, TwoCropTransform(train_transform))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)

    val_transform = get_transforms(name="val")

    select_confidence_dataset = CustomImageDataset(pseudo_label_dict, args.data_folder, val_transform)

    select_confidence_loader = torch.utils.data.DataLoader(
        select_confidence_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)

    return train_loader, select_confidence_loader


def set_model(args):
    model = ECL()
    criterion = ECLoss(temperature=args.temp)
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.backbone = torch.nn.DataParallel(model.backbone)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, args):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels, _) in enumerate(train_loader):

        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        bsz = labels.shape[0]

        # compute loss
        _, features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = criterion(features, labels)

        # update metric
        losses.update(loss.item(), bsz)

        # Adam optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


def select_confidence_sample(model, dataloader, args):
    with torch.no_grad():
        model.eval()
        all_features = []
        all_data_names = []
        for i, (img, _, image_name) in enumerate(dataloader):
            images = img.cuda(non_blocking=True)
            _, feat_contrast = model(images)
            all_features.append(feat_contrast.cpu().numpy())
            all_data_names.extend(image_name)

    all_features = np.concatenate(all_features)

    kmeans = KMeans(n_clusters=2, init='k-means++')
    kmeans.fit(all_features)
    cluster_centers = kmeans.cluster_centers_

    labels = kmeans.labels_
    all_distances = cosine_distances(all_features, cluster_centers)

    # Extract data names and labels for each cluster
    cluster_0_data = [(name, label) for name, label in zip(all_data_names, labels) if label == 0]
    cluster_1_data = [(name, label) for name, label in zip(all_data_names, labels) if label == 1]

    # Extract the corresponding distance for each cluster
    distances_cluster_0 = all_distances[np.array([i for i, label in enumerate(labels) if label == 0]), 0]
    distances_cluster_1 = all_distances[np.array([i for i, label in enumerate(labels) if label == 1]), 1]

    # Sort the data within each cluster by distance and take the first K
    K_0 = int(args.k * len(distances_cluster_0))
    K_1 = int(args.k * len(distances_cluster_1))
    top_k_cluster_0 = [data for _, data in sorted(zip(distances_cluster_0, cluster_0_data))][:K_0]
    top_k_cluster_1 = [data for _, data in sorted(zip(distances_cluster_1, cluster_1_data))][:K_1]

    result_list = top_k_cluster_0 + top_k_cluster_1
    result_dict = {image_name: str(label) for image_name, label in result_list}

    return result_dict


def save_path(args):
    data_folder_name = os.path.basename(args.data_folder)
    model_path = './save/SupCon/{}_models'.format(data_folder_name)
    tb_path = './save/SupCon/{}_tensorboard'.format(data_folder_name)

    save_time = str(datetime.datetime.now())

    tb_folder = os.path.join(tb_path, save_time)
    if not os.path.isdir(tb_folder):
        os.makedirs(tb_folder)

    save_folder = os.path.join(model_path, save_time)
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    return tb_folder, save_folder


def main():

    args = args_func()

    with open(args.pseudo_label_file, 'r') as file:
        pseudo_label_dict = json.load(file)

    # build data loader
    train_loader, select_confidence_loader = set_loader(pseudo_label_dict, args)

    # build model and criterion
    model, criterion = set_model(args)

    # build optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    tb_folder, save_folder = save_path(args)

    # tensorboard
    logger = tb_logger.Logger(logdir=tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, args.epochs + 1):

        # train for one epoch
        time1 = time.time()

        if epoch % args.select_confidence_sample == 0 and epoch != args.epochs:
            
            # save the model from the previous cycle
            save_file = os.path.join(save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, args, epoch, save_file)

            # select the training sample for the new cycle
            confidence_label_dict = select_confidence_sample(model, select_confidence_loader, args)
            train_loader, _ = set_loader(confidence_label_dict, args)
            model.init_weights() # alternative strategy

        loss = train(train_loader, model, criterion, optimizer, epoch, args)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

    # save the last model
    save_file = os.path.join(save_folder, 'last.pth')
    save_model(model, optimizer, args, args.epochs, save_file)


if __name__ == '__main__':
    main()
