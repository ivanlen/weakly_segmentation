import argparse
import csv
import copy
import json
import numpy as np
import os
import pickle
import time
import torch
import random
from tqdm import tqdm
from torch.utils.data import DataLoader

from nnet import data
from nnet.model import create_model
from nnet import losses as custom_losses
from utils.metrics import IoU

seed = 18
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def compute_metrics(model, val_dataloader, n_classes, device):
    """
    Compute the mIoU and class IoU in the validation set
    """
    model.eval()
    iou_calculator = IoU(num_classes=n_classes)
    for x, y in tqdm(iter(val_dataloader)):
        inputs = x.to(device)
        masks = (y[0]).to(device).long().squeeze()
        outputs = model(inputs)
        pred = outputs['out'].cpu()
        iou_calculator.add(pred, masks.squeeze())

    class_ious = iou_calculator.value()[0][:n_classes - 1]
    mean_iou = np.nanmean(iou_calculator.value()[0][:n_classes - 1])
    metrics_results = {
        'mIoU': mean_iou,
        'IoUs': class_ious}
    return metrics_results


def train_model(
        model,
        criterion,
        weak_criterion,
        n_classes,
        dataloaders,
        optimizer,
        metrics,
        lr_scheduler,
        num_epochs,
        bpath,
        device,
        weak_lambda,
        min_lr=0.001):
    """
    Training model function.
    Trains the model, and compute metrics.
    It retruns the trained model, the training history, and the model metrics on the validation set.

    # some code was taked from: https://towardsdatascience.com/transfer-learning-for-segmentation-using-deeplabv3-in-pytorch-f770863d6a42

    :param model: torch model to be trained
    :param criterion: strong loss function
    :param weak_criterion: weak loss function
    :para n_classes: number of classes including background
    :param weak_lambda: multiplication factor used to compute the total loss
    :param dataloaders: train and val dataloaders
    :param optimizer: torch optimizer
    :param metrics:
    :param lr_scheduler: learing rate scheduler
    :param bpath: log path
    :param device: device to use to train de model
    :param num_epochs: number of epochs
    :return: trained_model, history, metrics
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    model.to(device)
    # Initialize the log file for training and testing loss and metrics
    fieldnames = ['epoch', 'train_loss', 'val_loss'] + \
                 ['train_weak_loss', 'val_weak_loss', 'train_seg_loss', 'val_seg_loss'] + \
                 [f'train_{m}' for m in metrics.keys()] + \
                 [f'val_{m}' for m in metrics.keys()]
    with open(os.path.join(bpath, 'log.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    history = []
    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        # Initialize batch summary
        batchsummary = {a: [0] for a in fieldnames}

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            epoch_seg_loss = []
            epoch_weak_loss = []
            # Iterate over data.
            for x, y in tqdm(iter(dataloaders[phase])):
                inputs = x.to(device)
                masks = (y[0]).to(device).long()
                oh_classes = (y[1]).to(device).float()
                weak_filter = (y[2].squeeze()).to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    if weak_criterion:
                        torch_total_loss = 0
                        # print(outputs['out'][weak_filter].shape, outputs['out'][~weak_filter].shape)
                        if torch.any(weak_filter):
                            weak_loss = weak_criterion(outputs['out'][weak_filter], oh_classes[weak_filter])
                            weak_loss_value = weak_loss.item()
                            torch_total_loss += weak_lambda * weak_loss
                            epoch_weak_loss.append(weak_lambda * weak_loss_value)
                        else:
                            weak_loss_value = np.nan
                            epoch_weak_loss.append(weak_loss_value)
                        if torch.any(~weak_filter):
                            segmentation_loss = criterion(outputs['out'][~weak_filter], masks.squeeze()[~weak_filter])
                            segmentation_loss_value = segmentation_loss.item()
                            torch_total_loss += segmentation_loss
                            epoch_seg_loss.append(segmentation_loss_value)
                        else:
                            segmentation_loss_value = np.nan
                            epoch_weak_loss.append(np.nan)
                    else:
                        segmentation_loss = criterion(outputs['out'], masks.squeeze())
                        segmentation_loss_value = segmentation_loss.item()
                        epoch_seg_loss.append(segmentation_loss_value)
                        weak_loss_value = np.nan
                        torch_total_loss = segmentation_loss
                    if phase == 'train':
                        torch_total_loss.backward()
                        optimizer.step()
                print(segmentation_loss_value, weak_loss_value)
            if phase == 'train':
                if get_lr(optimizer) > min_lr:
                    lr_scheduler.step()

            batchsummary['epoch'] = epoch
            epoch_avg_weak_loss = np.nanmean(epoch_weak_loss) if epoch_weak_loss else 0
            epoch_avg_seg_loss = np.nanmean(epoch_seg_loss) if epoch_seg_loss else 0
            epoch_loss = epoch_avg_weak_loss + epoch_avg_seg_loss
            batchsummary[f'{phase}_seg_loss'] = epoch_avg_seg_loss
            batchsummary[f'{phase}_weak_loss'] = epoch_avg_weak_loss
            batchsummary[f'{phase}_loss'] = epoch_loss
            print('{} seg loss: {:.4f}'.format(phase, epoch_avg_seg_loss))
            print('{} weak Loss: {:.4f}'.format(phase, epoch_avg_weak_loss))
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
        for field in fieldnames[3:]:
            batchsummary[field] = np.mean(batchsummary[field])
        print(batchsummary)
        with open(os.path.join(bpath, 'log.csv'), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)
            # deep copy the model
            if phase == 'val' and segmentation_loss < best_loss:
                best_loss = segmentation_loss
                best_model_wts = copy.deepcopy(model.state_dict())
        history.append(batchsummary)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Lowest Loss: {:4f}'.format(best_loss))
    metrics = compute_metrics(model, dataloaders['val'], n_classes, device)
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history, metrics


def main(params):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'traning on :{device}')
    if not torch.cuda.is_available():
        print('WARNING, training on CPU, this might take a while, be patient...')

    model = create_model(params.n_classes)
    opt_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adadelta(opt_params, lr=params.init_lr)
    criterion = torch.nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8, verbose=True)

    num_epochs = params.num_epochs
    proc_data_file = f'{params.proc_data_path}/proc_data.json'
    # create data handlers
    aug_transform = data.augment_preprocess_generator()
    im_preproc = data.image_preprocess_transforms_generator()
    lab_preproc = data.labels_preprocess_transforms_generator()

    train_data = data.VaihingenDataset(
        proc_data_file,
        augment=aug_transform,
        x_transform=im_preproc,
        y_transform=lab_preproc,
        split=params.train_data)

    val_data = data.VaihingenDataset(
        proc_data_file,
        x_transform=im_preproc,
        y_transform=lab_preproc,
        split='val')

    print(f'train split: {params.train_data}\ttrain samples {train_data.__len__()}')
    print(f'                \tval samples {val_data.__len__()}')
    train_dataloader = DataLoader(train_data, batch_size=params.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=params.batch_size, shuffle=False)
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    if params.weak_loss == 'None':
        weak_criterion = False
        weak_lambda = 0.0
    elif params.weak_loss == 'average_pooling':
        # n_classes - 1, we discard background class here
        weak_criterion = custom_losses.AveragePoolingClassLoss(n_classes=params.n_classes - 1)
        weak_lambda = params.weak_lambda
    else:
        raise Exception(f'{params.weak_loss} is not a valid loss.')
    # train model
    trained_model, history, metrics = train_model(
        model,
        criterion,
        weak_criterion,
        params.n_classes,
        dataloaders,
        optimizer,
        {},
        lr_scheduler,
        num_epochs,
        bpath='./',
        device=device,
        weak_lambda=weak_lambda)

    # save results
    model_id = int(time.time())
    os.makedirs(params.save_path, exist_ok=True)
    # save params
    export_params_name = f'{params.save_path}/params_{model_id}.json'
    with open(export_params_name, 'w') as f:
        json.dump(vars(params), f)
    # save best model
    model_name = f'{params.save_path}/model_{model_id}.pt'
    torch.save(trained_model.state_dict(), model_name)
    # save training history
    history_path = f'{params.save_path}/history_{model_id}.pickle'
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    # save metrics
    metrics_path = f'{params.save_path}/metrics_{model_id}.pickle'
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--proc_data_path', required=True)
    parser.add_argument('--n_classes', required=False, default=6, type=int)
    parser.add_argument('--train_data', required=False, default='t1', type=str)
    parser.add_argument('--num_epochs', required=True, type=int)
    parser.add_argument('--init_lr', required=False, default=0.05, type=float)
    parser.add_argument('--batch_size', required=False, default=16, type=int)
    parser.add_argument('--weak_loss', required=False, default='None', type=str)
    parser.add_argument('--weak_lambda', required=False, default=0.1, type=float)
    parser.add_argument('--save_path', required=False, default='./trained_models')
    params = parser.parse_args()
    main(params)
