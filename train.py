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

from nnet import dataloader
from nnet.model import create_model

seed = 18
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_model(
        model,
        criterion,
        dataloaders,
        optimizer,
        metrics,
        lr_scheduler,
        num_epochs,
        bpath,
        device,
        min_lr=0.001
):
    """
    # source: https://towardsdatascience.com/transfer-learning-for-segmentation-using-deeplabv3-in-pytorch-f770863d6a42


    :param model:
    :param criterion:
    :param dataloaders:
    :param optimizer:
    :param metrics:
    :param lr_scheduler:
    :param bpath:
    :param num_epochs:
    :return:
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    # Use gpu if available

    model.to(device)
    # Initialize the log file for training and testing loss and metrics
    fieldnames = ['epoch', 'train_loss', 'val_loss'] + \
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

            # Iterate over data.
            for sample in tqdm(iter(dataloaders[phase])):
                # inputs = sample['image'].to(device)
                # masks = sample['mask'].to(device)
                inputs = sample[0].to(device)
                masks = sample[1].to(device).long()
                # zero the parameter gradients
                optimizer.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs['out'], masks.squeeze())
                    y_pred = outputs['out'].data.cpu().numpy().ravel()
                    y_true = masks.data.cpu().numpy().ravel()
                    # for name, metric in metrics.items():
                    #     if name == 'f1_score':
                    #         # Use a classification threshold of 0.1
                    #         batchsummary[f'{phase}_{name}'].append(
                    #             metric(y_true > 0, y_pred > 0.1))
                    #     else:
                    #         batchsummary[f'{phase}_{name}'].append(
                    #             metric(y_true.astype('uint8'), y_pred))

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
            if phase == 'train':
                if get_lr(optimizer) > min_lr:
                    lr_scheduler.step()

            batchsummary['epoch'] = epoch
            epoch_loss = loss
            batchsummary[f'{phase}_loss'] = epoch_loss.item()
            print('{} Loss: {:.4f}'.format(phase, loss))
        for field in fieldnames[3:]:
            batchsummary[field] = np.mean(batchsummary[field])
        print(batchsummary)
        with open(os.path.join(bpath, 'log.csv'), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)
            # deep copy the model
            if phase == 'val' and loss < best_loss:
                best_loss = loss
                best_model_wts = copy.deepcopy(model.state_dict())
        history.append(batchsummary)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Lowest Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history


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

    # create data handlers
    aug_transform = dataloader.augment_preprocess_generator()
    im_preproc = dataloader.image_preprocess_transforms_generator()
    lab_preproc = dataloader.labels_preprocess_transforms_generator()

    train_data = dataloader.VaihingenDataset(
        params.proc_data_path,
        augment=aug_transform,
        x_transform=im_preproc,
        y_transform=lab_preproc,
        split='t1')

    val_data = dataloader.VaihingenDataset(
        params.proc_data_path,
        x_transform=im_preproc,
        y_transform=lab_preproc,
        split='val')
    train_dataloader = DataLoader(train_data, batch_size=params.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=params.batch_size, shuffle=False)
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    # train model
    trained_model, history = train_model(
        model,
        criterion,
        dataloaders,
        optimizer,
        {},
        lr_scheduler,
        num_epochs,
        device=device,
        bpath='./')

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--proc_data_path', required=True)
    parser.add_argument('--n_classes', required=False, default=6, type=int)
    parser.add_argument('--num_epochs', required=True, type=int)
    parser.add_argument('--init_lr', required=False, default=0.05, type=float)
    parser.add_argument('--batch_size', required=False, default=16, type=int)
    parser.add_argument('--save_path', required=False, default='./trained_models')
    params = parser.parse_args()
    main(params)
