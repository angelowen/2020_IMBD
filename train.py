import math,os
import torch
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
from models import DNN,AttModel,ResModel,Transformer
from dataset import MLDataset
from torch.utils.data import DataLoader,random_split
from matplotlib import pyplot as plt
from argparse import ArgumentParser
import pandas as pd
from utils import model_builder,criterion_builder,schedule_builder,FILLNA,clean_file
from data_augment import data_aug
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# WRMSE
def WRMSE(preds, labels, device):
    weight = torch.tensor([
        0.05223, 0.0506, 0.05231, 0.05063, 0.05073,
        0.05227, 0.05177, 0.05186, 0.05076, 0.05063,
        0.0173, 0.05233, 0.05227, 0.05257, 0.05259,
        0.05222, 0.05204, 0.05185, 0.05229, 0.05074
    ]).to(device)
    wrmse = torch.pow(preds-labels, 2)
    wrmse = torch.sum(wrmse * weight)
    return wrmse.item()

# training curve
def visualize(record,valid_record, title):
    plt.title(title)
    a, = plt.plot(record)
    b, = plt.plot(valid_record)
    plt.legend([a,b], ["training", "validation"], loc=1)
    plt.savefig(f'./{title}.jpg')
    plt.show()
    plt.close()


# learning rate, epoch and batch size. Can change the parameters here.
def train(args): # Mymodel:epoch=1000
    train_loss_curve = []
    train_wrmse_curve = []
    valid_loss_curve = []
    valid_wrmse_curve = []
    # load model
    model = model_builder(args.model).to(device)
    model.train()
    data = pd.read_csv('train.csv', encoding='utf-8')
    if args.fillna:
        data = FILLNA()
    if args.data_aug:
        print("It may cost time!!")
        data = data_aug(data)
        print("Augmentation Complete!! Please check data_augment.csv")
    # dataset and dataloader
    dataset = MLDataset(data)
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=True)

    # loss function and optimizer
    criterion = criterion_builder(args.criterion)
    optimizer = torch.optim.AdamW(model.parameters(),lr = args.lr,weight_decay=0.01,amsgrad=1)
    n = int(train_size/args.batch_size)+1
    if args.scheduler:
        scheduler = schedule_builder(optimizer, args.scheduler, args.step, args.factor)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,epochs=epoch, steps_per_epoch=n, max_lr = 0.05)

    best = float("inf")
    # start training
    for e in range(args.epochs):
        train_loss = 0.0
        train_wrmse = 0.0
        valid_loss = 0.0
        valid_wrmse = 0.0

        print(f'\nEpoch: {e+1}/{args.epochs}')
        print('-' * len(f'Epoch: {e+1}/{args.epochs}'))
        # tqdm to disply progress bar
        for inputs, labels in tqdm(train_dataloader):
            # data from data_loader
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)
            outputs = model(inputs)
            # MSE loss and WRMSE
            loss = criterion(outputs, labels)
            wrmse = WRMSE(outputs, labels, device)
            # weights update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # update scheduler
            if args.scheduler:
                scheduler.step()
            # loss calculate
            train_loss += loss.item()
            train_wrmse += wrmse
        # =================================================================== #
        # If you have created the validation dataset,
        # you can refer to the for loop above and calculate the validation loss
        for inputs, labels in tqdm(valid_dataloader):
            # data from data_loader
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)
            outputs = model(inputs)
            # MSE loss and WRMSE
            loss = criterion(outputs, labels)
            wrmse = WRMSE(outputs, labels, device)
            # loss calculate
            valid_loss += loss.item()
            valid_wrmse += wrmse


        # =================================================================== #
        # save the best model weights as .pth file
        loss_epoch = train_loss / len(train_dataset)
        wrmse_epoch = math.sqrt(train_wrmse/len(train_dataset))
        valid_loss_epoch = valid_loss / len(valid_dataset)
        valid_wrmse_epoch = math.sqrt(valid_wrmse/len(valid_dataset))
        if valid_wrmse_epoch < best :
            best = valid_wrmse_epoch
            torch.save(model.state_dict(), 'mymodel.pth')
        print(f'Training loss: {loss_epoch:.4f}')
        print(f'Training WRMSE: {wrmse_epoch:.4f}')
        print(f'Valid loss: {valid_loss_epoch:.4f}')
        print(f'Valid WRMSE: {valid_wrmse_epoch:.4f}')
        # save loss and wrmse every epoch
        train_loss_curve.append(loss_epoch)
        train_wrmse_curve.append(wrmse_epoch)
        valid_loss_curve.append(valid_loss_epoch)
        valid_wrmse_curve.append(valid_wrmse_epoch)
        if args.tensorboard:
            writer.add_scalar('train/train_loss', loss_epoch, e)
            writer.add_scalar('train/wrmse_loss', wrmse_epoch, e)
            writer.add_scalar('validation/valid_loss', valid_loss_epoch, e)
            writer.add_scalar('validation/wrmse_loss', valid_wrmse_epoch, e)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], e)
    # generate training curve
    visualize(train_loss_curve,valid_loss_curve, 'Train Loss')
    visualize(train_wrmse_curve,valid_wrmse_curve, 'Train WRMSE')
    print("\nBest Validation WRMSE:",best)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str,default='dnn',metavar='DNN,AttModel,ResModel,Transformer',
                        help='choose the model to train(default: DNN)')
    parser.add_argument('--fillna',action='store_false', default=True,
                        help='Fill Nan with Kneighborsregressor (default: True)')
    parser.add_argument('--data_aug', action='store_true', default=False,
                        help='Data Augmentation with Autoencoder (default: False)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='set the learning rate (default: 1e-3)')
    parser.add_argument('--epochs', type=int, default=600,
                        help='set the epochs (default: 600)')
    parser.add_argument('--criterion', type=str, default='huber',choices=['huber', 'mse', 'l1'],
                        help="set criterion (default: 'huber')")
    parser.add_argument('--batch-size', type=int, default=128,
                        help='set the batch size (default: 128)')
    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='Use tensorboard to record data')
    # Sceduler setting
    parser.add_argument('--scheduler', action='store_true', default=False,
                        help='training with step or multi step scheduler (default: False)')
    parser.add_argument('--lr-method', type=str, dest='scheduler', 
                        help='training with chose lr scheduler (default: False)')
    parser.add_argument('--step', nargs='+', default=2,
                        help='decreate learning rate every few epochs (default: 2)')
    parser.add_argument('--factor', type=float, default=0.1,
                        help='set decreate factor (default: 0.1)')
    args = parser.parse_args()

    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        if  os.path.isdir('log'):
            log_dir = [int(i[3:]) for i in os.listdir('log')]
            n = max(log_dir)
            name = f"log{n+1}"
        else:
            name="log1"
            writer = SummaryWriter(f'./log/{name}')
            print(f"log stores in ./log/{name}") 
    train(args)