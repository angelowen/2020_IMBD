import math
import torch
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
from dataset import MLDataset,AEDataset
from torch.utils.data import DataLoader,random_split
from matplotlib import pyplot as plt
from models import AE
import os
import pandas as pd
from utils import clean_file
def visualize(record,valid_record, title):
    plt.title(title)
    a, = plt.plot(record)
    b, = plt.plot(valid_record)
    plt.legend([a,b], ["training", "validation"], loc=1)
    plt.savefig(f'./{title}.jpg')
    # plt.show()
    plt.close()

# learning rate, epoch and batch size. Can change the parameters here.
def data_aug(data, lr=0.001, epoch=800, batch_size=128):
    folder = 'data_aug'
    save_path = f'{folder}/data_augment.csv'
    clean_file(save_path)
    store_e = [700,750,800]
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        for i in store_e:
            result = test(data,folder,i)
        return result

    train_loss_curve = []
    valid_loss_curve = []
    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AE()
    model = model.to(device)
    model.train()

    dataset = AEDataset(data)
    train_size = int(0.85 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)

    # loss function and optimizer
    # can change loss function and optimizer you want
    criterion  = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = lr)
    best = 100
    # start training
    for e in range(epoch):
        train_loss = 0.0
        valid_loss = 0.0

        print(f'\nEpoch: {e+1}/{epoch}')
        print('-' * len(f'Epoch: {e+1}/{epoch}'))
        # tqdm to disply progress bar
        for inputs in tqdm(train_dataloader):
            # data from data_loader
            inputs = inputs.float().to(device)
            outputs = model(inputs,device)
            loss = criterion(outputs, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        for inputs in tqdm(valid_dataloader):
            # data from data_loader
            inputs = inputs.float().to(device)
            outputs = model(inputs,device)
            # MSE loss 
            loss = criterion(outputs, inputs)
            # loss calculate
            valid_loss += loss.item()
        # save the best model weights as .pth file
        loss_epoch = train_loss / len(train_dataset)
        valid_loss_epoch = valid_loss / len(valid_dataset)
        # if valid_loss_epoch < best :
        #     best = valid_loss_epoch
        #     torch.save(model.state_dict(), 'data_aug.pth')
        if e in store_e:
            torch.save(model.state_dict(), f'{folder}/ep{e}data_aug.pth')
            print(f"training in epoch  {e},start augment data!!")
            result = test(data,folder,e)
        print(f'Training loss: {loss_epoch:.4f}')
        print(f'Valid loss: {valid_loss_epoch:.4f}')
        # save loss  every epoch
        train_loss_curve.append(loss_epoch)
        valid_loss_curve.append(valid_loss_epoch)
    # generate training curve
    # visualize(train_loss_curve,valid_loss_curve, 'Data Augmentation')
    return result

def test(data,folder,e):
    label_col = list(data.columns)
    result = data
    model = AE()
    model.load_state_dict(torch.load(f'{folder}/ep{e}data_aug.pth', map_location='cpu'))
    model.eval()
    dataset = AEDataset(data)
    dataloader = DataLoader(dataset=dataset, batch_size=128, shuffle=False)
    for inputs in tqdm(dataloader):
        outputs = model(inputs.float(),'cpu')
        for i in range(len(outputs)):
            tmp = outputs[i].detach().numpy()
            tmp = pd.DataFrame([tmp], columns=label_col)
            result= pd.concat([result, tmp], ignore_index=True)
    result.to_csv(f'{folder}/data_augment.csv',mode='a', header=True, index=False)
    return result

if __name__ == '__main__':
    data = pd.read_csv('train.csv')
    data = data.fillna(data.median())
    data_aug(data)
    