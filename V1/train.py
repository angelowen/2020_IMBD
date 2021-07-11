import math
import torch
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
from models import MyModel
from dataset import MLDataset
from torch.utils.data import DataLoader,random_split
from matplotlib import pyplot as plt


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
    plt.show()
    plt.savefig(f'./{title}.png')
    plt.close()


# learning rate, epoch and batch size. Can change the parameters here.
def train(lr=0.001, epoch=600, batch_size=32):
    train_loss_curve = []
    train_wrmse_curve = []
    valid_loss_curve = []
    valid_wrmse_curve = []
    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyModel()
    model = model.to(device)
    model.train()

    # dataset and dataloader
    # can use torch random_split to create the validation dataset
    dataset = MLDataset()
    train_size = int(0.9 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)

    # loss function and optimizer
    # can change loss function and optimizer you want
    criterion  = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best = 100
    # start training
    for e in range(epoch):
        train_loss = 0.0
        train_wrmse = 0.0
        valid_loss = 0.0
        valid_wrmse = 0.0

        print(f'\nEpoch: {e+1}/{epoch}')
        print('-' * len(f'Epoch: {e+1}/{epoch}'))
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
    # generate training curve
    visualize(train_loss_curve,valid_loss_curve, 'Train Loss')
    visualize(train_wrmse_curve,valid_wrmse_curve, 'Train WRMSE')
    print("\nBest Validation WRMSE:",best)

if __name__ == '__main__':
    train()