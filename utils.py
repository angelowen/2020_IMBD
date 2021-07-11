import torch.nn as nn
import torch.optim as optim
import pandas as pd #匯入pandas庫
from sklearn.neighbors import KNeighborsRegressor #匯入機器學習庫中的K近鄰迴歸模型
from sklearn.metrics import mean_squared_error #匯入機器學習庫中的均方誤差迴歸損失模型
import numpy as np
import os

def model_builder(model_name,**kwargs):
    from models import DNN,AttModel,ResModel,Transformer
    """[summary]
    Args:
        model_name (str)

    Returns:
        model(torch.nn.module): instantiate model
    """
    model = {
        'dnn': DNN,    
        'attmodel': AttModel,      
        'resmodel': ResModel,         
        'transformer': Transformer       
    }.get(model_name.lower())
    return model(**kwargs)

def criterion_builder(criterion='huber', **kwargs):
    """build specific criterion
    mse: MSELoss
    huber: SmoothL1Loss
    L1: L1Loss
    Args:
        criterion (str, optional): to instantiate loss function. Defaults to 'huber'.
    Returns:
        nn.Module: return loss function
    """

    return {
        'l1': nn.L1Loss(**kwargs),
        'mse': nn.MSELoss(**kwargs),
        'huber': nn.SmoothL1Loss(**kwargs),
    }[criterion.lower()]

def schedule_builder(optimizer,epochs, lr_method='cosine'):
    """declare scheduler
    Args:
        optimizer : parameter of lr_scheduler 
        lr_method (str, optional): choose which scheduler to be used. Defaults to 'cosine'.
    Returns:
        torch.optmizer.lr_sceduler
    """
    def poly_lr_scheduler(epoch, num_epochs=epochs, power=0.9):
        return (1 - epoch/num_epochs)**power

    if lr_method == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs)
    elif lr_method == 'lambdalr':
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=poly_lr_scheduler)

    return scheduler

def FILLNA():
    """
    Fill nan in trainl.csv
    """
    data = pd.read_csv('train.csv', encoding='utf-8')
    col = data.columns
    # find NaN
    is_NaN = data.isnull()
    row_has_NaN = is_NaN.any(axis=1)
    row_NaN = data[row_has_NaN].index.values.tolist()
    col_NaN =  data.columns[data.isna().any()].tolist()
    # Get 'train' data:
    train = data.drop(row_NaN, axis=0)
    train_x = train.drop(col_NaN, axis=1)
    train_y = train[col_NaN]
    # Get 'test' data:
    test = data.loc[row_NaN,:]
    test_x = test.drop(col_NaN, axis=1)
    test_y = test[col_NaN]

    knn = KNeighborsRegressor(10) #模型近鄰值手動設定成K = 10，K的值變大後，決策邊界將變得更加的平緩，曲線也變得更加的簡單。這種更簡單的情形，比較適合於大多數的數據。
    knn.fit(train_x, train_y) #X放入訓練集資料，Y放入目標輸出資料
    outputs = knn.predict(test_x) #輸出測試集結果
    outputs = pd.DataFrame(outputs,columns = col_NaN,index=row_NaN)

    nan = np.where(np.asanyarray(np.isnan(data)))
    for i,(a,b) in enumerate(zip(nan[0],nan[1])):
        # print(data.loc[a,col[b]],round(outputs.loc[a,col[b]],3))
        data.loc[a,col[b]] = round(outputs.loc[a,col[b]],3)

    return data

def clean_file(path):
    if os.path.isfile(path):
        os.remove(path)