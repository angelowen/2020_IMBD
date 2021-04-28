import torch
from pathlib import Path
import pandas as pd
from dataset import MLDataset
from models import MyModel
from sklearn.preprocessing import MinMaxScaler,normalize


def test():
    # load model and use weights we saved before.
    model = MyModel()
    model.load_state_dict(torch.load('0.072model.pth', map_location='cpu'))
    model.eval()
    # load testing data
    data = pd.read_csv('test.csv', encoding='utf-8')
    label_col = [
        'Input_A6_024' ,'Input_A3_016', 'Input_C_013', 'Input_A2_016', 'Input_A3_017',
        'Input_C_050', 'Input_A6_001', 'Input_C_096', 'Input_A3_018', 'Input_A6_019',
        'Input_A1_020', 'Input_A6_011', 'Input_A3_015', 'Input_C_046', 'Input_C_049',
        'Input_A2_024', 'Input_C_058', 'Input_C_057', 'Input_A3_013', 'Input_A2_017'
    ]

    # ================================================================ #
    # if do some operations with training data,
    # do the same operations to the testing data in this block
    data = data.fillna(0)
    # testing data normalized
    # scaler = MinMaxScaler(feature_range=(-1, 1))  
    # data = scaler.fit_transform(data)
    data = normalize(data,norm='l1')
    data = pd.DataFrame(data)


    # ================================================================ #
    # convert dataframe to tensor, no need to rewrite
    # input: torch.Size([95, 223]) output: torch.Size([95, 20])
    inputs = data.values
    inputs = torch.tensor(inputs)
    # predict and save the result
    result = pd.DataFrame(columns=label_col)
    outputs = model(inputs.float())
    for i in range(len(outputs)):
        tmp = outputs[i].detach().numpy()
        tmp = pd.DataFrame([tmp], columns=label_col)
        result= pd.concat([result, tmp], ignore_index=True)
    result.to_csv('result.csv', index=False)

if __name__ == '__main__':
    test()
