import torch,os
from pathlib import Path
import pandas as pd
from dataset import MLDataset
from models import DNN,AttModel,ResModel,Transformer
from sklearn.preprocessing import MinMaxScaler,normalize
from argparse import ArgumentParser
from utils import model_builder,clean_file,FILLNA
def find_log():
    if  os.path.isdir('log'):
        log_dir = [int(i[3:]) for i in os.listdir('log')]
        n = max(log_dir)
        name = f"log{n}"
    else:
        name="output"
    print(f"load model in ./log/{name}")
    return f"./log/{name}/mymodel.pth"

def test(args,path):
    # load model and use weights we saved before.
    model = model_builder(args.model)
    pth_file = find_log()
    try:
        model.load_state_dict(torch.load(pth_file, map_location='cpu'))
    except:
        print("Please Remember to change the model name as same as your training model!!",end="\n")
        print("Exit with no results")
        return
    model.eval()
    # load testing data
    data = pd.read_csv('test.csv', encoding='utf-8')
    label_col = [
        'Input_A6_024' ,'Input_A3_016', 'Input_C_013', 'Input_A2_016', 'Input_A3_017',
        'Input_C_050', 'Input_A6_001', 'Input_C_096', 'Input_A3_018', 'Input_A6_019',
        'Input_A1_020', 'Input_A6_011', 'Input_A3_015', 'Input_C_046', 'Input_C_049',
        'Input_A2_024', 'Input_C_058', 'Input_C_057', 'Input_A3_013', 'Input_A2_017'
    ]
    data = FILLNA('test.csv')
    data = data.fillna(data.median())
    # testing data normalized
    # scaler = MinMaxScaler(feature_range=(-1, 1))  
    # data = scaler.fit_transform(data)
    data = normalize(data,norm='l1')
    data = pd.DataFrame(data)
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
    result.to_csv(path, index=False)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str,default='dnn',metavar='DNN,AttModel,ResModel,Transformer',
                        help='choose the model to train(default: DNN)')
    args = parser.parse_args()
    path = 'Result.csv'
    clean_file(path)
    test(args,path)
        
