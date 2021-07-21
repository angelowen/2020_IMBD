# 2020 智慧製造大數據比賽
* 加工機台參數預測
* 以加工機台完整的 加工參數 和 加工品質 作為訓練資料，於測試
階段預測 20 項重點參數
* 所有欄位以代號表示，無法得知確切意涵
* 總共 348 筆資料 每筆資料包含 217 項參數設定與 6 個鑽孔機加工品質的
輸出結果加工機台 Input 維度 = 223, Output 維度 = 20

* 目前最佳 WRMSE = 0.061 
* (評分檔) = "TestResult.xlsm"

* 待嘗試: 參數初始化問題-pretrain model，缺失值test.csv預測, model 改進 && ensambling 嘗試

## Usage
### Training
`python train.py [--model XXX] [--fillna] [--scheduler] [--data_aug] [--criterion XXX]`
`ex: python train.py --model resmodel --fillna --scheduler`
* 有4種model選擇，測試時模型名稱記得一同更改
    * DNN(default)
    * ResModel(better)
    * AttModel(bad)
    * Transformer(bad)
* 可使用 `fillna` 參數(kneighborsregressor方法)補值,default data.median()
* `data_aug` 參數用autoencoder 增加資料量
* `--tensorboard` 參數可選擇是否使用tensorboard紀錄,defalt = True in line 187
* `--criterion` 參數可選擇3種loss function
    * huber loss(default)
    * MSE loss
    * L1 loss
* `--scheduler` 啟用optim.lr_scheduler.cosineannealing , default lr=0.001
### Testing
`python test.py [--model XXX] `
## Reference
* [ensembling](https://ithelp.ithome.com.tw/articles/10250317)
* https://yanwei-liu.medium.com/python%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%E7%AD%86%E8%A8%98-%E5%8D%81%E4%B8%80-%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%E7%9A%84%E8%B3%87%E6%96%99%E5%89%8D%E8%99%95%E7%90%86%E6%8A%80%E8%A1%93-4dbd27560743
* https://ithelp.ithome.com.tw/articles/10202059
