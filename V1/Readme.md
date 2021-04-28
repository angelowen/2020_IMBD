# 2020 智慧製造大數據比賽
* 加工機台參數預測
* 以加工機台完整的 加工參數 和 加工品質 作為訓練資料，於測試
階段預測 20 項重點參數
* 所有欄位以代號表示，無法得知確切意涵
* 總共 348 筆資料 每筆資料包含 217 項參數設定與 6 個鑽孔機加工品質的
輸出結果加工機台 Input 維度 = 223, Output 維度 = 20

* 目前最佳 WRMSE = 0.068 file="result0.068.csv" pth-file="0.055model.pth" 
* (評分檔) = "TestResult.xlsm"

* 待嘗試: 參數初始化問題-pretrain model，缺失值改用 random forest預測, model 改進 && ensambling 嘗試

## Reference
* [ensembling](https://ithelp.ithome.com.tw/articles/10250317)
