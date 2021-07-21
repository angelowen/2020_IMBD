import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler,normalize


df = pd.read_csv('train.csv', encoding='utf-8')
df = df.fillna(df.median())
# df = pd.DataFrame(normalize(df,norm='l1'),columns = df.keys())
label_col = [
            'Input_A6_024' ,'Input_A3_016', 'Input_C_013', 'Input_A2_016', 'Input_A3_017',
            'Input_C_050', 'Input_A6_001', 'Input_C_096', 'Input_A3_018', 'Input_A6_019',
            'Input_A1_020', 'Input_A6_011', 'Input_A3_015', 'Input_C_046', 'Input_C_049',
            'Input_A2_024', 'Input_C_058', 'Input_C_057', 'Input_A3_013', 'Input_A2_017'
        ]
# data = df.fillna(df.median())
df_label = df[label_col] # (348, 20)
df_train = df.drop(label_col, axis=1)
train_col = df_train.columns.tolist()


## quantile
# y = df['Input_A1_003']
# removed_outliers = y.between(y.quantile(.05), y.quantile(.95))
# print(str(y[removed_outliers].size) + "/" + str(y.size) + " data points remain.") 

## correspondence matrix
# 產生相關係數矩陣
corr = df.corr()
print(corr)
print("\n")
corr = corr.loc[train_col,label_col]
print(corr)
# plt.figure()
# sns.heatmap(corr.loc[train_col,label_col])
# plt.show()
c = df_train.columns[corr['Input_A6_024'].abs().where(corr['Input_A6_024']>0.1).notna()].tolist()
print(c,corr['Input_A6_024'][c])
## 資料分布
# plt.figure()
# sns.distplot(df['Input_A1_003'],fit = norm) # 加入常態分布曲線
# plt.show()
