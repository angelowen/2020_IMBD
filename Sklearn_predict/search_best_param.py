from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
## Get data
label_col = [
            'Input_A6_024' ,'Input_A3_016', 'Input_C_013', 'Input_A2_016', 'Input_A3_017',
            'Input_C_050', 'Input_A6_001', 'Input_C_096', 'Input_A3_018', 'Input_A6_019',
            'Input_A1_020', 'Input_A6_011', 'Input_A3_015', 'Input_C_046', 'Input_C_049',
            'Input_A2_024', 'Input_C_058', 'Input_C_057', 'Input_A3_013', 'Input_A2_017'
        ]
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)
data = pd.read_csv('train.csv', encoding='utf-8')
data = data.fillna(data.median())
y = data[label_col] # (348, 20)
X = data.drop(label_col, axis=1)
test = pd.read_csv('test.csv', encoding='utf-8')
test = test.fillna(test.median())


gbm = LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.05, n_estimators=20)
pipe_gbm = Pipeline([('reg', MultiOutputRegressor(gbm))])

grid_param = {
    'reg__estimator__max_depth' : [3, 5, 10, 15, 20, 30,50],
    'reg__estimator__num_leaves' : [10,20,30,35],
    'reg__estimator__reg_alpha': [0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5],
    'reg__estimator__reg_lambda': [0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5],
     'reg__estimator__learning_rate': [0.005,0.01, 0.02, 0.05, 0.1],
     'reg__estimator__n_estimators' : [25 ,50 ,75,100,200]
}
print(pipe_gbm.get_params().keys())
gs_gbm = (GridSearchCV(estimator=pipe_gbm, 
                      param_grid=grid_param, 
                      cv=5,
                      scoring = 'neg_mean_squared_error',
                      n_jobs = -1))

gs_gbm = gs_gbm.fit(X,y)
outputs = pd.DataFrame(gs_gbm.predict(test)) 
outputs.to_csv('Result.csv')
print(gs_gbm.best_params_)


'''
# 网格搜索，参数优化
estimator = LGBMRegressor(num_leaves=31)
param_grid = {
    # 'reg_alpha': [0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5],
    # 'reg_lambda': [0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5],
    # 'learning_rate': [0.005,0.01, 0.02, 0.05, 0.1],
    'n_estimators': [25,50 ]#,100, 200
}
gbm = GridSearchCV(estimator, param_grid,cv = 5)
gbm = MultiOutputRegressor(gbm).fit(X, y)
outputs = pd.DataFrame(gbm.predict(test)) 
outputs.to_csv('Result.csv')
'''
