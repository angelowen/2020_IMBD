import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
import pandas as pd
from sklearn.svm import SVR
from sklearn.linear_model import RidgeCV, MultiTaskLassoCV, MultiTaskElasticNetCV
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from mlxtend.regressor import StackingCVRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib
from sklearn.preprocessing import MinMaxScaler,normalize
import os,shutil

folder = "Result"
if os.path.exists(folder):
    shutil.rmtree(folder)
os.makedirs(folder)

def make_prediction(model,name):
    outputs = pd.DataFrame(model.predict(test))
    outputs.to_csv(f'Result/{name}_Result.csv')
    print(f"{name} is training over!!")

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
X = normalize(X,norm='l2')
test = normalize(test,norm='l2')
# ridge是l2正則化的線性迴歸，lasso則是帶l1正則化的線性迴歸

# 線性模型
alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))
svr = make_pipeline(RobustScaler(), SVR(C= 10, epsilon= 0.08, gamma=0.03,))

lgbm = LGBMRegressor(objective='regression', 
                                       num_leaves=31,
                                       max_depth = 10,
                                       learning_rate=0.05, 
                                       n_estimators=50,
                                       max_bin=200, 
                                       verbose=-1,
                                       reg_alpha= 0.03,
                                       reg_lambda= 0.03,
                                       )

# Gradient Boosting regression

gbr = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =42)


# xgboost regression
xgboost = XGBRegressor(learning_rate=0.01,n_estimators=1000,
                                     max_depth=3, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:squarederror', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.00006)

## KNN
knn = KNeighborsRegressor(n_neighbors=14) #10-> 12 good

# random forest
rf = RandomForestRegressor(n_estimators = 2000, random_state = 26)
# DecisionTree

print("Start Training ~")

stack_gen = StackingCVRegressor(regressors=(xgboost,lgbm,gbr,knn,rf), #,svr ridge,svr,
                                meta_regressor=gbr,
                                use_features_in_secondary=True)
stack_gen_model = MultiOutputRegressor(stack_gen).fit(X, y)

outputs = pd.DataFrame((stack_gen_model.predict(test)))
outputs.to_csv('Result/Stack_Result.csv')
# joblib.dump(stack_gen_model, 'stack_model') # save model
# ---------------------------------------------------------------------
# rf_model = MultiOutputRegressor(rf).fit(X, y)
# ridge_model = ridge.fit(X, y) 
# svm_model = MultiOutputRegressor(svr).fit(X, y)
knn_model = knn.fit(X, y) 
lgbm_model = MultiOutputRegressor(lgbm).fit(X, y)
gbr_model = MultiOutputRegressor(gbr).fit(X, y)
xgb_model = MultiOutputRegressor(xgboost).fit(X, y)

print("Model Training Over")

# make_prediction(rf_model,"random_forest_model")
# make_prediction(ridge_model,"ridge_model")
# make_prediction(svm_model,"svm_model")
make_prediction(knn_model,"knn_model")
make_prediction(lgbm_model,"lgbm_model")
make_prediction(gbr_model,"gbr_model")
make_prediction(xgb_model,"xgb_model")


outputs = ( 
            (0.3 * gbr_model.predict(test)) + \
            (0.15 * xgb_model.predict(test)) + \
            (0.15 * lgbm_model.predict(test)) + \
            (0.2 * knn_model.predict(test)) + \
            pd.DataFrame(0.2 * stack_gen_model.predict(test)))

outputs.to_csv('Result/Ensemble_Result.csv')

# loaded_model = joblib.load('stack_model')
# result = loaded_model.predict(test)
# print(result)