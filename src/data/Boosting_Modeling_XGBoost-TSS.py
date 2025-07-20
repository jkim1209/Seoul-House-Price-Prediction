#%%
# 필요한 라이브러리 import & 데이터 불러오기
from IPython.display import display
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder
import joblib

# set kr font
import matplotlib.font_manager
fontlist = [font.name for font in matplotlib.font_manager.fontManager.ttflist]
# print(fontlist)
krfont = ['Malgun Gothic', 'NanumGothic', 'AppleGothic']
for font in krfont:
    if font in fontlist:
        mpl.rc('font', family=font)

# 마이너스 기호 깨짐 방지
mpl.rcParams['axes.unicode_minus'] = False

# Modeling algorithms
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LinearRegression, LassoCV
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR, SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor, BaggingRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from xgboost import plot_importance as plot_importance_xgb
from lightgbm import plot_importance as plot_importance_lgbm
from catboost import Pool, CatBoostRegressor
from mlxtend.regressor import StackingRegressor, StackingCVRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from reg_custom import *

#%%
df = pd.read_csv('../../data/processed/cleaned_data/train_row_cleaned.csv')
test_df = pd.read_csv('../../data/processed/cleaned_data/test_clean.csv')

df_cleaned = df.copy()






#%%
df['계약일자dt'] = pd.to_datetime(df['계약일자'])
df['계약년월dt'] = pd.to_datetime(df['계약년월'], format='%Y%m')
df['계약일자'] = df['계약일자dt'].dt.strftime('%Y%m%d').astype(int)
df['계약년월'] = df['계약년월dt'].dt.strftime('%Y%m').astype(int)
df = df.sort_values(by='계약일자dt').reset_index(drop=True)

# 거래가격은 로그를 씌워 이용
df['log_target'] = np.log1p(df['target'])

# 버스, 지하철 반경 거리에 있는 정보 활용
subway_weights = {'반경_300m_지하철역_수': 3, '반경_500m_지하철역_수': 2, '반경_1km_지하철역_수': 1}
bus_weights    = {'반경_300m_버스정류장_수': 3, '반경_500m_버스정류장_수': 2, '반경_1km_버스정류장_수': 1}
df['반경_지하철역_가중합'] = sum(df[col] * w for col, w in subway_weights.items())
df['반경_버스정류장_가중합'] = sum(df[col] * w for col, w in bus_weights.items())
test_df['반경_지하철역_가중합'] = sum(test_df[col] * w for col, w in subway_weights.items())
test_df['반경_버스정류장_가중합'] = sum(test_df[col] * w for col, w in bus_weights.items())

# # 자치구, 법정동은 타겟 인코딩
# te1 = TargetEncoder(cols=['자치구'])
# df[['자치구']] = te1.fit_transform(df[['자치구']], df['log_target'])
# test_df[['자치구']] = te1.transform(test_df[['자치구']])

# te2 = TargetEncoder(cols=['법정동'], smoothing=20)
# df[['법정동']] = te2.fit_transform(df[['법정동']], df['log_target'])
# test_df[['법정동']] = te2.transform(test_df[['법정동']])

# 브랜드등급은 레이블 인코딩
le = LabelEncoder()
df['브랜드등급'] = le.fit_transform(df['브랜드등급'])
test_df['브랜드등급'] = le.transform(test_df['브랜드등급'])

# 계약년월을 시계열인덱스화
df['계약년월idx'] = ((df['계약년도'] - 2007) * 12 + df['계약월']).astype(int)
test_df['계약년월idx'] = ((test_df['계약년도'] - 2007) * 12 + test_df['계약월']).astype(int)


# 이미지 저장경로 생성
image_save_dir = '../../images/jangwon/Model_XGB'
os.makedirs(image_save_dir, exist_ok=True)

# 모델 저장경로 생성
model_save_dir = '../../model'
os.makedirs(model_save_dir, exist_ok=True)

# 예측값 저장경로 생성
prediction_save_dir = '../../data/processed/submissions/jangwon'
os.makedirs(prediction_save_dir, exist_ok=True)

# GridSearchCV, RandomizedSearchCV에 TimeSeriesSplit적용
tscv = TimeSeriesSplit(n_splits=5, test_size=10000, gap=0, max_train_size=None) 



#%%
# Feature 수정방향 생각해보기

display(df['자치구'].nunique()), display(test_df['자치구'].unique())            # type: ignore # train에 자치구 25개 / test에 자치구 3개밖에 없음
display(df['법정동'].nunique()), display(test_df['법정동'].unique())            # type: ignore # train에 법정동 334개 / test에 법정동 14개밖에 없음
display(df['브랜드등급'].nunique()), display(test_df['브랜드등급'].unique())
df.info(), test_df.info()



#%%
X_col = [ 
    # '법정동',
    '계약년월idx', '강남3구여부', '전용면적', '층',
    # '홈페이지유무', '사용허가여부',                     # LightGBM결과 feature importance가 낮은 열 삭제
    '연식', '브랜드등급', '아파트이름길이', 
    '반경_지하철역_가중합', '지하철최단거리',
    '반경_버스정류장_가중합', '버스최단거리',
    '인구비중', '좌표X', '좌표Y'
    # '성비(남/여)', 
    # 'loanrate_12m'
]

y_col = ['log_target']

df = df[X_col + y_col]

# X_train, X_test, Y_train, Y_test = datasplit(df, y_col)

# TimeSeries 분할이용
X_train, X_test, Y_train, Y_test = datasplit_ts(df, y_col, test_size=0.01)



#%%
# 스케일링
## 최종 test에 대해서도 같은 스케일링 작업
# MinMaxScaler는 변수의 최소값과 최대값을 이용해서 모든 값을 0~1 사이로 바꾸는 도구
# fit() : 최소값, 최대값 계산만 함
# transform() : 이미 계산된 최소/최대 기준으로 값을 변환함
# 절대 test 데이터로 fit하면 안 됨 (그건 미래 정보로 과거 학습하는 것과 같음 (데이터 누수))

# 1. 먼저 train 기준으로 스케일러 fit
scaler = preprocessing.MinMaxScaler()
scaler.fit(X_train)  # train 데이터로만 최소/최대 계산

# 2. 나머지는 transform만 적용
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)
test_df_scaled = scaler.transform(test_df[X_col])  # 원래 test df도 transform만 적용


test_df_scaled = pd.DataFrame(test_df_scaled, columns=X_col, index=test_df.index)










#%%
################################
#  XGBoost with CV
################################
model_xgb = XGBRegressor(random_state=123, objective='reg:squarederror')

# # 코드 테스트용
# params = {
#     'n_estimators': [100],
#     'learning_rate': [0.01, 0.1],
#     'max_depth': [10],
#     'min_child_samples': [5],
#     'subsample': [1.0],
#     'colsample_bytree': [1.0],
# }

# Search 1
params = {
    'n_estimators': [1000],     # early stopping 할 것이므로 크게 설정
    'learning_rate': [0.1],
    'max_depth': [5, 7, 9],
    'gamma': [0],
    'min_child_weight': [5, 7],
    'subsample': [0.7, 1.0],
    'colsample_bytree': [1.0]
}
# => Best params =  {'colsample_bytree': 1.0, 'gamma': 0, 'learning_rate': 0.1, 'max_depth': 9, 'min_child_weight': 5, 'n_estimators': 1000, 'subsample': 1.0}
# Train / Valid / Test RMSE : 13187.70 / 25693.88 / 27692.26


model_xgb_cv = GridSearchCV(estimator=model_xgb, param_grid=params, 
                            cv=tscv, scoring='neg_root_mean_squared_error',   
                            n_jobs=-1, verbose=2)    
model_xgb_cv.fit(X_train_scaled, Y_train)
print("XGBoost 최적 하이퍼 파라미터: ", model_xgb_cv.best_params_)


# feature Importance
xgb_model = model_xgb_cv.best_estimator_
importances = xgb_model.feature_importances_
features = X_train.columns

plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.title("XGBoost Feature Importances")
save_path = os.path.join(image_save_dir, 'XGB_Feature_Importance.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()




#%%
# TimeSeriesSplit
for train_index, valid_index in tscv.split(X_train_scaled):
    pass  # 마지막 반복만 유지됨

X_train_final = X_train_scaled[train_index]
Y_train_final = Y_train.iloc[train_index]
X_valid_final = X_train_scaled[valid_index]
Y_valid_final = Y_train.iloc[valid_index]
X_valid_final

# 최적 파라미터로 최종 모델 학습 (early stopping 포함)
# best_params_ 복사 후 수정
params = model_xgb_cv.best_params_.copy()
params['n_estimators'] = 10000  # early stopping을 위해 충분히 크게 설정

# 수정된 파라미터로 모델 학습
model_xgb_cv_final = XGBRegressor(
    **params,
    eval_metric='rmse',
    random_state=123,
    objective='reg:squarederror',
    early_stopping_rounds=20
)

model_xgb_cv_final.fit(
    X_train_final, Y_train_final,
    eval_set = [(X_valid_final, Y_valid_final)],
    verbose=2
)




Y_trpred = pd.DataFrame(model_xgb_cv_final.predict(X_train_final), 
                        index=Y_train_final.index, columns=['Pred'])
Y_valpred = pd.DataFrame(model_xgb_cv_final.predict(X_valid_final), 
                        index=Y_valid_final.index, columns=['Pred'])
Y_tepred = pd.DataFrame(model_xgb_cv_final.predict(X_test_scaled), 
                        index=Y_test.index, columns=['Pred'])

plot_prediction(pd.concat([Y_train_final, Y_trpred], axis=1).reset_index().iloc[:, 1:])
save_path = os.path.join(image_save_dir, 'XGB_TestPred.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()

plot_prediction(pd.concat([Y_valid_final, Y_valpred], axis=1).reset_index().iloc[:, 1:])
save_path = os.path.join(image_save_dir, 'XGB_ValidPred.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()

plot_prediction(pd.concat([Y_test, Y_tepred], axis=1).reset_index().iloc[:, 1:])
save_path = os.path.join(image_save_dir, 'XGB_TestPred.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()

Y_train_final_true = np.expm1(Y_train_final)
Y_valid_final_true = np.expm1(Y_valid_final)
Y_test_true = np.expm1(Y_test)
Y_trpred_true = np.expm1(Y_trpred)
Y_valpred_true = np.expm1(Y_valpred)
Y_tepred_true = np.expm1(Y_tepred)

Score_xgb_valid = evaluation_reg_trte(Y_train_final_true, Y_trpred_true, Y_valid_final_true, Y_valpred_true)
Score_xgb_valid.index = ['Train', 'Valid']
display(Score_xgb_valid)

Score_xgb = evaluation_reg_trte(Y_train_final_true, Y_trpred_true, Y_test_true, Y_tepred_true)
display(Score_xgb)
##########################################################################


#%%
Resid_tr = Y_train.squeeze() - Y_trpred.squeeze()
Resid_val = Y_valid_final.squeeze() - Y_valpred.squeeze()
Resid_te = Y_test.squeeze() - Y_tepred.squeeze()

sns.scatterplot(x=Y_trpred.squeeze(), y=Resid_tr)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Train Residual Plot (log)")
save_path = os.path.join(image_save_dir, 'XGB_TrainLogResidPlot.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()


sns.scatterplot(x=Y_valpred.squeeze(), y=Resid_val)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Validation Residual Plot (log)")
save_path = os.path.join(image_save_dir, 'XGB_ValidLogResidPlot.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()


sns.scatterplot(x=Y_tepred.squeeze(), y=Resid_te)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Test Residual Plot (log)")
save_path = os.path.join(image_save_dir, 'XGB_TestLogResidPlot.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()

Resid_tr_true = Y_train_final_true.squeeze() - Y_trpred_true.squeeze()
Resid_val_true = Y_valid_final_true.squeeze() - Y_valpred_true.squeeze()
Resid_te_true = Y_test_true.squeeze() - Y_tepred_true.squeeze()

sns.scatterplot(x=Y_trpred_true.squeeze(), y=Resid_tr_true)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Train Residual Plot")
save_path = os.path.join(image_save_dir, 'XGB_TrainResidPlot.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()


sns.scatterplot(x=Y_valpred_true.squeeze(), y=Resid_val_true)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Validation Residual Plot")
save_path = os.path.join(image_save_dir, 'XGB_ValidResidPlot.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()


sns.scatterplot(x=Y_tepred_true.squeeze(), y=Resid_te_true)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Test Residual Plot")
save_path = os.path.join(image_save_dir, 'XGB_TestResidPlot.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()


#%%
# 모델 저장
model_path = os.path.join(model_save_dir, 'XGBoost.pkl')
joblib.dump(model_xgb_cv_final, model_path)

model_xgb_cv_final = joblib.load(model_path)

# 대회 test데이터에 대해 예측
test_pred_log = model_xgb_cv_final.predict(test_df_scaled)
test_pred = np.expm1(test_pred_log)

XGB_CV_prediction = test_pred.copy()
XGB_CV_prediction = pd.DataFrame(XGB_CV_prediction, columns=["target"])

# 정수형으로 변환
XGB_CV_prediction["target"] = np.clip(np.round(XGB_CV_prediction["target"]), 0, None).astype(int)
print(XGB_CV_prediction.head())

# 예측값저장
submission_path = os.path.join(prediction_save_dir, 'XGB_prediction.csv')
XGB_CV_prediction.to_csv(submission_path, index=False)

