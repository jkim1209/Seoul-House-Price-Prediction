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
image_save_dir = '../../images/jangwon/Model_RF'
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

train_subset = df[df['자치구'].isin(['강남구','강동구','강북구'])]
train_subset



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
#  Random Forest with CV
################################
model_rf = RandomForestRegressor(random_state=123)

# # 코드 테스트용
# params = {
#     'n_estimators': [200],
#     'max_depth': [10, 20],
#     'min_samples_split': [5],
#     'min_samples_leaf': [5]
# }

# Trial 1
params = {
    'n_estimators': [200, 300],
    'max_depth': [10, 20],
    # 'min_samples_split': [5, 10],
    'min_samples_leaf': [5, 10, 15],
    'max_features': ['sqrt'],
}
# => Best params = {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 5, 'n_estimators': 200}
# Train / Test RMSE :  6952  /  20178



# GridSearchCV 템플릿 생성
model_rf_cv = GridSearchCV(estimator=model_rf, param_grid=params, 
                           cv=tscv, scoring='neg_root_mean_squared_error',  
                           n_jobs=-1, verbose=2)   

# # RandomizedSearchCV 이용
# model_rf_cv = RandomizedSearchCV(
#     estimator=model_rf,
#     param_distributions=params,
#     n_iter=20,  
#     cv=tscv,    
#     scoring='neg_root_mean_squared_error',
#     n_jobs=-1,
#     verbose=2,
#     random_state=123
# )

# 학습
model_rf_cv.fit(X_train_scaled, Y_train)
print("최적 하이퍼 파라미터: ", model_rf_cv.best_params_)


#%%
model_rf_cv_final = model_rf_cv.best_estimator_
# model_rf_cv_final.fit(X_train_scaled, Y_train)        # 위에서 이미 학습한 정보로 또 할 필요없음

Y_trpred = pd.DataFrame(model_rf_cv_final.predict(X_train_scaled), 
                        index=Y_train.index, columns=['Pred'])
Y_tepred = pd.DataFrame(model_rf_cv_final.predict(X_test_scaled), 
                        index=Y_test.index, columns=['Pred'])

plot_prediction(pd.concat([Y_train, Y_trpred], axis=1).reset_index().iloc[:,1:])
save_path = os.path.join(image_save_dir, 'RF_TrainPred.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()

plot_prediction(pd.concat([Y_test, Y_tepred], axis=1).reset_index().iloc[:,1:])
save_path = os.path.join(image_save_dir, 'RF_TestPred.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()

Y_train_true = np.expm1(Y_train)
Y_test_true = np.expm1(Y_test)
Y_trpred_true = np.expm1(Y_trpred)
Y_tepred_true = np.expm1(Y_tepred)

Score_rf = evaluation_reg_trte(Y_train_true, Y_trpred_true, Y_test_true, Y_tepred_true)
display(Score_rf)


#%%
# Feature Importance
importances = model_rf_cv_final.feature_importances_
feat_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})
feat_df = feat_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feat_df)
plt.title('Random Forest Feature Importance')
plt.tight_layout()
save_path = os.path.join(image_save_dir, 'RF_Feature_Importance.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()



#%%
Resid_tr = Y_train.squeeze() - Y_trpred.squeeze()
Resid_te = Y_test.squeeze() - Y_tepred.squeeze()

sns.scatterplot(x=Y_trpred.squeeze(), y=Resid_tr)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Train Residual Plot")
save_path = os.path.join(image_save_dir, 'RF_TrainLogResidPlot.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()


sns.scatterplot(x=Y_tepred.squeeze(), y=Resid_te)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Test Residual Plot")
save_path = os.path.join(image_save_dir, 'RF_TestLogResidPlot.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()

Resid_tr_true = Y_train_true.squeeze() - Y_trpred_true.squeeze()
Resid_te_true = Y_test_true.squeeze() - Y_tepred_true.squeeze()

sns.scatterplot(x=Y_trpred_true.squeeze(), y=Resid_tr_true)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Train Residual Plot")
save_path = os.path.join(image_save_dir, 'RF_TrainResidPlot.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()


sns.scatterplot(x=Y_tepred_true.squeeze(), y=Resid_te_true)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Test Residual Plot")
save_path = os.path.join(image_save_dir, 'RF_TestResidPlot.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()



#%%
# 모델 저장
model_path = os.path.join(model_save_dir, 'RandomForest.pkl')
joblib.dump(model_rf_cv_final, model_path)

model_rf_cv_final = joblib.load(model_path)

# 대회 test데이터에 대해 예측
test_pred_log = model_rf_cv_final.predict(test_df_scaled)
test_pred = np.expm1(test_pred_log)

RF_CV_prediction = test_pred.copy()
RF_CV_prediction = pd.DataFrame(RF_CV_prediction, columns=["target"])

# 정수형으로 변환
RF_CV_prediction["target"] = np.clip(np.round(RF_CV_prediction["target"]), 0, None).astype(int)
print(RF_CV_prediction.head())

# 예측값저장
submission_path = os.path.join(prediction_save_dir, 'RF_prediction.csv')
RF_CV_prediction.to_csv(submission_path, index=False)

