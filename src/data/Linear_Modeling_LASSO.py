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
from sklearn import preprocessing, metrics
from sklearn.preprocessing import LabelEncoder
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
from sklearn.model_selection import TimeSeriesSplit

# for regression
from reg_custom import *



#%%
df = pd.read_csv('../../data/processed/cleaned_data/train_row_cleaned.csv')

df['계약일자dt'] = pd.to_datetime(df['계약일자'])
df['계약년월dt'] = pd.to_datetime(df['계약년월'], format='%Y%m')
df['계약일자'] = df['계약일자dt'].dt.strftime('%Y%m%d').astype(int)
df['계약년월'] = df['계약년월dt'].dt.strftime('%Y%m').astype(int)

df.info()
display(df.columns)
print(df.head())


df_cleaned = df.copy()


# 이미지 저장경로 생성
image_save_dir = '../../images/jangwon/Model_LASSO'
os.makedirs(image_save_dir, exist_ok=True)

# 모델 저장경로 생성
model_save_dir = '../../model'
os.makedirs(model_save_dir, exist_ok=True)

# 예측값 저장경로 생성
prediction_save_dir = '../../data/processed/submissions/jangwon'
os.makedirs(prediction_save_dir, exist_ok=True)

# LASSO에 TimeSeriesSplit적용
tscv = TimeSeriesSplit(n_splits=5, test_size=10000, gap=0, max_train_size=None) 


#%%
# 변수중요도 확인
# 각 범주간 target과의 상관관계 및 R^2분석
date_vars = ['계약년도', '계약년월', '계약일자']
loanrate_vars = ['loanrate_1m', 'loanrate_3m', 'loanrate_6m', 'loanrate_12m']
subway_vars = ['반경_1km_지하철역_수', '반경_500m_지하철역_수', '반경_300m_지하철역_수']
bus_vars = ['반경_1km_버스정류장_수', '반경_500m_버스정류장_수', '반경_300m_버스정류장_수']

group_dict = {
    'date' : date_vars,
    'loanrate': loanrate_vars,
    'subway': subway_vars,
    'bus': bus_vars,
}

importance_results = []

for group_name, var_list in group_dict.items():
    for var in var_list:
        x = df[[var]].values
        y = df['target'].values
        # 선형 회귀 적합
        model = LinearRegression().fit(x, y)
        y_pred = model.predict(x)
        r2 = metrics.r2_score(y, y_pred)
        corr = np.corrcoef(df[var], df['target'])[0, 1]
        importance_results.append({
            'group': group_name,
            'variable': var,
            'correlation': corr,
            'r_squared': r2,
        })

importance_df = pd.DataFrame(importance_results)
importance_df_sorted = importance_df.sort_values(by=['group', 'r_squared'], ascending=[True, False])

print(importance_df_sorted)

# 선형회귀 R^2기준 선택변수
# bus               : 반경_1km_버스정류장_수
# date              : 계약일자 혹은 계약년월
# loanrate          : loanrate_12m
# subway            : 반경_1km_지하철역_수





#%%
# target의 분포 확인
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))

sns.distplot(np.log1p(df['target']),kde=True, bins=100, ax = axes[0])
axes[0].set_title("Distribution of log(target)")

sm.qqplot(np.log1p(df['target']), line='s', ax = axes[1]) 
axes[1].set_title("QQ Plot of log(target)")

plt.tight_layout()
save_path = os.path.join(image_save_dir, '거래가격 정규성 확인.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()



#%%
# 회귀분석 준비
df['log_target'] = np.log1p(df['target'])

# '자치구' 원핫인코딩
district_dummy = pd.get_dummies(df['자치구'], prefix='자치구', drop_first=True)
df = pd.concat([df, district_dummy], axis = 1)

# '브랜드등급' 레이블인코딩
le = LabelEncoder()
df['브랜드등급_labeled'] = le.fit_transform(df['브랜드등급'])


# 이용할 변수들 선언
# exclude_cols = ['법정동', '계약일자dt', '계약년월dt', '자치구', '브랜드등급', 'target', 'log_target']
# X_col = [col for col in df.columns if col not in exclude_cols]

# 이용할 변수들 선언
# df['계약날짜idx'] = (df['계약일자dt'] - pd.to_datetime("2007-01-01")).dt.days 
df['계약년월idx'] = ((df['계약년도'] - 2007) * 12 + df['계약월']).astype(int)
X_col = ['계약년월idx', '전용면적', '층', 
          '강남3구여부', '홈페이지유무', '사용허가여부', '좌표X', '좌표Y',
          '연식', '아파트이름길이', '지하철최단거리', '버스최단거리', 
          '총인구수', '인구비중', '성비(남/여)',
          'loanrate_12m',
          '반경_1km_버스정류장_수',
          '반경_1km_지하철역_수',
          '브랜드등급_labeled'] + [col for col in df if col.startswith('자치구_')]
y_col = ['log_target']


lasso_df = df[X_col + y_col]

#%%
# 데이터분리
X_train, X_test, Y_train, Y_test = datasplit_ts(lasso_df, y_col)

scaler = preprocessing.MinMaxScaler()
scaler_fit = scaler.fit(X_train)
X_train_fe = pd.DataFrame(scaler_fit.transform(X_train), columns = X_train.columns, index= X_train.index)
X_test_fe = pd.DataFrame(scaler_fit.transform(X_test), columns = X_test.columns, index = X_test.index)


scaler = preprocessing.MinMaxScaler()
X_train_fe, X_test_fe = scale(scaler, X_train, X_test)

display(X_train_fe, X_test_fe)






#%%
# LASSO regression
X_train_fe = sm.add_constant(X_train_fe)
X_test_fe = sm.add_constant(X_test_fe)

# LASSO with cross-validation to choose best alpha
lasso_model = LassoCV(cv=tscv, random_state=123, alphas=np.logspace(-4, 1, 50), max_iter=10000)
lasso_model.fit(X_train_fe, Y_train)

print("Best alpha:", lasso_model.alpha_)

# 제거된 변수들 확인
coef_df = pd.DataFrame({'Feature': X_train_fe.columns, 'Coefficient': lasso_model.coef_})
display(coef_df[coef_df['Coefficient'] == 0])  

# 계수확인
coef_df = pd.DataFrame({
    'Feature': X_train_fe.columns,
    'Coefficient': lasso_model.coef_
})
coef_df.loc[len(coef_df)] = ['Intercept', lasso_model.intercept_]
display(coef_df)




#%%
# 예측
Y_trpred = lasso_model.predict(X_train_fe)
Y_tepred = lasso_model.predict(X_test_fe)

print(Y_train.shape, Y_trpred.shape)

#%%
# 실제값으로 복원한 값들
Y_train_real = np.expm1(Y_train)
Y_trpred_real = np.expm1(Y_trpred)
Y_test_real = np.expm1(Y_test)
Y_tepred_real = np.expm1(Y_tepred)


# 평가
Score_real = evaluation_reg_trte(Y_train_real, Y_trpred_real, Y_test_real, Y_tepred_real)
display(Score_real)


#%%
Resid_tr = Y_train.squeeze() - Y_trpred
Resid_te = Y_test.squeeze() - Y_tepred

plt.figure(figsize=(6, 6))
stats.probplot(Resid_tr, dist="norm", plot=plt)
plt.title("QQ Plot of Residuals")
plt.grid(True)
save_path = os.path.join(image_save_dir, 'LASSO_TrainResidQQPlot.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()


sns.scatterplot(x=Y_trpred, y=Resid_tr)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Train Residual Plot")
save_path = os.path.join(image_save_dir, 'RF_TrainLogResidPlot.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()


sns.scatterplot(x=Y_tepred, y=Resid_te)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Test Residual Plot")
save_path = os.path.join(image_save_dir, 'RF_TestLogResidPlot.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()

Resid_tr_true = Y_train_real.squeeze() - Y_trpred_real
Resid_te_true = Y_test_real.squeeze() - Y_tepred_real

sns.scatterplot(x=Y_trpred_real, y=Resid_tr_true)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Train Residual Plot")
save_path = os.path.join(image_save_dir, 'RF_TrainResidPlot.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()


sns.scatterplot(x=Y_tepred_real, y=Resid_te_true)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Test Residual Plot")
save_path = os.path.join(image_save_dir, 'RF_TestResidPlot.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()



#%%
# LASSO가 선택한 변수만 추출
selected_features = X_train_fe.columns[lasso_model.coef_ != 0]
X_selected = X_train_fe[selected_features]

# 상수항 추가
X_selected_const = sm.add_constant(X_selected)

# OLS 모델 적합
ols_model = sm.OLS(Y_train, X_selected_const).fit()

# 요약 출력
print(ols_model.summary())




#%%
######################################
# Submission  파일생성



# 모델 저장
model_path = os.path.join(model_save_dir, 'lasso_model.pkl')
joblib.dump(lasso_model, model_path)




#%%
lasso_model = joblib.load(model_path)

test_df = pd.read_csv('../../data/processed/cleaned_data/test_clean.csv')

test_df['계약일자dt'] = pd.to_datetime(test_df['계약일자'])
test_df['계약년월dt'] = pd.to_datetime(test_df['계약년월'], format='%Y%m')
test_df['계약일자'] = test_df['계약일자dt'].dt.strftime('%Y%m%d').astype(int)
test_df['계약년월'] = test_df['계약년월dt'].dt.strftime('%Y%m').astype(int)

district_dummy = pd.get_dummies(test_df['자치구'], prefix='자치구', drop_first=True)
test_df = pd.concat([test_df, district_dummy], axis = 1)

le = LabelEncoder()
test_df['브랜드등급_labeled'] = le.fit_transform(test_df['브랜드등급'])

test_df['계약일'] = (test_df['계약일자dt'] - pd.to_datetime("2007-01-01")).dt.days 
X_col = ['계약일', '전용면적', '층', 
          '강남3구여부', '홈페이지유무', '사용허가여부',
          '연식', '아파트이름길이', '지하철최단거리', '버스최단거리', 
          '총인구수', '성비(남/여)',
          'loanrate_12m',
          '반경_1km_버스정류장_수',
          '반경_1km_지하철역_수',
          '브랜드등급_labeled'] + [col for col in test_df if col.startswith('자치구_')]

test_df = sm.add_constant(test_df)

for col in X_train_fe.columns:
    if col not in test_df.columns:
        test_df[col] = 0

test_df = test_df[X_train_fe.columns]

scaler = preprocessing.MinMaxScaler()
scaler_fit = scaler.fit(test_df)
test_df = pd.DataFrame(scaler_fit.transform(test_df), columns = test_df.columns, index= test_df.index)


#%%
test_df.info()

#%%
X_train_fe.info()


#%%
test_pred = lasso_model.predict(test_df)
test_pred = np.expm1(test_pred)

lasso_prediction = test_pred.copy()
lasso_prediction = pd.DataFrame(lasso_prediction.astype(int), columns=["target"])

print(lasso_prediction.head())

#%%
submission_path = os.path.join(prediction_save_dir, 'lasso_prediction.csv')
lasso_prediction.to_csv(submission_path, index=False)

# %%
