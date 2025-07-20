#%%
import sys
print(sys.executable) #/opt/conda/bin/python
print(sys.version) #3.10.13 (main, Sep 11 2023, 13:44:35) [GCC 11.2.0]

#%%
# import package needed
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import re
#import folium
#import shap
import torch
import torchmetrics

from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from deepctr_torch.models import AutoInt
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names

from torchmetrics import MeanSquaredError, MeanAbsoluteError
import torchvision

print(f"torchmetrics 버전: {torchmetrics.__version__}")
print("MeanAbsoluteError, MeanSquaredError 임포트 성공!")
print(f"PyTorch CUDA Version: {torch.version.cuda}")
print(f"torchvision CUDA Version: {torchvision.version.cuda}")

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

#%%
featured_df = pd.read_csv('../../data/processed/engineered_data/FE_concated_data.csv',)
#featured_df = featured_df.drop(columns=["Unnamed: 0"])  #<- 저장할때 빼서 안돌려도 됨.

featured_df

print(featured_df.columns)
print(featured_df.info())

# %%
#print(featured_df.isnull().sum())

fill_na_cols = ['구_월별_거래량_lag1', '동_월별_거래량_lag1', '구_3개월평균거래량', '동_3개월평균거래량']
for col in fill_na_cols:
    if col in featured_df.columns:
        featured_df[col].fillna(0, inplace=True)
        print(f"'{col}' 컬럼의 결측치를 0으로 채웠습니다. 남은 결측치: {featured_df[col].isnull().sum()}")

# %%
train = featured_df[featured_df['isTest'] == 0].copy()
test = featured_df[featured_df['isTest'] == 1].copy()

train_labels = train['target']
train_data = train.drop(columns=['target', 'isTest'])
test_data = test.drop(columns=['target', 'isTest']) # 테스트 데이터는 target이 없으므로 drop

train_labels = train_labels.astype(np.float32)

print(f"훈련 데이터 shape: {train_data.shape}")
print(f"테스트 데이터 shape: {test_data.shape}")

# 결측치 최종 확인
print("\n--- 최종 훈련 데이터 결측치 확인 ---")
print(train_data.isnull().sum().sum()) # 0 
print("\n--- 최종 테스트 데이터 결측치 확인 ---")
print(test_data.isnull().sum().sum()) # 0 




# %% [Makrdown]
# 피처 타입 정의 및 전처리 (for AutoInt)

#범주형 피처 (SparseFeat): LabelEncoder를 사용하여 각 고유값을 0부터 시작하는 정수로 변환합니다.
sparse_features = [
    '계약년월', '계약년도', '계약요일', '계약주차', '계약분기',
    '월말여부', '월초여부', '강남3구여부', '홈페이지유무', '사용허가여부',
    '연식_bin', '브랜드등급',
    '반경_1km_지하철역_수', '반경_500m_지하철역_수', '반경_300m_지하철역_수',
    '반경_1km_버스정류장_수', '반경_500m_버스정류장_수', '반경_300m_버스정류장_수'
]

# 수치형 피처 (DenseFeat): StandardScaler를 사용하여 평균 0, 표준편차 1로 스케일링합니다.
dense_features = [
    '계약월_sin', '계약월_cos', '계약일로부터_경과일수',
    '월별_거래량', '월별_거래량_동', '구_월별_거래량_lag1', '동_월별_거래량_lag1',
    '구_누적거래량', '동_누적거래량', '구_3개월평균거래량', '동_3개월평균거래량',
    '전용면적', '층', '아파트이름길이',
    '지하철최단거리', '버스최단거리',
    '총인구수', '성비(남/여)'
]


data_for_fit = pd.concat([train_data, test_data], ignore_index=True)

# Sparse Features (Label Encoding)
for feat in sparse_features:
    lbe = LabelEncoder()
    # fit_transform은 훈련 데이터와 테스트 데이터 모두에 적용해야 함
    data_for_fit[feat] = lbe.fit_transform(data_for_fit[feat])
    # LabelEncoder로 변환된 값을 다시 훈련/테스트 데이터에 적용
    train_data[feat] = lbe.transform(train_data[feat])
    test_data[feat] = lbe.transform(test_data[feat])

# Dense Features (Standard Scaling)
mms = StandardScaler()
# fit은 훈련 데이터로만 하고, transform은 훈련/테스트 모두에 적용
# 여기서는 편의상 전체 데이터를 fit_transform 하지만, 실제로는 train_data로 fit하고 train_data, test_data에 transform 하는 것이 더 일반적입니다.
# 하지만 DeepCTR은 입력 스케일링에 덜 민감하므로 전체 데이터로 fit_transform 해도 큰 문제는 없습니다.
data_for_fit[dense_features] = mms.fit_transform(data_for_fit[dense_features])
train_data[dense_features] = mms.transform(train_data[dense_features])
test_data[dense_features] = mms.transform(test_data[dense_features])




# %% 
# DeepCTR 모델에 필요한 Feature Columns 정의
feature_columns = [SparseFeat(feat, vocabulary_size=data_for_fit[feat].nunique(), embedding_dim=4)
                   for feat in sparse_features] + \
                  [DenseFeat(feat, 1) for feat in dense_features]

# 모델 입력에 사용할 피처 이름 목록 가져오기
dnn_feature_names = get_feature_names(feature_columns)
dnn_feature_names
# %% 
# 모델 입력 데이터 준비 (딕셔너리 형태)

# 훈련 데이터
train_model_input = {name: train_data[name].values for name in dnn_feature_names}
# 테스트 데이터
test_model_input = {name: test_data[name].values for name in dnn_feature_names}

print("\n--- DeepCTR 입력 데이터 준비 완료 ---")
print(f"Sparse Features: {sparse_features}")
print(f"Dense Features: {dense_features}")
print(f"Feature Columns 정의 개수: {len(feature_columns)}")
print(f"훈련 데이터 입력 딕셔너리 키: {train_model_input.keys()}")

"""
- Sparse Features: ['계약년월', '계약년도', '계약요일', '계약주차', '계약분기', '월말여부', '월초여부', '강남3구여부', '홈페이지유무', '사용허가여부', '연식_bin', '브랜드등급', '반경_1km_지하철역_수', '반경_500m_지하철역_수', '반경_300m_지하철역_수', '반경_1km_버스정류장_수', '반경_500m_버스정류장_수', '반경_300m_버스정류장_수']
- Dense Features: ['계약월_sin', '계약월_cos', '계약일로부터_경과일수', '월별_거래량', '월별_거래량_동', '구_월별_거래량_lag1', '동_월별_거래량_lag1', '구_누적거래량', '동_누적거래량', '구_3개월평균거래량', '동_3개월평균거래량', '전용면적', '층', '아파트이름길이', '지하철최단거리', '버스최단거리', '총인구수', '성비(남/여)']
- Feature Columns 정의 개수: 36

훈련 데이터 입력 딕셔너리 키: dict_keys(['계약년월', '계약년도', '계약요일', '계약주차', '계약분기', 
'월말여부', '월초여부', '강남3구여부', '홈페이지유무', '사용허가여부', 
'연식_bin', '브랜드등급', 
'반경_1km_지하철역_수', '반경_500m_지하철역_수', '반경_300m_지하철역_수', '반경_1km_버스정류장_수', '반경_500m_버스정류장_수', '반경_300m_버스정류장_수', 
'계약월_sin', '계약월_cos', '계약일로부터_경과일수', 
'월별_거래량', '월별_거래량_동', '구_월별_거래량_lag1', '동_월별_거래량_lag1', '구_누적거래량', '동_누적거래량', '구_3개월평균거래량', '동_3개월평균거래량', 
'전용면적', '층', '아파트이름길이', 
'지하철최단거리', '버스최단거리', 
'총인구수', '성비(남/여)']"""
# %% [Markdown]
# AutoInt 모델 정의
# dnn_hidden_units: DNN 레이어의 뉴런 수 (예: (256, 128, 64))
# att_layer_num: Multi-head Attention 레이어의 수
# att_head_num: Multi-head Attention의 헤드 수
# att_res: Attention Residual Connection 사용 여부
# task: 'binary' (이진 분류) 또는 'regression' (회귀)
# device: 'cpu' 또는 'cuda:0' (GPU 사용 시)

# %% 
import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA (NVIDIA GPU) 장치를 사용합니다.")
    # 특정 GPU를 지정하고 싶다면: device = torch.device("cuda:0") (0번 GPU)

else:
    # 혹시 로컬 맥 환경에서 실행할 경우를 대비하여 MPS도 한 번 더 확인 (선택 사항)
    if torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            device = torch.device("mps")
            print("MPS (Metal Performance Shaders) 장치를 사용합니다. (로컬 맥)")
        else:
            device = torch.device("cpu")
            print("MPS가 빌드되지 않아 CPU를 사용합니다.")
    else:
        device = torch.device("cpu")
        print("CUDA 및 MPS 장치를 사용할 수 없어 CPU를 사용합니다.")



# %% 
# =============================1차=============================
model = AutoInt(
    linear_feature_columns=feature_columns,  # 선형 부분에 사용될 피처 컬럼들
    dnn_feature_columns=feature_columns,     # DNN 부분에 사용될 피처 컬럼들

    dnn_hidden_units=(256, 128, 64), # DNN 레이어의 뉴런 수
    att_layer_num=2, # Attention 레이어 수
    att_head_num=2, # Attention 헤드 수
    att_res=True, # Residual Connection 사용
    task='regression', # 회귀 문제이므로 'regression'
    device='cuda' # 맥OS 사용 환경이므로 'cpu' 또는 Metal(MPS) 지원 여부에 따라 'mps'
)


# 모델 컴파일 (이 부분은 이전과 동일)
# optimizer: Adam, SGD 등
# loss: MSE, MAE 등 (회귀의 경우)
# metrics: 평가 지표
model.compile(
    optimizer="adam",
    loss="mse",
    #metrics=[ MeanAbsoluteError(), MeanSquaredError(squared=False)]
        # -> 아파트 매매가는 연속적인 값이지만, np.log1p() 변환을 하셨다고 해도 원본 target 값에 0이 있었다면 변환 후에도 0.0이 됩니다. 또한, 만약 원본 target 값 중에 1이 있었다면 np.log1p(1)은 약 0.693이 됩니다.
        # -> 딥러닝 라이브러리나 메트릭 함수 중 일부는 이러한 **특정 값(특히 0이나 0에 가까운 값)**이 많거나, 다른 연속적인 값들과 섞여 있을 때, 이를 분류 문제의 "이진" 타겟(0 또는 1)으로 착각하는 경우가 있습니다.
        # => DeepCTR 라이브러리가 torchmetrics와 같은 외부 메트릭 객체를 받을 때, 내부적으로 target 데이터의 타입을 검사하는 과정이 있습니다.
    metrics=["mae", "mse"] 
    # MeanAbsoluteError()와 MeanSquaredError(squared=False)와 같은 torchmetrics 객체 대신, 
    # DeepCTR이 자체적으로 지원하는 문자열 형태의 메트릭인 metrics=["mae", "mse"]를 사용하는 것입니다. <- DeepCTR의 내장 메트릭 핸들러는 이러한 타입 검사에서 더 유연하거나, 해당 오류를 발생시키지 않도록 구현되어 있기 때문에 문제를 우회할 수 있습니다.
)

# 모델 학습
history = model.fit(
    train_model_input,
    train_labels.values, # numpy array 형태로 전달
    batch_size=256,
    epochs=10,
    verbose=1,
    validation_split=0.2
)

print("\n--- AutoInt 모델 학습 완료 ---")

# 예측
train_pred_log = model.predict(train_model_input, batch_size=256)
test_pred_log = model.predict(test_model_input, batch_size=256)

# 예측 결과를 원래 스케일로 되돌리기 (np.expm1 사용)
train_pred = np.expm1(train_pred_log)
test_pred = np.expm1(test_pred_log)

print("\n--- 예측 결과 (일부) ---")
print("훈련 데이터 예측 (원래 스케일):")
print(train_pred[:5])
print("\n테스트 데이터 예측 (원래 스케일):")
print(test_pred[:5])
# %%
print(test_pred.shape)

# %%
# 제출
submission_df = pd.read_csv('/data/ephemeral/home/aibootcamp14/upstageailab-ml-competition-ml-5/rawdata/sample_submission.csv')
#submission_df

submission_df['target'] = np.round(test_pred.flatten()).astype(int)

# %%
print("\n--- 최종 Submission DataFrame (일부) ---")
print(submission_df.head())
print("\n--- 최종 Submission DataFrame 정보 ---")
print(submission_df.info())

# %%
# 저장 
submission_df.to_csv('/data/ephemeral/home/aibootcamp14/upstageailab-ml-competition-ml-5/data/processed/submissions/jangwon/submission_AutoInt_0717.csv', index=False)
print("\n--- submission_AutoInt.csv 파일이 성공적으로 생성되었습니다. ---")

# %%
print(history.history.keys())


# %%
import matplotlib.pyplot as plt
import numpy as np # 혹시 모르니 numpy도 임포트

# --- MSE (평균 제곱 오차) 그래프 ---
plt.figure(figsize=(10, 6))
plt.plot(history.history['mse'], label='Train MSE')      # 훈련 MSE
plt.plot(history.history['val_mse'], label='Validation MSE') # 검증 MSE
plt.title('MSE over Epochs')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.show()

# --- Loss (이 경우 MSE와 동일) 그래프 ---
# 'loss' 키도 MSE 값을 나타낼 것이므로, 위 MSE 그래프와 유사할 것입니다.
# 만약 'loss'와 'mse' 값이 미묘하게 다르다면, 두 그래프를 따로 확인하거나
# 'loss'를 기준으로 보는 것도 좋습니다.
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')      # 훈련 손실 (MSE)
plt.plot(history.history['val_mse'], label='Validation MSE') # 검증 MSE
plt.title('Loss over Epochs (or Train MSE vs Val MSE)')
plt.xlabel('Epochs')
plt.ylabel('Loss/MSE')
plt.legend()
plt.grid(True)
plt.show()

# --- MAE (평균 절대 오차) 그래프 ---
# 만약 `model.compile` 시 `metrics=["mae", "mse"]`로 설정하셨다면 'mae'도 있을 수 있습니다.
# (이전 대화에서 "mae"를 포함했었으므로)
if 'mae' in history.history.keys():
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['mae'], label='Train MAE')      # 훈련 MAE
    if 'val_mae' in history.history.keys(): # val_mae도 있다면
        plt.plot(history.history['val_mae'], label='Validation MAE') # 검증 MAE
    plt.title('MAE over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    plt.show()
# %% [Markdown]
# 훈련 MSE는 거의 0에 가깝게 떨어지는 반면, 검증 MSE는 여전히 높은 수준을 유지하며 큰 격차를 보이는 것은 전형적인 **과적합(Overfitting)**의 증상
# 104216.3198






# =============================2차=============================
from deepctr_torch.callbacks import EarlyStopping

# AutoInt 모델 정의
# 과적합 완화를 위한 dnn_dropout 및 l2_reg_dnn, l2_reg_embedding 추가/조정
model = AutoInt(
    linear_feature_columns=feature_columns,
    dnn_feature_columns=feature_columns,

    dnn_hidden_units=(256, 128, 64), # DNN 레이어의 뉴런 수
    att_layer_num=2, # Attention 레이어 수
    att_head_num=2, # Attention 헤드 수
    att_res=True, # Residual Connection 사용
    dnn_dropout=0.2,  # <--- DNN 레이어에 20% 드롭아웃 적용 (과적합 방지)
    l2_reg_dnn=1e-4,  # <--- DNN에 L2 정규화 적용 (기본값보다 약간 높게, 과적합 방지)
    l2_reg_embedding=1e-5, # <--- 임베딩 레이어에 L2 정규화 적용 (기본값보다 약간 높게, 과적합 방지)
    task='regression', # 회귀 문제이므로 'regression'
    device=device # 'cuda' 또는 확인된 device 변수 사용
)

# 모델 컴파일 (이 부분은 이전과 동일)
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae", "mse"]
)

# EarlyStopping 콜백 정의
early_stopping = EarlyStopping(
    monitor='val_mse',          # 검증 MSE를 모니터링
    patience=5,                 # 5 에포크 동안 개선이 없으면 중단
    mode='min',                 # 'val_mse'는 작을수록 좋으므로 'min'
    #restore_best_weights=True,  # 최적의 검증 성능을 보였을 때의 가중치로 모델 복원
    verbose=1                   # 학습 중단 시 메시지 출력
)

# 모델 학습
history = model.fit(
    train_model_input,
    train_labels.values, # numpy array 형태로 전달
    batch_size=256,
    epochs=100, # EarlyStopping을 사용하므로 에포크를 충분히 크게 설정
    verbose=1,
    validation_split=0.2,
    callbacks=[early_stopping] # <--- EarlyStopping 콜백 추가
)

print("\n--- AutoInt 모델 학습 완료 ---")

# 예측
train_pred_log = model.predict(train_model_input, batch_size=256)
test_pred_log = model.predict(test_model_input, batch_size=256)

# 예측 결과를 원래 스케일로 되돌리기 (np.expm1 사용)
train_pred = np.expm1(train_pred_log)
test_pred = np.expm1(test_pred_log)

print("\n--- 예측 결과 (일부) ---")
print("훈련 데이터 예측 (원래 스케일):")
print(train_pred[:5])
print("\n테스트 데이터 예측 (원래 스케일):")
print(test_pred[:5])
# %%
# 최종 RMSE 계산 (학습 완료 후 별도 계산)
from sklearn.metrics import mean_squared_error

train_rmse = np.sqrt(mean_squared_error(np.expm1(train_labels.values), train_pred))
print(f"\n훈련 데이터 RMSE (원래 스케일): {train_rmse}")

# 테스트 데이터 RMSE (제출 후 점수 확인)
# 실제 테스트 데이터의 정답이 없으므로, submission 파일로만 점수 확인 가능합니다.
# %%
submission_df = pd.read_csv('/data/ephemeral/home/aibootcamp14/upstageailab-ml-competition-ml-5/rawdata/sample_submission.csv')
submission_df['target'] = np.round(test_pred.flatten()).astype(int)

# %%
print("\n--- 최종 Submission DataFrame (일부) ---")
print(submission_df.head())
print("\n--- 최종 Submission DataFrame 정보 ---")
print(submission_df.info())


# %%
# 저장 
submission_df.to_csv('/data/ephemeral/home/aibootcamp14/upstageailab-ml-competition-ml-5/data/processed/submissions/jangwon/submission_AutoInt_0717(2).csv', index=False)
print("\n--- submission_AutoInt(2).csv 파일이 성공적으로 생성되었습니다. ---")
# 105320.2558

# %% [Makrdown]
# 그렇게 성능이 좋이지진 않았따

# %%
# --- MSE (평균 제곱 오차) 그래프 ---
plt.figure(figsize=(10, 6))
plt.plot(history.history['mse'], label='Train MSE')      # 훈련 MSE
plt.plot(history.history['val_mse'], label='Validation MSE') # 검증 MSE
plt.title('MSE over Epochs')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.show()

# --- Loss (이 경우 MSE와 동일) 그래프 ---
# 'loss' 키도 MSE 값을 나타낼 것이므로, 위 MSE 그래프와 유사할 것입니다.
# 만약 'loss'와 'mse' 값이 미묘하게 다르다면, 두 그래프를 따로 확인하거나
# 'loss'를 기준으로 보는 것도 좋습니다.
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')      # 훈련 손실 (MSE)
plt.plot(history.history['val_mse'], label='Validation MSE') # 검증 MSE
plt.title('Loss over Epochs (or Train MSE vs Val MSE)')
plt.xlabel('Epochs')
plt.ylabel('Loss/MSE')
plt.legend()
plt.grid(True)
plt.show()

# --- MAE (평균 절대 오차) 그래프 ---
# 만약 `model.compile` 시 `metrics=["mae", "mse"]`로 설정하셨다면 'mae'도 있을 수 있습니다.
# (이전 대화에서 "mae"를 포함했었으므로)
if 'mae' in history.history.keys():
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['mae'], label='Train MAE')      # 훈련 MAE
    if 'val_mae' in history.history.keys(): # val_mae도 있다면
        plt.plot(history.history['val_mae'], label='Validation MAE') # 검증 MAE
    plt.title('MAE over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    plt.show()
# %% [Markdown]
# Train MSE (파란색 선):
# 이전과 마찬가지로 에포크 0에서 시작하여 첫 몇 에포크 만에 거의 0에 가까운 매우 낮은 값으로 급락합니다. 이는 모델이 훈련 데이터를 여전히 완벽하게 학습(거의 외우다시피)하고 있음을 보여줍니다.

# Validation MSE (주황색 선):
# 가장 중요한 변화입니다. Validation MSE는 이전 그래프에서 약 0.7~1.0 사이에서 머물렀지만, 이번에는 약 0.4~0.5 수준까지 안정적으로 감소하여 유지되는 것을 볼 수 있습니다.
# 여전히 Train MSE와 Validation MSE 사이에 격차는 존재하지만, Validation MSE 자체가 크게 낮아졌다는 것은 모델이 새로운 데이터(검증 데이터)에 대해 훨씬 더 정확하게 예측하고 있다는 의미입니다. 즉, 모델의 일반화 능력이 향상된 것입니다.

#Early Stopping의 작동:그래프가 약 40~42 에포크에서 멈춘 것을 볼 수 있습니다. 이는 설정한 patience=5에 따라 val_mse가 더 이상 개선되지 않거나 미세하게 증가하는 시점에서 Early Stopping이 성공적으로 작동하여 학습을 조기에 중단했음을 의미합니다. 불필요한 학습을 막아 과적합 심화를 방지하고 학습 시간을 절약했습니다.


# =============================3차=============================

# AutoInt 모델 정의
# 과적합 완화를 위한 dnn_dropout 및 l2_reg_dnn, l2_reg_embedding 추가/조정
model = AutoInt(
    linear_feature_columns=feature_columns,
    dnn_feature_columns=feature_columns,

    dnn_hidden_units=(128, 64, 32), # DNN 레이어의 뉴런 수
    att_layer_num=2, # Attention 레이어 수
    att_head_num=2, # Attention 헤드 수
    att_res=True, # Residual Connection 사용
    dnn_dropout=0.05,  # <--- DNN 레이어에 5% 드롭아웃 적용 (과적합 방지)
    l2_reg_dnn=1e-3,  # <--- DNN에 L2 정규화 적용 (기본값보다 약간 높게, 과적합 방지)
    l2_reg_embedding=1e-4, # <--- 임베딩 레이어에 L2 정규화 적용 (기본값보다 약간 높게, 과적합 방지)
    task='regression', # 회귀 문제이므로 'regression'
    device=device # 'cuda' 또는 확인된 device 변수 사용
)

# 모델 컴파일 (이 부분은 이전과 동일)
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae", "mse"]
)

# EarlyStopping 콜백 정의
early_stopping = EarlyStopping(
    monitor='val_mse',          # 검증 MSE를 모니터링
    patience=10,                 # 5 에포크 동안 개선이 없으면 중단
    mode='min',                 # 'val_mse'는 작을수록 좋으므로 'min'
    #restore_best_weights=True,  # 최적의 검증 성능을 보였을 때의 가중치로 모델 복원
    verbose=1                   # 학습 중단 시 메시지 출력
)

# 모델 학습
history = model.fit(
    train_model_input,
    train_labels.values, # numpy array 형태로 전달
    batch_size=256,
    epochs=100, # EarlyStopping을 사용하므로 에포크를 충분히 크게 설정
    verbose=1,
    validation_split=0.2,
    callbacks=[early_stopping] # <--- EarlyStopping 콜백 추가
)

print("\n--- AutoInt 모델 학습 완료 ---")

# 예측
train_pred_log = model.predict(train_model_input, batch_size=256)
test_pred_log = model.predict(test_model_input, batch_size=256)

# 예측 결과를 원래 스케일로 되돌리기 (np.expm1 사용)
train_pred = np.expm1(train_pred_log)
test_pred = np.expm1(test_pred_log)

print("\n--- 예측 결과 (일부) ---")
print("훈련 데이터 예측 (원래 스케일):")
print(train_pred[:5])
print("\n테스트 데이터 예측 (원래 스케일):")
print(test_pred[:5])
# %%
# %%
# 최종 RMSE 계산 (학습 완료 후 별도 계산)
from sklearn.metrics import mean_squared_error

train_rmse = np.sqrt(mean_squared_error(np.expm1(train_labels.values), train_pred))
print(f"\n훈련 데이터 RMSE (원래 스케일): {train_rmse}")

# 테스트 데이터 RMSE (제출 후 점수 확인)
# 실제 테스트 데이터의 정답이 없으므로, submission 파일로만 점수 확인 가능합니다.
# %%
submission_df = pd.read_csv('/data/ephemeral/home/aibootcamp14/upstageailab-ml-competition-ml-5/rawdata/sample_submission.csv')
submission_df['target'] = np.round(test_pred.flatten()).astype(int)

# %%
print("\n--- 최종 Submission DataFrame (일부) ---")
print(submission_df.head())
print("\n--- 최종 Submission DataFrame 정보 ---")
print(submission_df.info())


# %%
# 저장 
submission_df.to_csv('/data/ephemeral/home/aibootcamp14/upstageailab-ml-competition-ml-5/data/processed/submissions/jangwon/submission_AutoInt_0717(3).csv', index=False)
print("\n--- submission_AutoInt(3).csv 파일이 성공적으로 생성되었습니다. ---")
# 점수: