#%%
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import time
from datetime import datetime
from xgboost import XGBRegressor
import lightgbm as lgb

# set kr font
import matplotlib.font_manager
fontlist = [font.name for font in matplotlib.font_manager.fontManager.ttflist]
# print(fontlist)
krfont = ['Malgun Gothic', 'NanumGothic', 'AppleGothic']
for font in krfont:
    if font in fontlist:
        mpl.rc('font', family=font)

#%%
# train 데이터셋 불러오기

data_dir = '../../data/processed/cleaned_data'
os.makedirs(data_dir, exist_ok=True)

traindata_filename = 'train_clean.csv'
testdata_filename = 'test_clean.csv'

traindata_path = os.path.join(data_dir, traindata_filename)
testdata_path = os.path.join(data_dir, testdata_filename)

df = pd.read_csv(traindata_path)
df_test = pd.read_csv(testdata_path)
# 데이터 확인
print(df.info())
print(df_test.info())

#%%
# 범주형, 수치형 feature 분리
obj_cols = df.select_dtypes(include='object').columns.tolist()
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(obj_cols)
print(num_cols)

#%%
# 원본 df 복사하여 인코딩 처리할 df 생성
df_encoded = df.copy()
df_test_encoded = df_test.copy()

#%%
# OneHotEncoding => 지역구, 브랜드등급을 인코딩 하려 했으나 변수중요도 측정이 잘못되는 것 같아 지역구는 drop
def one_hot_encoding_with_dataset(df_train_org, df_test_org, encoding_cols):
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

    # train 데이터 fit
    encoder.fit(df_train_org[encoding_cols])

    # transform train/test
    train_ohe = encoder.transform(df_train_org[encoding_cols])
    test_ohe = encoder.transform(df_test_org[encoding_cols])

    # 결과는 numpy 배열이므로, 컬럼명 붙이기 (get_feature_names_out)
    ohe_columns = encoder.get_feature_names_out(encoding_cols)

    train_ohe_df = pd.DataFrame(train_ohe, columns=ohe_columns, index=df_train_org.index)
    test_ohe_df = pd.DataFrame(test_ohe, columns=ohe_columns, index=df_test_org.index)

    # 기존 숫자형 피처랑 합치기 예시
    train_final = pd.concat([df_train_org.drop(columns=encoding_cols), train_ohe_df], axis=1)
    test_final = pd.concat([df_test_org.drop(columns=encoding_cols), test_ohe_df], axis=1)

    return train_final, test_final

#%%
# # OneHotEncoding 적용 (브랜드등급)
df_encoded, df_test_encoded = one_hot_encoding_with_dataset(df_encoded, df_test_encoded, ['브랜드등급'])

#%%
# print(df_encoded.info())
# print(df_test_encoded.info())

#%%
# 로그 변환 (log1p) 적용
df_encoded["log_target"] = np.log1p(df["target"])


#%%
# 피처 선택
selected_features = [
    '계약년도',
    '계약월',
    '강남3구여부',
    '전용면적',
    '층',
    '건축년도',
    '홈페이지유무',
    '사용허가여부',
    '아파트이름길이',
    '좌표X',
    '좌표Y',
    '지하철최단거리',
    '반경_500m_지하철역_수',
    '버스최단거리',
    '반경_500m_버스정류장_수',
    '브랜드등급_기타',
    '브랜드등급_프리미엄',
    '브랜드등급_하이엔드'
]

# print(selected_features)

#%%
# df_encoded.columns

#%%

# 수치형 변수 상관관계
selected_num_cols = df_encoded[selected_features].select_dtypes(include=['int64', 'float64']).columns.tolist()
print(selected_num_cols)

df_num = df_encoded[selected_num_cols]
corr_matrix = df_num.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Heatmap of Selected Columns')
plt.show()

#%%
seed = 42

#%%
df_modeling = df_encoded.copy()

# train 데이터셋 feature, target 분리
X = df_modeling[selected_features]
y = df_modeling['log_target']

# 테스트 데이터셋
X_test = df_test_encoded[selected_features]

#%%

# 데이터셋
print(X.info())
print(y.info())
print(X_test.info())

#%%

def get_next_dir_number(base_path="results"):
    os.makedirs(base_path, exist_ok=True)
    print(f'test :: base path > {base_path}')
    existing = [d for d in os.listdir(base_path) if d.startswith("modeling_")]
    print(f'existing : {existing}')
    numbers = []
    for name in existing:
        try:
            number = int(name.split("_")[1])
            numbers.append(number)
        except:
            pass
    next_number = max(numbers, default=0) + 1

    return f'{base_path}/modeling_{next_number:03d}'

#%%
def show_feature_importances(model, name, selected_features):
    importances = pd.Series(model.feature_importances_, index=selected_features)
    importances_sorted = importances.sort_values(ascending=False)[:20]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances_sorted.values, y=importances_sorted.index)
    plt.title(f'{name} - Feature Importances (Top 20)')
    plt.tight_layout()
    plt.show()

#%%
def model_training(is_submission,
                   model, # 학습할 모델
                   model_name, # 학습할 모델명
                   selected_features, # 학습에 사용한 features
                   X, # 학습 데이터셋 (features) (함수 내부에서 분할)
                   y, # 학습 데이터 정답(target) (함수 내부에서 분할)
                   X_test=None, # 제출용 test 데이터셋 (features)
                   show_feature_importance = False): # 변수 중요도 시각화 표시여부
    
    basepath_dir = f"../../../output"
    output_dir = get_next_dir_number(basepath_dir)
    os.makedirs(output_dir, exist_ok=True)  # 경로 없으면 생성
    print(f"🔹 modeling directory: {output_dir}")

    features_filename = 'selected_features.txt'
    features_filepath = os.path.join(output_dir, features_filename)
    
    with open(features_filepath, 'w', encoding='utf-8') as f:
        for feature in selected_features:
            f.write(feature + '\n')
        print(f"🔹 selected_features 파일 저장: {features_filepath}")

    print(f'🔹 Training {model_name}...')
    start_time = time.time()

    # k-fold 사용하여 학습용 데이터셋 나누기
    fold_count = 5
    kf = KFold(n_splits=fold_count, shuffle=True, random_state=seed)

    # 각 학습결과 저장, 평균값 저장할 예정
    rmse_scores = []
    submission_df = pd.DataFrame()

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f" - Fold {fold_idx+1} start")

        # 예측값 시각화를 위한 정보 저장
        result_bundle = {
            "model": model,
        }

        if is_submission:
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]

            # 학습 전체 데이터 train 으로 사용, test.csv 를 test 로 사용
            model.fit(X_train, y_train)
            y_pred_log = model.predict(X_test)
            y_pred = np.expm1(y_pred_log)
            y_pred = np.round(y_pred).astype(int)  # 정수 변환

            # 제출용 DataFrame 생성
            submission_df[f'fold_{fold_idx+1}'] = y_pred

            result_bundle['y_pred'] = y_pred
        else:
            # 학습 데이터를 나눠서 train, test 로 사용
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            if model_name in 'XGB':
                model.fit(
                    X_train, 
                    y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=100,
                    verbose=100
                )
            elif model_name in 'light':
                model.fit(
                    X_train, 
                    y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric="rmse",
                    callbacks=[
                        model.early_stopping(100)
                    ]
                )
            else:
                model.fit(X_train, y_train)

            y_pred_log = model.predict(X_val)
            y_pred = np.expm1(y_pred_log)  # 예측 복원
            y_true = np.expm1(y_val)      # 정답 복원

            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            rmse_scores.append(rmse)

            print(f'✅ {model_name} fold {fold_idx} RMSE: {rmse:.2f}')

            result_bundle['y_pred'] = y_pred
            result_bundle['y_true'] = y_true
            result_bundle['rmse'] = rmse
            result_bundle['X_val'] = X_val

            # 변수 중요도 시각화 (가능한 경우만)
            if hasattr(model, 'feature_importances_'):
                show_feature_importances(model, model_name, selected_features)

        # 하나의 k-fold 모델 학습 종료
        mode_str = 's' if is_submission else 't'
        output_filename = f"model_{model_name}_{fold_idx+1}_output_{mode_str}_yj.pkl"
        output_filepath = os.path.join(output_dir, output_filename)

        joblib.dump(result_bundle, output_filepath)
        print(f"{output_filename} 파일 저장: {output_filepath}")

    # 전체 k-fold 학습 종료
    end_time = time.time()
    print(f"학습 시간: {end_time - start_time:.2f}초")

    if is_submission:
        submission_df['target'] = submission_df[[f'fold_{i+1}' for i in range(kf.n_splits)]].mean(axis=1)
        final_submission_df = submission_df['target']

        output_filename = f"submission_{model_name}_yj.csv"
        output_filepath = os.path.join(output_dir, output_filename)

        # CSV 파일로 저장 (index=False로 인덱스 제외)
        final_submission_df.to_csv(output_filepath, index=False)
    else:
        rmse_avg = np.mean(rmse_scores)
    
    if is_submission:
        return 0
    else:
        return rmse_avg

# %%
xgb = XGBRegressor(
    n_estimators=3000,
    learning_rate=0.01,
    max_depth=12,
    min_child_weight=7,
    subsample=0.6,
    colsample_bytree=0.8,
    gamma=0, 
    reg_alpha=1,
    reg_lambda=1,
    random_state=seed,
    n_jobs=-1
)

model = xgb
model_name = 'XGBoost'

# # 학습용
results = model_training(is_submission=False, 
                         model=model,
                         model_name=model_name, 
                         selected_features=selected_features, 
                         X=X,
                         y=y,
                         show_feature_importance=True)

# # 제출용
# results = model_training(is_submission=True, 
#                          model=model,
#                          model_name=model_name, 
#                          selected_features=selected_features, 
#                          X=X,
#                          y=y, 
#                          X_test=X_test)

print(results)

# %%
