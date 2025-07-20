
# ============ feature engineering & selection ========================

#%%
from urllib.request import urlopen
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from deepctr_torch.models import AutoInt
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
import torch

# 한글 폰트 설정
import matplotlib.font_manager
fontlist = [font.name for font in matplotlib.font_manager.fontManager.ttflist]
krfont = ['Malgun Gothic', 'NanumGothic', 'AppleGothic', 'Apple SD Gothic Neo'] # Apple SD Gothic Neo 추가 (맥 유저라서)
for font in krfont:
    if font in fontlist:
        mpl.rc('font', family=font)
        break # 찾으면 반복 중지

# 마이너스 기호 깨짐 방지
mpl.rcParams['axes.unicode_minus'] = False

# %%
# 데이터 복사
train_df = pd.read_csv('../../data/processed/cleaned_data/train_clean.csv')
test_df = pd.read_csv('../../data/processed/cleaned_data/test_clean.csv')

concatdf = pd.concat([train_df, test_df], ignore_index=True)
df = concatdf.copy()

# %%
# 1. 금리데이터 drop
df.drop(columns=['loanrate_1m', 'loanrate_3m', 'loanrate_6m', 'loanrate_12m'], inplace = True)
print(df.shape) #(1107114, 27)



# %%
# 2. 시간 feature 생성 및 drop
df['계약일자'] = pd.to_datetime(df['계약일자'], format='%Y-%m-%d')
df['계약요일'] = df['계약일자'].dt.dayofweek

df['계약월'] = df['계약년월'] % 100 
df['계약월_sin'] = np.sin(2 * np.pi * df['계약월'] / 12)
df['계약월_cos'] = np.cos(2 * np.pi * df['계약월'] / 12)

df['계약주차'] = df['계약일자'].dt.isocalendar().week.astype(int)
df['계약분기'] = df['계약일자'].dt.quarter

df['월말여부'] = (df['계약일자'].dt.is_month_end).astype(int)
df['월초여부'] = (df['계약일자'].dt.is_month_start).astype(int)

train_df['계약일자'] = pd.to_datetime(train_df['계약일자'], format='%Y-%m-%d') # train_df도 변환!
min_date = train_df['계약일자'].min() #200701

# test데이터의 시작 시점에 가까울수록 값이 커지(작아)도록 설계하여 최근 데이터에 더 가중치를 두기
df['계약일로부터_경과일수'] = (df['계약일자'] - min_date).dt.days

df.drop(columns=['계약월','계약일자'],inplace=True)
print(df.shape) #(1107114, 33)

#print(df.head())
# %%
# 3. 
monthly_counts_gu = df.groupby(['자치구', '계약년월']).size().reset_index(name='월별_거래량')
monthly_counts_dong = df.groupby(['법정동', '계약년월']).size().reset_index(name='월별_거래량')

df = pd.merge(df, monthly_counts_gu, on=['자치구', '계약년월'], how='left', suffixes=('', '_구'))
df = pd.merge(df, monthly_counts_dong, on=['법정동', '계약년월'], how='left', suffixes=('', '_동'))

df['구_월별_거래량_lag1'] = df.groupby('자치구')['월별_거래량'].shift(1)
df['동_월별_거래량_lag1'] = df.groupby('법정동')['월별_거래량_동'].shift(1) # suffixes로 인해 컬럼명 변경됨

df['구_누적거래량'] = df.groupby('자치구').cumcount() + 1 # 각 그룹 내 누적 카운트
df['동_누적거래량'] = df.groupby('법정동').cumcount() + 1

df['구_3개월평균거래량'] = df.groupby('자치구')['월별_거래량'].transform(lambda x: x.rolling(window=3, min_periods=1).mean().shift(1))
df['동_3개월평균거래량'] = df.groupby('법정동')['월별_거래량_동'].transform(lambda x: x.rolling(window=3, min_periods=1).mean().shift(1))


print(df.shape) #(1107114, 41)

df.drop(columns=['자치구','법정동'],inplace=True)
print(df.shape) #(1107114, 39)

# %%
#df.head()
#잠시 결측치 확인
df = df.sort_values(by='계약년월')

df.isnull().sum() # shift(1) 한 위의 4컬럼에서 336, 25개의 결측치 존재 -> 나중에 잘라낼것 

# %%
# 5. 데이터 cutting
#df['계약년월'] = df['계약년월'].astype(int)
#print(df['계약년월'])

df= df[df['계약년월'] >= 201310] #2013년 10월 이후로 튄다고 생각하여 자르기로 함
print(df.shape)  


# %%
# 6. 지하철, 버스는 bin으로 묶기

df['반경_1km_지하철역_수'] = pd.cut(df['반경_1km_지하철역_수'], bins=5, labels=False, include_lowest=True).astype('int64')
df['반경_500m_지하철역_수'] = pd.cut(df['반경_500m_지하철역_수'], bins=5, labels=False, include_lowest=True).astype('int64')
df['반경_300m_지하철역_수'] = pd.cut(df['반경_300m_지하철역_수'], bins=5, labels=False, include_lowest=True).astype('int64')

df['반경_1km_버스정류장_수'] = pd.cut(df['반경_1km_버스정류장_수'], bins=5, labels=False, include_lowest=True).astype('int64')
df['반경_500m_버스정류장_수'] = pd.cut(df['반경_500m_버스정류장_수'], bins=5, labels=False, include_lowest=True).astype('int64')
df['반경_300m_버스정류장_수'] = pd.cut(df['반경_300m_버스정류장_수'], bins=5, labels=False, include_lowest=True).astype('int64')

# %%
print("--- 연식 Binning 전 분포 확인 ---")
print(df['연식'].describe()) # 연식의 min, max, 분위수 확인
print(df['연식'].value_counts().sort_index().head(10)) # 값별 개수 확인 (초기 연식 값 위주로)
#mean: 17.465267 / std: 9.544896 / min: 0.000000 / max: 62.000000

custom_bins = [0, 5, 15, 30, df['연식'].max() + 1] 
custom_labels = ['0-5년', '5-15년', '15-30년', '30년이상'] # labels는 bins보다 하나 적어야 합니다.

df['연식_bin'] = pd.cut(df['연식'],
                        bins=custom_bins,
                        labels=custom_labels,
                        right=True, # 5년 이하, 15년 이하, 30년 이하로 포함
                        include_lowest=True # 가장 낮은 값(0년 초과) 포함
                       ).astype('object') # 'object' 대신 'category'로 변환하는 것이 메모리 효율적

print("\n--- 연식 Binning 후 분포 확인 (직접 레이블 지정) ---")
print(df['연식_bin'].value_counts().sort_index())
print(f"Dtype: {df['연식_bin'].dtype}")
#0-5년: 80614 / 5-15년: 252697 / 15-30년: 331148 / 30년이상: 75638
# %%
df.drop(columns=['연식'], inplace=True)
# %%
df.info()
# Data: (total 740097 entries,38 columns)

# %%
#이상치 검토
"""
numeric_cols_for_iqr = [
    '전용면적', '층', '연식', '아파트이름길이',
    '지하철최단거리', '버스최단거리', '총인구수', '성비(남/여)',
     '계약일로부터_경과일수',
    '월별_거래량', '월별_거래량_동', '구_월별_거래량_lag1', '동_월별_거래량_lag1',
    '구_누적거래량', '동_누적거래량', '구_3개월평균거래량', '동_3개월평균거래량'
]

print("--- IQR 기반 이상치 분석 결과 ---")
print(f"전체 샘플 수: {df.shape[0]}\n")

outlier_summary = []

for col in numeric_cols_for_iqr:
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Count outliers
        # Ensure we only count for the non-NaN values if any
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        num_outliers = outliers.shape[0]

        # Calculate percentage of outliers, only for non-null values
        num_non_null = df[col].count()
        outlier_percentage = (num_outliers / num_non_null * 100) if num_non_null > 0 else 0

        outlier_summary.append({
            'Feature': col,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'Lower Bound': lower_bound,
            'Upper Bound': upper_bound,
            'Num Outliers': num_outliers,
            'Outlier Percentage': f"{outlier_percentage:.2f}%"
        })
    else:
        print(f"Skipping '{col}' - not found or not numeric.")

# Convert summary to DataFrame for better readability
outlier_df = pd.DataFrame(outlier_summary)
outlier_df = outlier_df.sort_values(by='Num Outliers', ascending=False) # Sort by number of outliers

# Display columns with severe outliers (e.g., > 1% or > 5%)
print("\n--- 이상치가 너무 심하다고 판단되는 피처들 (비율 높은 순) ---")
# You can adjust this threshold (e.g., 1.0 for 1%, 5.0 for 5%)
threshold_percentage = 1.0 # Adjust this to define "too severe"

severe_outlier_features = outlier_df[outlier_df['Outlier Percentage'].str.replace('%','').astype(float) > threshold_percentage]

if not severe_outlier_features.empty:
    print(severe_outlier_features.to_string(index=False))
else:
    print(f"IQR 기준 {threshold_percentage}%를 초과하는 심한 이상치를 가진 피처가 없습니다.")

print("\n--- 모든 Numeric Features의 IQR 이상치 분석 결과 ---")
print(outlier_df.to_string(index=False)) # Display all results
"""

# %%
# 이상치가 많이 나온 아래 컬럼들 np.log1p() 처리

outlier_cols = ['target', '전용면적', 
    '동_3개월평균거래량', '월별_거래량_동', '동_월별_거래량_lag1', '동_누적거래량',
    '구_3개월평균거래량', '월별_거래량', '구_월별_거래량_lag1', '구_누적거래량'
    '지하철최단거리', '버스최단거리']

for col in outlier_cols:
    if col in df.columns:
        df[col] = np.log1p(df[col])


df.head(10)

#  %%
all_columns= df.columns.tolist()
print(all_columns)

final_columns = ['계약년월', '계약년도', '계약요일',
        '계약월_sin', '계약월_cos', '계약주차', '계약분기', 
        '계약일로부터_경과일수', 
        '월말여부', '월초여부', 
        '월별_거래량', '월별_거래량_동', '구_월별_거래량_lag1', '동_월별_거래량_lag1',
        '구_누적거래량', '동_누적거래량', '구_3개월평균거래량', '동_3개월평균거래량',
        '강남3구여부', '홈페이지유무', '사용허가여부', 
        '전용면적', '층', '연식_bin',
        '브랜드등급', '아파트이름길이', 
        '지하철최단거리', '반경_1km_지하철역_수', '반경_500m_지하철역_수', '반경_300m_지하철역_수', 
        '버스최단거리', '반경_1km_버스정류장_수', '반경_500m_버스정류장_수', '반경_300m_버스정류장_수', 
        '총인구수', '성비(남/여)',
        'target', 'isTest',]

final_df = df[final_columns]

final_df.info()

# %%
final_df.head(30)
"""
Int64Index: 740097 entries, 821958 to 1107113
Data columns (total 38 columns):
 #   Column           Non-Null Count   Dtype   
---  ------           --------------   -----   
 0   계약년월             740097 non-null  int64   
 1   계약년도             740097 non-null  int64   
 2   계약요일             740097 non-null  int64   
 3   계약월_sin          740097 non-null  float64 
 4   계약월_cos          740097 non-null  float64 
 5   계약주차             740097 non-null  int64   
 6   계약분기             740097 non-null  int64   
 7   계약일로부터_경과일수      740097 non-null  int64   
 8   월말여부             740097 non-null  int64   
 9   월초여부             740097 non-null  int64   
 10  월별_거래량           740097 non-null  float64 
 11  월별_거래량_동         740097 non-null  float64 
 12  구_월별_거래량_lag1    740072 non-null  float64 
 13  동_월별_거래량_lag1    739773 non-null  float64 
 14  구_누적거래량          740097 non-null  int64   
 15  동_누적거래량          740097 non-null  float64 
 16  구_3개월평균거래량       740072 non-null  float64 
 17  동_3개월평균거래량       739773 non-null  float64 
 18  강남3구여부           740097 non-null  int64   
...
 36  target           730825 non-null  float64 
 37  isTest           740097 non-null  int64   
dtypes: category(1), float64(14), int64(22), object(1)
"""

# %%
#=====================데이터 저장===============================================
#final_df.to_csv('/data/ephemeral/home/aibootcamp14/upstageailab-ml-competition-ml-5/data/processed/engineered_data/FE_concated_data.csv')

final_df.to_csv('../../data/processed/engineered_data/FE_concated_data.csv', index=False)

# %%
test_subset = final_df[final_df['isTest'] == 1]
num_test_rows = test_subset.shape[0]

print(f"isTest가 1인 행의 개수: {num_test_rows}")
#isTest가 1인 행의 개수: 9272
