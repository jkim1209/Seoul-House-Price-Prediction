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


#%%
# 데이터 불러오기
train_df = pd.read_csv('../../data/processed/cleaned_data/train_clean.csv')
test_df = pd.read_csv('../../data/processed/cleaned_data/test_clean.csv')


# %%
train_df.info()
#test_df.info()
print(train_df.select_dtypes(include=np.number).columns.tolist())
#%% [markdown]
#test vs train 데이터셋
#%% 
combined_df = pd.concat(
    [train_df.assign(dataset='Train'), test_df.assign(dataset='Test')],
    ignore_index=True #인덱스가 섞이지 않게, 0부터 다시 붙여줘.
)


#%% 
#1. 수치형 분포 비교 - sns.histplot

numeric_cols = train_df.select_dtypes(include=np.number).columns.tolist()
print(numeric_cols)
""""""
'계약년월', '계약년도', '계약월', '강남3구여부',
'전용면적', '층', '홈페이지유무', '사용허가여부', 
'연식','아파트이름길이', 
'지하철최단거리', '반경_1km_지하철역_수', '반경_500m_지하철역_수', '반경_300m_지하철역_수',
'버스최단거리', '반경_1km_버스정류장_수', '반경_500m_버스정류장_수', '반경_300m_버스정류장_수', 
'총인구수', '성비(남/여)', 
'loanrate_1m', 'loanrate_3m', 'loanrate_6m', 'loanrate_12m', 
'target', 'isTest'
""""""

selected_num = [
    #'계약년월', 
    '계약년도', '계약월', 
    '전용면적', '층', '연식','아파트이름길이', 
    #'홈페이지유무', '사용허가여부','강남3구여부',
    '지하철최단거리', '반경_1km_지하철역_수', '반경_500m_지하철역_수', '반경_300m_지하철역_수',
    '버스최단거리', '반경_1km_버스정류장_수', '반경_500m_버스정류장_수', '반경_300m_버스정류장_수', 
    '총인구수', '성비(남/여)', 
    'loanrate_1m', 'loanrate_3m', 'loanrate_6m', 'loanrate_12m',
    'target']

for col in selected_num:
    plt.figure(figsize=(10,6))
    sns.histplot(data=combined_df, x=col, 
                hue ='dataset',  #색깔로 구분할 기준
                kde = True, # 부드러운 곡선 (Kernel Density Estimate) 추가
                palette='dark',
                common_norm = False) # 각 그룹을 개별로 정규화 (비율 아님, 실제 갯수 비교)
                #common_norm=False니까 train과 test의 분포를 독립적으로 보여줘. (비율이 아닌 실제 데이터 갯수)
    plt.title(f'Train vs Test: {col} 분포')
    plt.show()

# %%
#2. 범주형 변수 분포 - countplot
object_cols = train_df.select_dtypes(include=object).columns.tolist()
#print(object_cols)
#'계약일자', '자치구', '법정동', '브랜드등급'

selected_obj = ['강남3구여부','사용허가여부','홈페이지유무','브랜드등급','자치구','법정동',]
                
for col in selected_obj:
    plt.figure(figsize=(12,7))
    sns.countplot(data=combined_df, x=col,
                hue = 'dataset',
                palette = 'coolwarm')
    plt.title(f'Train vs Test: {col} 분포')
    plt.xticks(rotation=45)
    plt.show()
# %%
#print(train_df['계약년월'].unique())
#print(test_df['계약년월'].unique())

plt.figure(figsize=(58,16))
sns.boxplot(data = train_df, 
               x = '계약년월',
               y = 'target',
               palette= 'muted')
plt.title('계약년월 별 target 분포')
plt.xticks(rotation=45)
plt.show

# %% [markdown]
### 2013년 10월 부터 튄다.



