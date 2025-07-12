#%%
# import package needed
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import re

# set kr font
import matplotlib.font_manager
fontlist = [font.name for font in matplotlib.font_manager.fontManager.ttflist]
# print(fontlist)
krfont = ['Malgun Gothic', 'NanumGothic', 'AppleGothic']
for font in krfont:
    if font in fontlist:
        mpl.rc('font', family=font)


df = pd.read_csv('../../cleaned_data/train_clean.csv')

print(df.info())
df_feature_cleaned = df.copy()


#%%
# X,Y 좌표가 없는 행 삭제
df = df.dropna(subset=[
    '지하철최단거리', '버스최단거리',
    '반경_1km_지하철역_수', '반경_1km_버스정류장_수',
    '반경_500m_지하철역_수', '반경_500m_버스정류장_수',
    '반경_300m_지하철역_수', '반경_300m_버스정류장_수'
])

df.info()



#%%
# 아파트명별 거래 건수 세기
apt_counts = df['아파트명'].value_counts().reset_index()
apt_counts.columns = ['아파트명', '거래건수']

# plt.figure(figsize=(14,6))
# sns.barplot(data=apt_counts, x='아파트명', y='거래건수')

# plt.title('아파트별 거래 건수')
# plt.xlabel('아파트명(거래가 많은 순으로 정렬)')
# plt.ylabel('거래건수')
# plt.xticks([]) 
# plt.tight_layout()
# plt.grid(axis='y')
# plt.show()

# 거래수가 1건인 아파트 제거 (오타, 노이즈 제거)
numtraded_threshold = 1

print(f"-- 해당 아파트에 대한 거래수가 전체 거래수의 {numtraded_threshold}건 이하이면 삭제 --")

rare_apts = apt_counts[apt_counts['거래건수'] <= numtraded_threshold]

rare_apts_list = rare_apts['아파트명'].tolist()
rare_apts_df = df[df['아파트명'].isin(rare_apts_list)]

print(f"Total rows to be deleted: {len(rare_apts_df)}")

display(rare_apts_df[['계약년도','아파트명']])

df = df[~df['아파트명'].isin(rare_apts_list)]
print(df.info())

########################### 위 코드로 대체 ################################
# # 2023년 이전 드물게 등장한 단지 제거 (오타, 노이즈 제거)
# numtraded_threshold = 0.00001

# print(f"-- 해당 아파트에 대한 거래수가 전체 거래수의 {numtraded_threshold*100}% 미만이면 삭제 --")
# rare_apts = apt_counts[apt_counts['거래건수'] <= apt_counts['거래건수'].sum() * numtraded_threshold]
# rare_apts_list = rare_apts['아파트명'].tolist()
# rare_apts_df = df[(df['아파트명'].isin(rare_apts)) & (df['계약년도'] < 2023)]

# print(f"Total rows to be deleted: {len(rare_apts_df)}")
# display(rare_apts_df)

# df = df[~df['아파트명'].isin(rare_apts_list)]
# display(df.info())
#########################################################################



#%%
# 아파트/전용면적(정수로 변환) 별 이상치거래 확인

# 판단기준:
## 같은 아파트 및 전용면적별로 각 거래일자의 거래와 앞뒤의 거래를 포함한 11개 target을 이용하여
## robust Z-score를 계산하였을 때 4.5 이상이면 이상치 거래로 판단함

from tqdm import tqdm

threshold = 5
window = 11
result_list = []

df['전용면적_floor'] = np.floor(df['전용면적'])

def robust_z_score_window(x):
    center_value = x[len(x) // 2]
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    if mad == 0:
        return 0
    return 0.6745 * (center_value - median) / mad

for apt, group in tqdm(df.groupby(['아파트명','전용면적_floor'])):
    group = group.sort_values('계약일자').copy()

    group['rolling_robust_z'] = (
        group['target']
        .rolling(window=window, center=True)
        .apply(robust_z_score_window, raw=True)
    )

    group['is_outlier_robust_target'] = group['rolling_robust_z'].abs() > threshold
    result_list.append(group)

outlier_labeled_df = pd.concat(result_list).sort_index()

print(f"이상치 개수: ", len(outlier_labeled_df[outlier_labeled_df['is_outlier_robust_target'] == True]))
display(outlier_labeled_df[outlier_labeled_df['is_outlier_robust_target'] == True])

df = outlier_labeled_df[~outlier_labeled_df['is_outlier_robust_target']]

print(df.info())



#%%
# 필요없는 column제거
del_col_list = ['전용면적_floor', 'rolling_robust_z', 'is_outlier_robust_target']
df_cleaned = df.drop(columns=del_col_list, errors='ignore').reset_index()

display(df_cleaned.info())
display(df_cleaned.head())

df_cleaned.to_csv('../../cleaned_data/train_row_cleaned.csv', index=False)
