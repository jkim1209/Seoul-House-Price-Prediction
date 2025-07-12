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

# 마이너스 기호 깨짐 방지
mpl.rcParams['axes.unicode_minus'] = False


df = pd.read_csv('../../cleaned_data/train_row_cleaned.csv')

df['계약일자'] = pd.to_datetime(df['계약일자'])
df['계약년월'] = pd.to_datetime(df['계약년월'])
print(df.info())
display(df.columns)


df_cleaned = df.copy()


#%%
# 월별 평균 거래가 계산
monthly_mean = df.groupby('계약년월')['target'].mean().reset_index()
monthly_median = df.groupby('계약년월')['target'].median().reset_index()

# 그래프 그리기
plt.figure(figsize=(20, 10))
sns.lineplot(x='계약년월', y='target', data=monthly_mean, label='평균 집값')
sns.lineplot(x='계약년월', y='target', data=monthly_median, label='중앙값 집값')

plt.title('월별 평균 및 중앙값 집값 추이')
plt.xlabel('계약년월')
plt.ylabel('거래금액(만원)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()


#%%
# 수치형변수 간 상관관계 확인

num_cols = [
    'target', '전용면적', '층', '연식', '아파트이름길이',
    '지하철최단거리', '반경_1km_지하철역_수', '반경_500m_지하철역_수', '반경_300m_지하철역_수', 
    '버스최단거리', '반경_1km_버스정류장_수', '반경_500m_버스정류장_수', '반경_300m_버스정류장_수',
    '총인구수', '성비(남/여)',
    '과거1개월_거래수', '과거3개월_거래수', '과거6개월_거래수', '과거12개월_거래수',
    'loanrate_1m', 'loanrate_3m', 'loanrate_6m', 'loanrate_12m'
]

corr_matrix = df_cleaned[num_cols].corr()

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(20,15))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("수치형 변수 간 상관관계")
plt.show()


## 과거 거래수들간이나 loanrate간에 상관관계가 높은것은 당연합니다.
## 단순히 상관계수로만 따지자면 '과거3개월_거래수' 와 'loanrate_12m'을 선택하는 것이 좋아보이나
## 전부 넣고 나중에 회귀모델 혹은 트리모델 사용 후 변수 중요도 확인 후 가장 중요한 것 하나씩만 살리는 것이 좋을 것 같습니다.


#%%
# 범주형변수처리
cat_cols = ['자치구', '법정동', '강남3구여부', '홈페이지유무', '사용허가여부', '브랜드등급']

#%%
## 자치구별 총 거래량 확인
dist_df = df['자치구'].value_counts().reset_index()
dist_df.columns = ['자치구', '거래건수']
display(dist_df)

plt.figure(figsize=(20, 10))
sns.barplot(data=dist_df, x='자치구', y='거래건수', palette='Set2')
plt.title('자치구별 총 거래건수')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()


#%%
# 자치구와 거래가격분포의 상관관계 확인
# 자치구 원핫인코딩
dummy_df = pd.get_dummies(df['자치구'], prefix='자치구')
df = pd.concat([df, dummy_df], axis=1)

## 자치구와 거래가격분포의 상관관계 확인
dist = [col for col in df.columns if col.startswith('자치구_')]
correlations = df[dist + ['target']].corr()['target'].drop('target')
correlations.sort_values(ascending=False)
correlations.sort_values().plot(kind='barh', figsize=(10,6))
plt.title("자치구와 거래금액간의 상관관계")
plt.xlabel("상관계수")
plt.grid(True)
plt.show()


#%%
# 브랜드등급 레이블인코딩 및 브랜드등급별 거래가격분포 확인
brand_df = df['브랜드등급'].value_counts().reset_index()
brand_df.columns = ['브랜드등급', '거래건수']
display(brand_df)

brand= {'하이엔드': 2, '프리미엄': 1, '기타': 0}

df['브랜드등급인코딩'] = df['브랜드등급'].map(brand).fillna(0).astype(int)

plt.figure(figsize=(6,4))
sns.boxplot(data=df, x='브랜드등급', y='target', order=['기타', '프리미엄', '하이엔드'])
plt.title("브랜드등급별 거래가격 분포")
plt.ylabel("거래가격") 
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(data=df, x='브랜드등급', y=np.log1p(df['target']), order=['기타', '프리미엄', '하이엔드'])
plt.title("브랜드등급별 log(거래가격) 분포")
plt.ylabel("log(거래가격)") 
plt.grid(axis='y')
plt.show()


#%%
# 강남3구여부에 따른 거래가격분포 확인
brand_df = df['강남3구여부'].value_counts().reset_index()
brand_df.columns = ['강남3구여부', '거래건수']
display(brand_df)

plt.figure(figsize=(6,4))
sns.boxplot(data=df, x='강남3구여부', y='target')
plt.title('강남3구 여부에 따른 거래금액 분포')
plt.xlabel('강남3구 여부')
plt.ylabel('거래가격')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(data=df, x='강남3구여부', y=np.log1p(df['target']))
plt.title('강남3구 여부에 따른 log(거래금액) 분포')
plt.xlabel('강남3구 여부')
plt.ylabel('log(거래가격)')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

#%%
# 홈페이지유무에 따른 거래가격분포 확인
brand_df = df['홈페이지유무'].value_counts().reset_index()
brand_df.columns = ['홈페이지유무', '거래건수']
display(brand_df)

plt.figure(figsize=(6,4))
sns.boxplot(data=df, x='홈페이지유무', y='target')
plt.title('홈페이지유무에 따른 거래금액 분포')
plt.xlabel('홈페이지유무')
plt.ylabel('거래가격')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(data=df, x='홈페이지유무', y=np.log1p(df['target']))
plt.title('홈페이지유무에 따른 log(거래금액) 분포')
plt.xlabel('홈페이지유무')
plt.ylabel('log(거래가격)')
plt.grid(axis='y')
plt.tight_layout()
plt.show()


#%%
# 사용허가여부에 따른 거래가격분포 확인
brand_df = df['사용허가여부'].value_counts().reset_index()
brand_df.columns = ['사용허가여부', '거래건수']
display(brand_df)

plt.figure(figsize=(6,4))
sns.boxplot(data=df, x='사용허가여부', y='target')
plt.title('사용허가여부에 따른 거래금액 분포')
plt.xlabel('사용허가여부')
plt.ylabel('거래가격')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(data=df, x='사용허가여부', y=np.log1p(df['target']))
plt.title('사용허가여부에 따른 log(거래금액) 분포')
plt.xlabel('사용허가여부')
plt.ylabel('log(거래가격)')
plt.grid(axis='y')
plt.tight_layout()
plt.show()


#%%
# 자치구별 시기별 거래건수
plt.figure(figsize=(18, 10))
sns.lineplot(data=df.groupby(['계약년월', '자치구']).size().reset_index(name='거래건수'), x='계약년월', y='거래건수', hue='자치구', palette='Set2')
plt.title("자치구별 월별 거래건수")
plt.xlabel("계약년월")
plt.ylabel("거래건수")
plt.xticks(rotation=45)
plt.legend(title='자치구', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True)
plt.show()

plt.figure(figsize=(18, 10))
sns.lineplot(data=df.groupby(['계약년도', '자치구']).size().reset_index(name='거래건수'), x='계약년도', y='거래건수', hue='자치구', palette='Set2')
plt.title("자치구별 년도별 거래건수")
plt.xlabel("계약년도")
plt.ylabel("거래건수")
plt.xticks(rotation=45)
plt.legend(title='자치구', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True)
plt.show()

#%%
# 자치구별 시기별 거래액수 평균
plt.figure(figsize=(18, 10))
sns.lineplot(data=df.groupby(['계약년월', '자치구'])['target'].mean().reset_index(), x='계약년월', y='target', hue='자치구', palette='Set2')
plt.title("자치구별 월별 거래가격 평균")
plt.xlabel("계약년월")
plt.ylabel("거래가격(만원)")
plt.xticks(rotation=45)
plt.legend(title='자치구', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True)
plt.show()

plt.figure(figsize=(18, 10))
sns.lineplot(data=df.groupby(['계약년도', '자치구'])['target'].mean().reset_index(), x='계약년도', y='target', hue='자치구', palette='Set2')
plt.title("자치구별 년도별 거래가격 평균")
plt.xlabel("계약년도")
plt.ylabel("거래가격(만원)")
plt.xticks(rotation=45)
plt.legend(title='자치구', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True)
plt.show()

#%%
# 자치구별 시기별 거래액수 중앙값
plt.figure(figsize=(18, 10))
sns.lineplot(data=df.groupby(['계약년월', '자치구'])['target'].median().reset_index(), x='계약년월', y='target', hue='자치구', palette='Set2')
plt.title("자치구별 월별 거래가격 중앙값")
plt.xlabel("계약년월")
plt.ylabel("거래가격(만원)")
plt.xticks(rotation=45)
plt.legend(title='자치구', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True)
plt.show()

plt.figure(figsize=(18, 10))
sns.lineplot(data=df.groupby(['계약년도', '자치구'])['target'].median().reset_index(), x='계약년도', y='target', hue='자치구', palette='Set2')
plt.title("자치구별 년도별 거래가격 중앙값")
plt.xlabel("계약년도")
plt.ylabel("거래가격(만원)")
plt.xticks(rotation=45)
plt.legend(title='자치구', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True)
plt.show()

#%%
# 자치구별 시기별 거래액수 최소값
plt.figure(figsize=(18, 10))
sns.lineplot(data=df.groupby(['계약년월', '자치구'])['target'].min().reset_index(), x='계약년월', y='target', hue='자치구', palette='Set2')
plt.title("자치구별 월별 거래가격 최소값")
plt.xlabel("계약년월")
plt.ylabel("거래가격(만원)")
plt.xticks(rotation=45)
plt.legend(title='자치구', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True)
plt.show()

plt.figure(figsize=(18, 10))
sns.lineplot(data=df.groupby(['계약년도', '자치구'])['target'].min().reset_index(), x='계약년도', y='target', hue='자치구', palette='Set2')
plt.title("자치구별 년도별 거래가격 최소값")
plt.xlabel("계약년도")
plt.ylabel("거래가격(만원)")
plt.xticks(rotation=45)
plt.legend(title='자치구', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True)
plt.show()

#%%
# 자치구별 시기별 거래액수 최대값
plt.figure(figsize=(18, 10))
sns.lineplot(data=df.groupby(['계약년월', '자치구'])['target'].max().reset_index(), x='계약년월', y='target', hue='자치구', palette='Set2')
plt.title("자치구별 월별 거래가격 최대값")
plt.xlabel("계약년월")
plt.ylabel("거래가격(만원)")
plt.xticks(rotation=45)
plt.legend(title='자치구', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True)
plt.show()

plt.figure(figsize=(18, 10))
sns.lineplot(data=df.groupby(['계약년도', '자치구'])['target'].max().reset_index(), x='계약년도', y='target', hue='자치구', palette='Set2')
plt.title("자치구별 년도별 거래가격 최대값")
plt.xlabel("계약년도")
plt.ylabel("거래가격(만원)")
plt.xticks(rotation=45)
plt.legend(title='자치구', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True)
plt.show()

