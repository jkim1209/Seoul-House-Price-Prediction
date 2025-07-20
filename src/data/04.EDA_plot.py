#%%
# import package needed
from IPython.display import display
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

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

# 이미지 저장경로 생성
save_dir = '../../images/jangwon/Basic'
os.makedirs(save_dir, exist_ok=True)



df = pd.read_csv('../../data/processed/cleaned_data/train_row_cleaned.csv')

df['계약일자dt'] = pd.to_datetime(df['계약일자'])
df['계약년월dt'] = pd.to_datetime(df['계약년월'], format='%Y%m')
df['계약일자'] = df['계약일자dt'].dt.strftime('%Y%m%d').astype(int)
df['계약년월'] = df['계약년월dt'].dt.strftime('%Y%m').astype(int)

df.info()
display(df.columns)





#%%
# 월별 평균 거래가 계산
monthly_mean = df.groupby('계약년월dt')['target'].mean().reset_index()
monthly_median = df.groupby('계약년월dt')['target'].median().reset_index()

# 그래프 그리기
plt.figure(figsize=(20, 10))
sns.lineplot(x='계약년월dt', y='target', data=monthly_mean, label='거래가격 평균')
sns.lineplot(x='계약년월dt', y='target', data=monthly_median, label='거래가격 중앙값')

plt.title('월별 거래가격 평균 및 중앙값 추이')
plt.xlabel('계약년월')
plt.ylabel('거래금액(만원)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.legend()
save_path = os.path.join(save_dir, '월별 거래가격 평균 및 중앙값 추이.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()





#%%
# 상관관계분석
# 브랜드등급 레이블인코딩
le = preprocessing.LabelEncoder()
df['브랜드등급(labeled)'] = le.fit_transform(df['브랜드등급'])

cols = [
    'target', 
    '계약년도', '계약년월', '계약일자', 
    '강남3구여부',
    '전용면적', '층', '연식', '아파트이름길이', '홈페이지유무', '사용허가여부', '브랜드등급(labeled)',
    '지하철최단거리', '반경_1km_지하철역_수', '반경_500m_지하철역_수', '반경_300m_지하철역_수', 
    '버스최단거리', '반경_1km_버스정류장_수', '반경_500m_버스정류장_수', '반경_300m_버스정류장_수',
    '총인구수', '인구비중', '성비(남/여)',
    # '과거1개월_거래수', '과거3개월_거래수', '과거6개월_거래수', '과거12개월_거래수',
    'loanrate_1m', 'loanrate_3m', 'loanrate_6m', 'loanrate_12m',
    '좌표X', '좌표Y'
]

corr_matrix = df[cols].corr()

plt.figure(figsize=(20,15))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("변수 간 상관관계")
save_path = os.path.join(save_dir, '상관관계.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()


#%%
# 범주형변수처리
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
save_path = os.path.join(save_dir, '자치구별 총 거래건수.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
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
save_path = os.path.join(save_dir, '자치구와 거래금액간의 상관관계.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()


#%%
# 브랜드등급 레이블인코딩 및 브랜드등급별 거래가격분포 확인
brand_df = df['브랜드등급'].value_counts().reset_index()
brand_df.columns = ['브랜드등급', '거래건수']
display(brand_df)

brand= {'하이엔드': 2, '프리미엄': 1, '기타': 0}

df['브랜드등급인코딩'] = df['브랜드등급'].map(brand).fillna(0).astype(int)

# plt.figure(figsize=(6,4))
# sns.boxplot(data=df, x='브랜드등급', y='target', order=['기타', '프리미엄', '하이엔드'])
# plt.title("브랜드등급별 거래가격 분포")
# plt.ylabel("거래가격") 
# plt.grid(axis='y')
# save_path = os.path.join(save_dir, '브랜드등급별 거래가격 분포.png')
# plt.savefig(save_path, bbox_inches='tight', dpi=300)
# plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(data=df, x='브랜드등급', y=np.log1p(df['target']), order=['기타', '프리미엄', '하이엔드'])
plt.title("브랜드등급별 log(거래가격) 분포")
plt.ylabel("log(거래가격)") 
plt.grid(axis='y')
save_path = os.path.join(save_dir, '브랜드등급별 log(거래가격) 분포.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()


#%%
# 강남3구여부에 따른 거래가격분포 확인
brand_df = df['강남3구여부'].value_counts().reset_index()
brand_df.columns = ['강남3구여부', '거래건수']
display(brand_df)

# plt.figure(figsize=(6,4))
# sns.boxplot(data=df, x='강남3구여부', y='target')
# plt.title('강남3구 여부에 따른 거래금액 분포')
# plt.xlabel('강남3구 여부')
# plt.ylabel('거래가격')
# plt.grid(axis='y')
# plt.tight_layout()
# save_path = os.path.join(save_dir, '강남3구(강남, 서초, 송파) 여부에 따른 거래금액 분포.png')
# plt.savefig(save_path, bbox_inches='tight', dpi=300)
# plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(data=df, x='강남3구여부', y=np.log1p(df['target']))
plt.title('강남3구 여부에 따른 log(거래금액) 분포')
plt.xlabel('강남3구 여부')
plt.ylabel('log(거래가격)')
plt.grid(axis='y')
plt.tight_layout()
save_path = os.path.join(save_dir, '강남3구(강남, 서초, 송파) 여부에 따른 log(거래금액) 분포.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()

#%%
# 홈페이지유무에 따른 거래가격분포 확인
brand_df = df['홈페이지유무'].value_counts().reset_index()
brand_df.columns = ['홈페이지유무', '거래건수']
display(brand_df)

# plt.figure(figsize=(6,4))
# sns.boxplot(data=df, x='홈페이지유무', y='target')
# plt.title('홈페이지유무에 따른 거래금액 분포')
# plt.xlabel('홈페이지유무')
# plt.ylabel('거래가격')
# plt.grid(axis='y')
# plt.tight_layout()
# save_path = os.path.join(save_dir, '홈페이지유무에 따른 거래금액 분포.png')
# plt.savefig(save_path, bbox_inches='tight', dpi=300)
# plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(data=df, x='홈페이지유무', y=np.log1p(df['target']))
plt.title('홈페이지유무에 따른 log(거래금액) 분포')
plt.xlabel('홈페이지유무')
plt.ylabel('log(거래가격)')
plt.grid(axis='y')
plt.tight_layout()
save_path = os.path.join(save_dir, '홈페이지유무에 따른 log(거래금액) 분포.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()


#%%
# 사용허가여부에 따른 거래가격분포 확인
brand_df = df['사용허가여부'].value_counts().reset_index()
brand_df.columns = ['사용허가여부', '거래건수']
display(brand_df)

# plt.figure(figsize=(6,4))
# sns.boxplot(data=df, x='사용허가여부', y='target')
# plt.title('사용허가여부에 따른 거래금액 분포')
# plt.xlabel('사용허가여부')
# plt.ylabel('거래가격')
# plt.grid(axis='y')
# plt.tight_layout()
# save_path = os.path.join(save_dir, '사용허가여부에 따른 거래금액 분포.png')
# plt.savefig(save_path, bbox_inches='tight', dpi=300)
# plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(data=df, x='사용허가여부', y=np.log1p(df['target']))
plt.title('사용허가여부에 따른 log(거래금액) 분포')
plt.xlabel('사용허가여부')
plt.ylabel('log(거래가격)')
plt.grid(axis='y')
plt.tight_layout()
save_path = os.path.join(save_dir, '사용허가여부에 따른 log(거래금액) 분포.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()


#%%
# 자치구별 시기별 거래건수
plt.figure(figsize=(18, 10))
sns.lineplot(data=df.groupby(['계약년월dt', '자치구']).size().reset_index(name='거래건수'), x='계약년월dt', y='거래건수', hue='자치구', palette='Set2')
plt.title("자치구별 월별 거래건수")
plt.xlabel("계약년월")
plt.ylabel("거래건수")
plt.xticks(rotation=45)
plt.legend(title='자치구', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True)
save_path = os.path.join(save_dir, '자치구별 월별 거래건수.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
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
save_path = os.path.join(save_dir, '자치구별 년도별 거래건수.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()


#%%
# 자치구별 시기별 거래액수 평균
plt.figure(figsize=(18, 10))
sns.lineplot(data=df.groupby(['계약년월dt', '자치구'])['target'].mean().reset_index(), x='계약년월dt', y='target', hue='자치구', palette='Set2')
plt.title("자치구별 월별 거래가격 평균")
plt.xlabel("계약년월")
plt.ylabel("거래가격(만원)")
plt.xticks(rotation=45)
plt.legend(title='자치구', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True)
save_path = os.path.join(save_dir, '자치구별 월별 거래가격 평균.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
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
save_path = os.path.join(save_dir, '자치구별 년도별 거래가격 평균.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()


#%%
# 자치구별 시기별 거래액수 중앙값
# plt.figure(figsize=(18, 10))
# sns.lineplot(data=df.groupby(['계약년월dt', '자치구'])['target'].median().reset_index(), x='계약년월dt', y='target', hue='자치구', palette='Set2')
# plt.title("자치구별 월별 거래가격 중앙값")
# plt.xlabel("계약년월")
# plt.ylabel("거래가격(만원)")
# plt.xticks(rotation=45)
# plt.legend(title='자치구', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.grid(True)
# plt.show()

plt.figure(figsize=(18, 10))
sns.lineplot(data=df.groupby(['계약년도', '자치구'])['target'].median().reset_index(), x='계약년도', y='target', hue='자치구', palette='Set2')
plt.title("자치구별 년도별 거래가격 중앙값")
plt.xlabel("계약년도")
plt.ylabel("거래가격(만원)")
plt.xticks(rotation=45)
plt.legend(title='자치구', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True)
save_path = os.path.join(save_dir, '자치구별 년도별 거래가격 중앙값.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()


#%%
# 자치구별 시기별 거래액수 최소값
# plt.figure(figsize=(18, 10))
# sns.lineplot(data=df.groupby(['계약년월dt', '자치구'])['target'].min().reset_index(), x='계약년월dt', y='target', hue='자치구', palette='Set2')
# plt.title("자치구별 월별 거래가격 최소값")
# plt.xlabel("계약년월")
# plt.ylabel("거래가격(만원)")
# plt.xticks(rotation=45)
# plt.legend(title='자치구', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.grid(True)
# plt.show()

plt.figure(figsize=(18, 10))
sns.lineplot(data=df.groupby(['계약년도', '자치구'])['target'].min().reset_index(), x='계약년도', y='target', hue='자치구', palette='Set2')
plt.title("자치구별 년도별 거래가격 최소값")
plt.xlabel("계약년도")
plt.ylabel("거래가격(만원)")
plt.xticks(rotation=45)
plt.legend(title='자치구', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True)
save_path = os.path.join(save_dir, '자치구별 년도별 거래가격 최소값.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()


#%%
# 자치구별 시기별 거래액수 최대값
# plt.figure(figsize=(18, 10))
# sns.lineplot(data=df.groupby(['계약년월dt', '자치구'])['target'].max().reset_index(), x='계약년월dt', y='target', hue='자치구', palette='Set2')
# plt.title("자치구별 월별 거래가격 최대값")
# plt.xlabel("계약년월")
# plt.ylabel("거래가격(만원)")
# plt.xticks(rotation=45)
# plt.legend(title='자치구', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.grid(True)
# plt.show()

plt.figure(figsize=(18, 10))
sns.lineplot(data=df.groupby(['계약년도', '자치구'])['target'].max().reset_index(), x='계약년도', y='target', hue='자치구', palette='Set2')
plt.title("자치구별 년도별 거래가격 최대값")
plt.xlabel("계약년도")
plt.ylabel("거래가격(만원)")
plt.xticks(rotation=45)
plt.legend(title='자치구', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True)
save_path = os.path.join(save_dir, '자치구별 년도별 거래가격 최대값.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()

