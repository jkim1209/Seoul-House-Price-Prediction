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

# 현재 파일경로 확인
print(os.getcwd()) 

#%%
# import data
train_rawdf = pd.read_csv('../../data/raw/train.csv', dtype = {16: 'str', 17: 'str', 36: 'str'})
test_rawdf = pd.read_csv('../../data/raw/test.csv')
bus_rawdf = pd.read_csv('../../data/raw/bus_feature.csv')
subway_rawdf = pd.read_csv('../../data/raw/subway_feature.csv')
loanrate_df = pd.read_csv('../../data/raw/loanrate.csv')
population_df = pd.read_csv('../../data/raw/population.csv')



#%%
# 사용할 column 선택
train_rawdf['isTest'] = 0
test_rawdf['isTest'] = 1
rawdf = pd.concat([train_rawdf, test_rawdf])

# print(rawdf.columns)

columns = [ # 처리한 변수
        '계약년월', '계약일', '시군구', '아파트명', '전용면적(㎡)', '층', '건축년도',    
        'k-단지분류(아파트,주상복합등등)', 'k-홈페이지', '사용허가여부',

        # 처리중 변수
        '도로명', '좌표X', '좌표Y',                                         
        # '해제사유발생일',                                                                                                                                    
        # 'k-연면적',
         
        # 처리 안 된 변수 중 결측치 많은 변수
        '주차대수', 'k-전체동수', 
        'k-건설사(시공사)',
        'k-난방방식', 'k-관리방식', 'k-복도유형', 'k-전체세대수', 

        # target 및 train/test 구분
        'target', 'isTest']

df = rawdf[columns]

# column 이름 정제
df.columns = [col[2:] if col.startswith('k-') else col for col in df.columns]
df = df.rename(columns = {'전용면적(㎡)' : '전용면적',
                          '건설사(시공사)' : '건설사',
                          '단지분류(아파트,주상복합등등)' : '단지분류'})
# print(df.columns)
# df.info()

df_original = df.copy()


#%%
## 계약일자 datetime으로 변경
df['계약일자'] = df.apply(lambda row: f"{row['계약년월']:06d}{row['계약일']:02d}", axis=1)
df['계약일자'] = pd.to_datetime(df['계약일자'], format='%Y%m%d')

## 계약년도 및 계약 월 생성
df['계약년도'] = df['계약일자'].dt.year
df['계약월'] = df['계약일자'].dt.month

## 건축년도 (이미 int타입이어서 주석해제)
# df['건축년도'] = pd.to_datetime(df['건축년도'],format='%Y').dt.year





#%%
## 시군구 -> 자치구, 동으로 분리

district = []
area = []

for i in df['시군구']:
    district.append(i.split()[1])
    area.append(i.split()[2])

# print(len(district))
# print(len(area))

df['자치구'] = pd.Series(district)
df['법정동'] = pd.Series(area)

# tmp = [i for i in df['동'].unique() if i.endswith('가')]
# print(tmp)
# print(df['자치구'].unique())



#%%
# 아파트명 -> 결측치는 ''으로 처리

# print(df['아파트명'].nunique())

df['아파트명'] = [i if type(i) == str else '' for i in df['아파트명']]





#%%
# 단지분류 -> 결측치는 '기타'로 처리

# print(df['단지분류'].unique()) # 5

df['단지분류'] = [i if type(i) == str else '기타' for i in df['단지분류']]
# print(df['단지분류'].unique())





#%%
## 홈페이지 -> 0:없음/1:있음

# print(df['홈페이지'].unique())

df['홈페이지'] = [i if '@' not in str(i) and len(str(i)) > 4 and re.search('[a-zA-Z]', i) else '' for i in df['홈페이지']]
df['홈페이지유무'] = [1 if i != '' else 0 for i in df['홈페이지']]





#%%
## 사용허가여부 -> 0:허가아님 / 1: 허가

# print(df['사용허가여부'].unique())

df['사용허가여부'] = [1 if i == 'Y' else 0 for i in df['사용허가여부']]









#%%
# # 해제사유발생일이 존재하는 row 삭제??? (실제로 거래성사된 계약이 아님)
# # 집값을 올리려는 허위 계약을 막기 위해서 2021년부터 시행

# df['해제발생여부'] = df['해제사유발생일'].notna()
# df['계약일자'] = df['계약년월'] * 100 + df['계약일']
# # 아파트명 + 계약일자 조합별 해제발생여부 종류 수 계산
# variation = df.groupby(['아파트명', '계약일자'])['해제발생여부'].nunique().reset_index()
# # 해제발생여부가 0과 1 둘 다 존재하는 경우만 필터링
# conflict_cases = variation[variation['해제발생여부'] > 1]
# conflict_df = df.merge(conflict_cases[['아파트명', '계약일자']], on=['아파트명', '계약일자'], how='inner')

# print(conflict_df.groupby('해제발생여부')['target'].describe())

# sns.boxplot(x='해제발생여부', y='target', data=conflict_df)
# plt.title('해제발생여부에 따른 target 분포')
# plt.show()

# # 해제사유발생일 column을 삭제하는 방향으로?



#%%
# # 건축년도 2개 이상인 건에 대한 검사
# df['시군구 아파트명'] = df['시군구'] + ' ' + df['아파트명']
# df_group = df.groupby(['시군구 아파트명', '도로명'])['건축년도'].nunique().reset_index()
# df_group.rename(columns= {'건축년도': '건축년도 개수'}, inplace=True)
# multi_year = df_group[df_group['건축년도 개수'] > 1]
# apt_list = multi_year['시군구 아파트명'].unique()

# print(apt_list)
# # print(multi_year)

# df_merge = pd.merge(df, multi_year, how='inner', on=('시군구 아파트명', '도로명'))
# df_merge[['계약년월','시군구','아파트명','도로명','건축년도']]
# inspect_multi_year = df_merge.groupby(['시군구', '아파트명', '도로명', ])['건축년도'].agg(set).reset_index()
# inspect_multi_year



#%%
# 파생변수 추가

## 연식
df['연식'] = df['계약년도'] - df['건축년도']
# df[df['연식'] < 0]
df['연식'] = df['연식'].clip(lower=0)



#%%
## 아파트 이름 길이
df['아파트이름길이'] = [len(i) for i in df['아파트명']]
# print(df['아파트이름길이'].describe())




#%%
# 외부데이터 추가

## 인구수 데이터 추가
population_pivot_df = population_df.pivot(index=['year', 'area'], columns='class', values='population').reset_index()
# 성비
population_pivot_df['성비(남/여)'] = round(population_pivot_df['남자인구수'] / population_pivot_df['여자인구수'], 4)
# print(population_pivot_df.head())

df = pd.merge(df, population_pivot_df, how = 'left', left_on=('계약년도', '자치구'), right_on=('year', 'area'))



#%%
## 대출금리 데이터 추가
# 직전 1개월, 직전 3개월 이동평균, 직전 6개월 이동평균, 직전 1년 이동평균 주택담보대출금리
loanrate_df['loanrate_1m'] = loanrate_df['loanrate'].shift(1)
loanrate_df['loanrate_3m'] = round(loanrate_df['loanrate'].shift(1).rolling(window=3).mean(), 2)
loanrate_df['loanrate_6m'] = round(loanrate_df['loanrate'].shift(1).rolling(window=6).mean(), 2)
loanrate_df['loanrate_12m'] = round(loanrate_df['loanrate'].shift(1).rolling(window=12).mean(), 2)

# print(loanrate_df.head())

df = pd.merge(df, loanrate_df, how = 'left', left_on = '계약년월', right_on = 'month')
# print(df.head())





#%%
# 아파트 브랜드등급 추가
highend_aptlist = ['디에이치', '아크로', '써밋', '트리마제', '르엘', '푸르지오써밋', '위브더제니스']
premium_aptlist = ['삼성', '현대', '대우', '대림', 'GS', '지에스', '포스코', '롯데', 'SK', '에스케이', '한화']
# 출처: 한국기업평판연구소 브랜드평판지수 https://brikorea.com/

df['브랜드등급'] = '기타'
df.loc[df['아파트명'].str.contains('|'.join(highend_aptlist), case=False, na=False), '브랜드등급'] = '하이엔드'
df.loc[(df['브랜드등급'] == '기타') & df['건설사'].str.contains('|'.join(premium_aptlist), case=False, na=False), '브랜드등급'] = '프리미엄'


# a = df[df['브랜드등급'] == '하이엔드'][['시군구','아파트명','단지분류','건설사','브랜드등급']]
# display(a)
# b = df[df['브랜드등급'] == '프리미엄'][['시군구','아파트명','단지분류','브랜드등급']]
# display(b)





#%%
# 강남3구여부 추가

# grouped = df.groupby(['계약년도','자치구'])['target'].median().reset_index()
# sns.lineplot(data=grouped, x='계약년도', y='target',hue='자치구', palette='Set2')
# plt.legend(loc='lower left')
# plt.show()

premium_areas = ['강남구', '서초구', '송파구']
df['강남3구여부'] = df['자치구'].isin(premium_areas).astype(int)

# df[df['강남3구여부'] == 1]





#%%
# 지하철 및 버스 정보 병합
transportation_train_df = pd.read_csv("../../data/processed/transportation-features/train_transportation_features.csv")
transportation_test_df = pd.read_csv("../../data/processed/transportation-features/test_transportation_features.csv")

transportation_df = pd.concat([transportation_train_df, transportation_test_df])
transportation_df = transportation_df[transportation_df['아파트명'].notna() & (transportation_df['아파트명'].str.strip() != '')]

# # 같은 아파트명에 대해 정보가 2개 이상인 건이 있는지 확인
# check_cols = transportation_df.columns.difference(['아파트명'])
# agg_df = transportation_df.groupby('아파트명')[check_cols].nunique()
# unique_check = (agg_df == 1).all(axis=1)
# not_unique_df = unique_check[unique_check == False].reset_index()
# print(not_unique_df)

# # transportation_df에서 해당 아파트명만 필터링
# conflict_df = transportation_df[transportation_df['아파트명'].isin(not_unique_df['아파트명'])]
# display(conflict_df)

# # conflict 예시
# display(transfortation_df[transportation['아파트명] == 'DMC아이파크'])

# 같은 아파트명인데 여러개의 고유값을 가진 아파트들에 대해서는 최빈값으로 값 통일
unique_transportation_df = transportation_df.groupby(['아파트명']).agg(lambda x: x.mode().iloc[0]).reset_index()
df = df.merge(unique_transportation_df, how='left', on=('아파트명'))


#%%
# 최근 1개월, 3개월, 6개월, 1년간 해당 아파트의 거래건수
df['계약년월'] = pd.to_datetime(df['계약년월'], format='%Y%m', errors='coerce').dt.to_period('M').dt.to_timestamp()
df = df.sort_values(['아파트명', '계약년월'])

from tqdm import tqdm
tqdm.pandas()

def count_past_transactions(group, months):
    group = group.sort_values('계약년월')
    
    results = {f'과거{m}개월_거래수': [] for m in months}
    
    for idx, row in group.iterrows():
        for m in months:
            past_start = row['계약년월'] - pd.DateOffset(months=m)
            past_end = row['계약년월']
            count = group[(group['계약년월'] >= past_start) & (group['계약년월'] < past_end)].shape[0]
            results[f'과거{m}개월_거래수'].append(count)
    
    return pd.DataFrame(results, index=group.index)


months = [1, 3, 6, 12]
monthly_counts_df = df.groupby('아파트명').progress_apply(count_past_transactions, months=months)
monthly_counts_df.reset_index(level=0, drop=True, inplace=True)

df = pd.concat([df, monthly_counts_df], axis=1)


#%%
final_columns = [
                # 날짜(계약일 관련 변수)
                '계약일자', '계약년월', '계약년도', '계약월',

                # 위치 변수
                '자치구', '법정동', 
                '강남3구여부',
                
                # 아파트 특성 변수
                '아파트명',
                '전용면적',
                '층',
                '홈페이지유무',
                '사용허가여부',
                '연식',
                '브랜드등급',
                '아파트이름길이', 

                # 지하철관련 변수
                '지하철최단거리',
                '반경_1km_지하철역_수',
                '반경_500m_지하철역_수',
                '반경_300m_지하철역_수',

                # 버스관련 변수
                '버스최단거리',
                '반경_1km_버스정류장_수',
                '반경_500m_버스정류장_수',
                '반경_300m_버스정류장_수',

                # 인구수관련 변수
                '총인구수',
                '성비(남/여)',

                # 과거 거래수 변수
                '과거1개월_거래수',
                '과거3개월_거래수',
                '과거6개월_거래수',
                '과거12개월_거래수',

                # 대출금리 변수
                'loanrate_1m', 'loanrate_3m', 'loanrate_6m', 'loanrate_12m',

                # target, train/test 구분 변수
                'target', 'isTest'
]





#%%
cleandf = df[final_columns]
train_clean = cleandf[cleandf['isTest'] == 0]
test_clean = cleandf[cleandf['isTest'] == 1]


train_clean.info()


#%%
# make data folder 'cleaned_data'
data_dir = '../../cleaned_data'
os.makedirs(data_dir, exist_ok=True)

# save 'train_clean.csv' and 'test_clean.csv'
traindata_filename = 'train_clean.csv'
testdata_filename = 'test_clean.csv'

traindata_path = os.path.join(data_dir, traindata_filename)
testdata_path = os.path.join(data_dir, testdata_filename)

train_clean.to_csv(traindata_path, index=False)
test_clean.to_csv(testdata_path, index=False)


# %%
train_clean.info()
