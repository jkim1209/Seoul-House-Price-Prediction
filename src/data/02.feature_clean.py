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
from tqdm import tqdm

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

###############################################################################################
###################### 01.data_download.py를 진행하여 나온 파일이 아닌 ############################
##################### 버스와 지하철 관련 정보가 결합된 파일로 변경하여 진행 ###########################
############################################################################################### 
# train_rawdf = pd.read_csv('../../data/raw/train.csv', dtype = {16: 'str', 17: 'str', 36: 'str'})
# test_rawdf = pd.read_csv('../../data/raw/test.csv')
# bus_rawdf = pd.read_csv('../../data/raw/bus_feature.csv')
# subway_rawdf = pd.read_csv('../../data/raw/subway_feature.csv')
train_rawdf = pd.read_csv("../../data/processed/transportation-features/train_transportation_merged.csv", dtype={16: 'str', 17: 'str'})
test_rawdf = pd.read_csv("../../data/processed/transportation-features/test_transportation_merged.csv")
loanrate_df = pd.read_csv('../../data/raw/loanrate.csv')
population_df = pd.read_csv('../../data/raw/population.csv')


#%%
# (0715추가) 위의 test_rawdf가 원본 test와 인덱스 일치하는지 확인
test_rawdf = pd.read_csv("../../data/processed/transportation-features/test_transportation_merged.csv")
test_rawdf_original = pd.read_csv('../../data/raw/test.csv')

cols = ['시군구', '번지', '본번', '부번', '아파트명', '계약년월', '계약일']

# 모든 관측치가 동일한지 확인
equal_rows = test_rawdf_original[cols].reset_index(drop=True).equals(test_rawdf[cols].reset_index(drop=True))
equal_rows
# True


#%%
# 사용할 column 선택
train_rawdf['isTest'] = 0
test_rawdf['isTest'] = 1
rawdf = pd.concat([train_rawdf, test_rawdf])

# print(rawdf.columns)

####################### 처음부터 버스 및 지하철 관련 정보 병합되어 있어 columns 아래와 같이 수정 #########################
# columns = [ # 처리한 변수
#         '계약년월', '계약일', '시군구', '아파트명', '전용면적(㎡)', '층', '건축년도',    
#         'k-단지분류(아파트,주상복합등등)', 'k-홈페이지', '사용허가여부',

#         # 처리중 변수
#         '본번', '부번', '번지', '도로명', '좌표X', '좌표Y',                                         
#         # '해제사유발생일',                                                                                                                                    
#         # 'k-연면적',
         
#         # 처리 안 된 변수 중 결측치 많은 변수
#         '주차대수', 'k-전체동수', 
#         'k-건설사(시공사)',
#         'k-난방방식', 'k-관리방식', 'k-복도유형', 'k-전체세대수', 

#         # target 및 train/test 구분
#         'target', 'isTest']
##################################################################################################################

columns = [ # 처리한 변수
        '계약년월', '계약일', '시군구', '아파트명', '전용면적(㎡)', '층', '건축년도',    
        'k-단지분류(아파트,주상복합등등)', 'k-홈페이지', '사용허가여부',

        # 처리중 변수
        '본번', '부번', '번지', '도로명', '좌표X', '좌표Y',                                         
        # '해제사유발생일',                                                                                                                                    
        # 'k-연면적',

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
print(df.columns)
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
df['아파트이름길이'] = [len(i.strip()) if pd.notnull(i) else 0 for i in df['아파트명']]
# print(df['아파트이름길이'].describe())




#%%
# 외부데이터 추가

## 인구수 데이터 추가
population_pivot_df = population_df.pivot(index=['year', 'area'], columns='class', values='population').reset_index()
# 성비
population_pivot_df['성비(남/여)'] = round(population_pivot_df['남자인구수'] / population_pivot_df['여자인구수'], 4)
# 인구비중
population_pivot_df['서울전체인구'] = population_pivot_df.groupby('year')['총인구수'].transform('sum')
population_pivot_df['인구비중'] = population_pivot_df['총인구수'] / population_pivot_df['서울전체인구']
print(population_pivot_df.head())

df = pd.merge(df, population_pivot_df, how = 'left', left_on=('계약년도', '자치구'), right_on=('year', 'area'))

# # 잘 들어갔는지 확인
# check = df.groupby(['계약년월', '자치구'])['인구비중'].nunique()
# inconsistent = check[check > 1]
# inconsistent

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
highend_aptlist = ['디에이치', '아크로', '써밋', '트리마제', '르엘', '푸르지오써밋', '위브더제니스', 
                   'PH129', '파르크한남', '나인원한남', '포제스한강', '한남더힐', '갤러리아포레']
premium_constlist = ['삼성', '현대', '대우', '대림', 'GS', '지에스', '포스코', '롯데', 'SK', '에스케이', '한화']
# 출처: 한국기업평판연구소 브랜드평판지수 https://brikorea.com/
#      https://blog.naver.com/hocho0820/223354805570

df['브랜드등급'] = '기타'
df.loc[df['아파트명'].str.contains('|'.join(highend_aptlist), case=False, na=False), '브랜드등급'] = '하이엔드'
df.loc[(df['브랜드등급'] == '기타') & df['건설사'].str.contains('|'.join(premium_constlist), case=False, na=False), '브랜드등급'] = '프리미엄'


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
##################################################################################################################
############ 최상단의 X, Y결측치 채워서 지오코딩된 업데이트된 파일로 변경함으로써 교통정보를 따로 취합할 필요 없어짐 ############
##################################################################################################################

# transportation_train_df = pd.read_csv("../../data/processed/transportation-features/train_transportation_features.csv")
# transportation_test_df = pd.read_csv("../../data/processed/transportation-features/test_transportation_features.csv")

# transportation_df = pd.concat([transportation_train_df, transportation_test_df])
# transportation_df = transportation_df[transportation_df['아파트명'].notna() & (transportation_df['아파트명'].str.strip() != '')]

# # # 같은 아파트명에 대해 정보가 2개 이상인 건이 있는지 확인
# # check_cols = transportation_df.columns.difference(['아파트명'])
# # agg_df = transportation_df.groupby('아파트명')[check_cols].nunique()
# # unique_check = (agg_df == 1).all(axis=1)
# # not_unique_df = unique_check[unique_check == False].reset_index()
# # print(not_unique_df)

# # # transportation_df에서 해당 아파트명만 필터링
# # conflict_df = transportation_df[transportation_df['아파트명'].isin(not_unique_df['아파트명'])]
# # display(conflict_df)

# # # conflict 예시
# # display(transfortation_df[transportation['아파트명] == 'DMC아이파크'])

# # 같은 아파트명인데 여러개의 고유값을 가진 아파트들에 대해서는 최빈값으로 값 통일
# unique_transportation_df = transportation_df.groupby(['아파트명']).agg(lambda x: x.mode().iloc[0]).reset_index()
# df = df.merge(unique_transportation_df, how='left', on=('아파트명'))

# # 교통정보가 결측값인 행들에 대해 본번과 부번을 이용하여 대체합니다.
# # case1. 본번과 부번이 같은 관측값들이 있다면 해당 관측값들의 unique한 교통정보 column값들의 평균치로 대체합니다.
# # case2. 본번이 같지만 부번이 1 차이나는 관측값들이 있다면 해당 관측값들의 unique한 교통정보 column값들의 평균치로 대체합니다.

# # 가령
# # 교통정보가 결측치인 행이 본번 123, 부번 12, 번지 12-34 를 갖고 있다면
# # 본번 123, 부번 12인 값들이 있는 경우 (case1) 해당 관측치들이 갖는 고유한 교통정보 값들의 평균을 이용
# # 없다면 본번 123, 부번이 10~14인 값들이 있는 경우 (case2) 해당 관측치들이 갖는 고유한 교통정보 값들의 평균을 이용

# transport_cols = [
#     '지하철최단거리',
#     '반경_1km_지하철역_수',
#     '반경_500m_지하철역_수',
#     '반경_300m_지하철역_수',
#     '버스최단거리',
#     '반경_1km_버스정류장_수',
#     '반경_500m_버스정류장_수',
#     '반경_300m_버스정류장_수'
# ]

# # 교통정보가 결측값 아닌 행 추출
# trans_nonnull_df = df.dropna(subset=transport_cols)

# # 교통정보가 결측값인 행 추출
# missing_rows = df[df[transport_cols].isna().any(axis=1)]
# # print('교통정보가 결측값인 행들에 대한 본번, 부번, 번지:')
# # display(missing_rows[['본번','부번','번지']])

# # 결측치 채우기
# from tqdm import tqdm

# for idx, row in tqdm(missing_rows.iterrows(), total=len(missing_rows)):
#     di1, di2, di3 = row['본번'], row['부번'], row['번지']
    
#     case1 = trans_nonnull_df[
#         (trans_nonnull_df['본번'] == di1) &
#         (trans_nonnull_df['부번'] == di2)
#     ][transport_cols].drop_duplicates()
#     if not case1.empty:
#         mean_vals = case1.mean()

#     else:
#         case2 = trans_nonnull_df[
#             (trans_nonnull_df['본번'] == di1) &
#             (trans_nonnull_df['부번'].between(di2 - 2, di2 + 2))
#         ][transport_cols].drop_duplicates()
#         if not case2.empty:
#             mean_vals = case2.mean()
#         else:
#             continue

#     for col in transport_cols:
#         if pd.isna(df.loc[idx, col]):
#             df.loc[idx, col] = mean_vals[col]


# df.info()

##################################################################################################################
##################################################################################################################
##################################################################################################################

#%%
print("--- train 데이터에 대한 결측치 확인 ---")
df[df['isTest'] == 0].info()
print()

print("--- test 데이터에 대한 결측치 확인 ---")
df[df['isTest'] == 1].info()


# #%%
# # 최근 1개월, 3개월, 6개월, 1년간 해당 아파트의 거래건수
# df['계약년월'] = pd.to_datetime(df['계약년월'], format='%Y%m', errors='coerce').dt.to_period('M').dt.to_timestamp()
# df = df.sort_values(['아파트명', '계약년월'])

# tqdm.pandas()

# def count_past_transactions(group, months):
#     group = group.sort_values('계약년월')
    
#     results = {f'과거{m}개월_거래수': [] for m in months}
    
#     for idx, row in group.iterrows():
#         for m in months:
#             past_start = row['계약년월'] - pd.DateOffset(months=m)
#             past_end = row['계약년월']
#             count = group[(group['계약년월'] >= past_start) & (group['계약년월'] < past_end)].shape[0]
#             results[f'과거{m}개월_거래수'].append(count)
    
#     return pd.DataFrame(results, index=group.index)


# months = [1, 3, 6, 12]
# monthly_counts_df = df.groupby('아파트명').progress_apply(count_past_transactions, months=months)
# monthly_counts_df.reset_index(level=0, drop=True, inplace=True)

# df = pd.concat([df, monthly_counts_df], axis=1)



#%%
# 파일 저장
# make data folder 'cleaned_data'
data_dir = '../../data/processed/cleaned_data'
os.makedirs(data_dir, exist_ok=True)

totaldata_filename = 'total_clean.csv'
totaldata_path = os.path.join(data_dir, totaldata_filename)

df.to_csv(totaldata_path, index=False, encoding='utf-8')


# dtype = {8: 'str',18: 'str',19: 'str',20: 'str',21: 'str'}
#%%
df = pd.read_csv('../../data/processed/cleaned_data/total_clean.csv')
df.info()

final_columns = [
                # 날짜(계약일 관련 변수)
                '계약일자', '계약년월', '계약년도', '계약월',

                # 위치 변수
                '자치구', '법정동', 
                '본번', '부번', '번지', '도로명',     # 모델링에 쓰일 변수 아님. 03.row_clean.py에서 결측치 보간에 이용할 변수
                '강남3구여부', '좌표X', '좌표Y',
                
                # 아파트 특성 변수
                '아파트명',                         # 모델링에 쓰일 변수 아님. 03.row_clean.py에서 결측치 제거에 이용할 변수
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
                '총인구수', '인구비중',
                '성비(남/여)',

                # # 과거 거래수 변수
                # '과거1개월_거래수',
                # '과거3개월_거래수',
                # '과거6개월_거래수',
                # '과거12개월_거래수',

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
test_clean.info()



#%%
# 추가: Train 데이터 내에서 Test 데이터 내에있는 자치구의 분포 살펴보기
sns.boxplot(data=train_clean[train_clean['자치구'].isin(test_clean['자치구'].unique())], x='자치구', y='target')
plt.title("Train 내 test 자치구의 거래가 분포")
plt.show()

#%%
# 추가: Train 데이터 내에서 Test 데이터 내에있는 법정동의 분포 살펴보기
sns.boxplot(data=train_clean[train_clean['법정동'].isin(test_clean['법정동'].unique())], x='법정동', y='target')
plt.title("Train 내 test 법정동의 거래가 분포")
plt.show()


#%%
# save 'train_clean.csv' and 'test_clean.csv'
traindata_filename = 'train_clean.csv'
testdata_filename = 'test_clean.csv'

traindata_path = os.path.join(data_dir, traindata_filename)
testdata_path = os.path.join(data_dir, testdata_filename)

train_clean.to_csv(traindata_path, index=False, encoding='utf-8')
test_clean.to_csv(testdata_path, index=False, encoding='utf-8')


