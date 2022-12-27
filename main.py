#%%
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import learning_curve
import lightgbm
import xgboost

# %%
fstats = 'data/Seasons_Stats.csv'
fplayers = 'data/Players.csv'
fdata = 'data/player_data.csv'
fsalary =  'data/NBA_season1718_salary.csv'

stat = pd.read_csv(fstats)
players = pd.read_csv(fplayers)
player_data= pd.read_csv(fdata)
salary = pd.read_csv(fsalary)
# %%
# stat data
stat.head(3)
# %%
# player data
players.head(3)
# %%
# player data
player_data.head(3)
# %%
# salary data
salary.head(3)
# %%
# dataframe 컬럼명 소문자 변환
stat.columns = stat.columns.str.lower() #
players.columns = players.columns.str.lower()
player_data.columns = player_data.columns.str.lower()
salary.columns = salary.columns.str.lower() #
# %%
# 확인
print(stat.columns)
print(players.columns)
print(player_data.columns)
print(salary.columns)
# %%
# player_data 컬럼명 변경: name > player
player_data.rename(columns={'name':'player'}, inplace=True)
player_data.head(1)
# %%
# stat + salar
player_stat = pd.merge(stat, salary, on='player', how='inner')
print(f'player_stat shape: {player_stat.shape}')
player_stat.head(1)

# %%
stat_avg = stat.groupby('player').mean()
stat_avg
# %%
df = pd.merge(stat_avg, salary, on='player', how='inner').drop(['unnamed: 0_x', 'year', 'unnamed: 0_y'], axis=1)
print(f'df shape: {df.shape}')
# %%
player_data.head(3)

# %%
# plyaer_data > position, weight
# players > height

m1 = player_data[['player', 'position', 'weight']]
m2 = players[['player', 'height']]
m3 = pd.merge(m1, m2, on='player', how='inner')
df = pd.merge(df, m3, on='player', how='inner')

print(f'df shape: {df.shape}')
df.head(3)

# %%
# 결측치 확인
df.isnull().sum()
#%%
# 컬럼명 변경(salary=target)
df.rename(columns={'season17_18':'salary'}, inplace=True)
# %%
df = df.drop(['blanl', 'blank2', '3p%', '2p%'], axis=1)
df.isnull().sum()

#%%
player_df = df['player']
obj = df.select_dtypes('object')
num = df.select_dtypes('number')
obj = obj.drop('player', axis=1)
obj
#%%
#레이블링
le = LabelEncoder()
a = pd.DataFrame(le.fit_transform(obj['tm']))
b = pd.DataFrame(le.fit_transform(obj['position']))
c = pd.concat([a, b], axis=1)
d = pd.concat([c, num], axis=1)
df = pd.concat([player_df, d], axis=1)

print(f'df shape: {df.shape}')
df.info()

#%%
# featre들과 target(=salary)와의 상관관계 dataframe
df_corr = pd.DataFrame(df.corr().iloc[:,46])
df_corr
#%%
df5 = df_corr[df_corr['salary']>=0.5]
df5
# %%
# corr heatmap
mask = np.zeros_like(df.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(df.corr(), cmap='PuBu', annot=False, mask=mask, linewidths=0.5, cbar_kws={"shrink": .5}, vmin = -1,vmax = 1)  
# %%
print(f'스케일링이 필요한 컬럼들: {df5.transpose().columns}')
# %%
cols = ['gs', 'mp', 'ows', 'dws', 'ws', 'vorp', 'fg', 'fga', '2p', '2pa', 'ft', 'fta', 'drb', 'trb', 'stl', 'tov', 'pf', 'pts', 'salary']

df = df[cols]

#%%
#스케일링
mms = MinMaxScaler()
df = pd.DataFrame(mms.fit_transform(df), columns=cols)

# %%
# dataframe 확인
df.head()

dff = df.iloc[:, :-1]
# %%
# PCA(차원축소) 진행
# n=2
pca = PCA(n_components=2)  #주성분 개수 결정
pcs = pca.fit_transform(dff)
pcs_df = pd.DataFrame(data=pcs, columns=['pc1', 'pc2'])
print(f'주성분 2개 일 때의 분산설명력: {sum(pca.explained_variance_ratio_)}')

# n=3
pca = PCA(n_components=3)  #주성분 개수 결정
pcs = pca.fit_transform(dff)
pcs_df = pd.DataFrame(data=pcs, columns=['pc1', 'pc2', 'pc3'])
print(f'주성분 3개 일 때의 분산설명력: {sum(pca.explained_variance_ratio_)}')


# n=4
pca = PCA(n_components=4)  #주성분 개수 결정
pcs = pca.fit_transform(dff)
pcs_df = pd.DataFrame(data=pcs, columns=['pc1', 'pc2', 'pc3', 'pc4'])
print(f'주성분 4개 일 때의 분산설명력: {sum(pca.explained_variance_ratio_)}')

# %%
pcs_df
# %%
### 머신러닝
X = pcs_df
y = df.iloc[:, -1]

dt = DecisionTreeRegressor()
rf = RandomForestRegressor()
lgbm = lightgbm.LGBMRegressor()
xgb = xgboost.XGBRegressor()

mse = mean_squared_error

# %%
# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# %%
# 회귀 함수
def reg(algo, X_train, X_test, y_train, y_test):
    algo.fit(X_train, y_train)
    pred = algo.predict(X_test)
    rmse = np.sqrt(mse(y_test, pred))
    r2 = r2_score(y_test, pred)

    print(f"{algo.__class__.__name__} rmse score: {rmse}")
    print(f"{algo.__class__.__name__} r2 score: {r2}")
    print('='*60)

# %%
algos = [dt, rf, lgbm, xgb]
for algo in algos:
    reg(algo, X_train, X_test, y_train, y_test)
# %%)