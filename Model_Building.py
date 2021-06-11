import json
import pandas as pd
import numpy as np
import millify
# Preprocessing
from collections import Counter
from sklearn import ensemble
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pandas as pd

df=pd.read_pickle("Movies_Data.pkl")

with open('genre.json', 'r') as f:
    genres = json.load(f)

imdb_score_per_genre = {}
gross_per_genre = {}
genre_columns = [col for col in df.columns if 'genre_' in col]
df_genres = df[genre_columns]
for genre, value in genres.items():
    mask = np.column_stack([df_genres[col] == value for col in df_genres])
    df_specific_genre = df.loc[mask.any(axis=1)][['genres', 'idmb_score', 'worldwide_gross']]
    imdb_score_per_genre[genre] = df_specific_genre.idmb_score.mean()
    gross_per_genre[genre] = df_specific_genre.worldwide_gross.mean()
gross_per_genre = {k:v for k,v in gross_per_genre.items() if pd.notnull(v)}

df_combine = pd.concat([pd.Series(gross_per_genre), pd.Series(imdb_score_per_genre)], axis=1)
df_combine = df_combine.sort_values(1, ascending=False)
df_combine.columns = ['Gross', 'Score']




## Fill NA for genres
df.genres  = df.genres.fillna('')

## Mean Inputer
from sklearn.impute import SimpleImputer

col_to_impute = ['actor_1_fb_likes', 'actor_2_fb_likes', 'actor_3_fb_likes',
                'domestic_gross', 'duration_sec', 'num_critic_for_reviews', 'num_facebook_like', 'num_user_for_reviews',
                'production_budget', 'total_cast_fb_likes', 'worldwide_gross', 'director_fb_likes']

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
df[col_to_impute] = imp.fit_transform(df[col_to_impute])

numerical_cols = list(df.dtypes[df.dtypes != 'object'].index)
not_wanted_cols = ['title_year', 'storyline', 'release_date', 'image_urls', 'movie_title', 'keywords', 'movie_imdb_link', 'num_voted_users'] + genre_columns
df.country = df.country.apply(lambda x:x.split('|'))
df.language = df.language.apply(lambda x:x.split('|'))
list_cols = ['country', 'genres', 'language']
cols_to_transform = [cols for cols in df.columns if cols not in numerical_cols + not_wanted_cols + list_cols]
df2 = df[cols_to_transform]

## Dummies for columns with list
df_col_list = pd.DataFrame()
for col in list_cols:
    df_col_list = pd.concat([df_col_list, pd.get_dummies(df[col].apply(pd.Series).stack()).sum(level=0)], axis=1)


## Dummies for columns with string
df_col_string = pd.get_dummies(df2, columns=cols_to_transform)

X_raw = pd.concat([df[numerical_cols], df_col_string, df_col_list], axis=1)
print('Columns dtypes :', Counter(X_raw.dtypes))

y = list(X_raw.idmb_score)
X = X_raw.drop('idmb_score', axis=1)


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print('Train', X_train.shape, 'Test', X_test.shape)

rf=ensemble.RandomForestRegressor(n_estimators=500,oob_score=True, )
rf.fit(X,y)
print("Training Score RandomForest: ", str(rf.score(X,y)))
print("Cross Validation (10 fold) Score: ", np.mean(cross_val_score(rf, X_train, Y_train, cv=10)))

