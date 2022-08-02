import pandas as pd
import csv
import json

from sqlalchemy import false



path = 'C:/home_study/home_study/'

actor_fillmo = pd.read_csv(path + 'actor_filmo_test.csv')
movie_score = pd.read_csv(path + 'movie_score_test.csv')

df1 = pd.DataFrame(actor_fillmo)
df2 = pd.DataFrame(movie_score)

df3 = pd.merge(df1, df2, on='영화제목')

df3.loc[df3.배우코드 == 10000558,'배우코드'] = 'kang_dongwon'
df3.loc[df3.배우코드 == 10000955,'배우코드'] = 'kang_hyejeong'
df3.loc[df3.배우코드 == 10001670,'배우코드'] = 'go_ahseong'
df3.loc[df3.배우코드 == 10005276,'배우코드'] = 'kim_yunseok'
df3.loc[df3.배우코드 == 20201026,'배우코드'] = 'kim_taeri'
df3.loc[df3.배우코드 == 10006380,'배우코드'] = 'kim_hyesoo'
df3.loc[df3.배우코드 == 10026732,'배우코드'] = 'moon_sori'
df3.loc[df3.배우코드 == 20209686,'배우코드'] = 'park_sodam'
df3.loc[df3.배우코드 == 20312856,'배우코드'] = 'son_seokgu'
df3.loc[df3.배우코드 == 10036883,'배우코드'] = 'son_yejin'
df3.loc[df3.배우코드 == 10037018,'배우코드'] = 'song_kangho'
df3.loc[df3.배우코드 == 10037291,'배우코드'] = 'song_hyekyo'
df3.loc[df3.배우코드 == 20125828,'배우코드'] = 'suzy'
df3.loc[df3.배우코드 == 10040665,'배우코드'] = 'shin_hakyun'
df3.loc[df3.배우코드 == 10054128,'배우코드'] = 'yoo_haejin'
df3.loc[df3.배우코드 == 10054391,'배우코드'] = 'yoon_yeojeong'
df3.loc[df3.배우코드 == 10055626,'배우코드'] = 'lee_byunghun'
df3.loc[df3.배우코드 == 20110323,'배우코드'] = 'e_som'
df3.loc[df3.배우코드 == 10057315,'배우코드'] = 'lee_jungjae'
df3.loc[df3.배우코드 == 10057349,'배우코드'] = 'lee_jehoon'
df3.loc[df3.배우코드 == 10061252,'배우코드'] = 'jeon_doyeon'
df3.loc[df3.배우코드 == 20282652,'배우코드'] = 'jeon_jongseo'
df3.loc[df3.배우코드 == 10061467,'배우코드'] = 'jeon_jihyun'
df3.loc[df3.배우코드 == 10062025,'배우코드'] = 'jung_woosung'
df3.loc[df3.배우코드 == 10066380,'배우코드'] = 'cho_seungwoo'
df3.loc[df3.배우코드 == 10066899,'배우코드'] = 'jo_inseong'
df3.loc[df3.배우코드 == 20133966,'배우코드'] = 'chun_woohee'
df3.loc[df3.배우코드 == 10072251,'배우코드'] = 'choi_minsik'
df3.loc[df3.배우코드 == 10087253,'배우코드'] = 'ha_jungwoo'
df3.loc[df3.배우코드 == 10090290,'배우코드'] = 'hwang_jeongmin'

print(df3)
 
df3.to_csv('actor_filmo_n_score.csv', index=false)

