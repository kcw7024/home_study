import os
import sys
import requests

#네이버 영화 API 키 값
client_id = "dL152DAZ0zRBMN64XCor"
client_secret = "SNtgIMlyF3"

movie='관상'
header_parms ={"X-Naver-Client-Id":client_id,"X-Naver-Client-Secret":client_secret}
url = f"https://openapi.naver.com/v1/search/movie.json?query={movie}"
res=requests.get(url,headers=header_parms)
data =res.json()



title=[]
links=[]
dates=[]
director=[]
actors=[]
rating=[]

#데이터 전처리
for i in data['items']:
    title.append(i['title'].strip('</b>').replace('<b>','').replace('</b>',''))
    links.append(i['link'])
    dates.append(i['pubDate'])
    director.append(i['director'].split('|')[0])
    actors.append(i['actor'].split('|')[:-1])
    rating.append(float(i['userRating']))

import pandas as pd
data=pd.DataFrame([title,director,actors,rating]).T
data.columns=['영화 제목','감독','출연진','평점']

data.to_csv('movie_score.csv', header=False, index=False, mode='a')

print(data)