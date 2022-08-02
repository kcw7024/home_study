import os
import sys
from matplotlib.pyplot import title
import requests
import pandas as pd
import csv

client_id = "dL152DAZ0zRBMN64XCor"
client_secret = "SNtgIMlyF3"


movie='가족'
header_parms ={"X-Naver-Client-Id":client_id,"X-Naver-Client-Secret":client_secret}
url = f"https://openapi.naver.com/v1/search/movie.json?query={movie}&display=100"
res=requests.get(url,headers=header_parms)
data=res.json()
##print(data)

titles=[]
links=[]
dates=[]
directors=[]
actors=[]
ratings=[]

for i in data['items']:
    titles.append(i['title'].strip('</b>').replace('<b>','').replace('</b>',''))
    links.append(i['link'])
    dates.append(i['pubDate'])
    directors.append(i['director'].split('|')[0])
    actors.append(i['actor'].split('|')[:-1])
    ratings.append(float(i['userRating']))
    


df = pd.DataFrame({'제목':titles, '감독':directors, '배우':actors, '평점':ratings})
df.to_csv('samples.csv', index=False)


# columns_name = ["movie_title","actor","score"]
# with open ( "samples.csv", "w", newline ="",encoding = 'utf8' ) as f:
#     write = csv.writer(f)
#     write.writerow(columns_name)
#     write.writerows(review_data