import os
import sys
from matplotlib.pyplot import title
import requests
import pandas as pd
import csv
import json


#영화진흥위원회의 배우 코드를 수집한다
#영화진흥위원회에서 제공하는 배우 필모그래피 API를 사용하여 해당 배우들의 작품을 각각 수집, CSV 파일로 저장한다
#마지막에 서른명의 배우들의 필모그래피 CSV 파일을 합쳐준다.
#배우들 필모그래피 데이터 수집완료


key = '18e98023854faf2f7b24dd771fc99b58' #영화진흥위원회 DB 키
actor_code = '10090290' #각 배우들의 고유코드
 
url = 'http://kobis.or.kr/kobisopenapi/webservice/rest/people/searchPeopleInfo.json?key='+key+'&peopleCd='+actor_code
res = requests.get(url)
text= res.text

d = json.loads(text)
print(d)

date = []
titles = []
actor = []
a_code = []

for b in d['peopleInfoResult']['peopleInfo']['filmos'] :
    a_code.append(d['peopleInfoResult']['peopleInfo']['peopleCd'])
    actor.append(d['peopleInfoResult']['peopleInfo']['peopleNm'])
    date.append(b['movieCd'])
    titles.append(b['movieNm'])

data = pd.DataFrame([a_code,titles,date,actor]).T
data.columns=['배우코드','영화제목','개봉날짜','배우이름']
data.to_csv('actor_filmo.csv', header=False,index=False,mode='a')
#data.to_csv('actor_filmo.csv', index=False)


print(data)    


# {'peopleInfoResult': 
#     {'peopleInfo': 
#         {'peopleCd': '10000955', 'peopleNm': '강혜정', 'peopleNmEn': 'GANG Hye-jung', 'sex': '여자', 'repRoleNm': '배우', 'homepages': [], 
#           'filmos': [
#               {'movieCd': '20156562', 'movieNm': '루시드 드림', 'moviePartNm': '배우'}, 
#               {'movieCd': '20130574', 'movieNm': '개를 훔치는 완벽한 방법', 'moviePartNm': '배우'},
#               {'movieCd': '20120106', 'movieNm': '뒷담화: 감독이 미쳤어요', 'moviePartNm': '배우'}, 
#               {'movieCd': '20090875', 'movieNm': '걸프렌즈', 'moviePartNm': '배우'}, 
#               {'movieCd': '20090802', 'movieNm': '트라이앵글', 'moviePartNm': '배우'}, 
#               {'movieCd': '20081729', 'movieNm': '킬 미', 'moviePartNm': '배우'}, 
#               {'movieCd': '20090193', 'movieNm': '우리 집에 왜 왔니', 'moviePartNm': '배우'}, 
#               {'movieCd': '20060412', 'movieNm': '허브', 'moviePartNm': '배우'}, 
#               {'movieCd': '20060085', 'movieNm': '도마뱀', 'moviePartNm': '배우'}, 
#               {'movieCd': '20060110', 'movieNm': '보이지 않는 물결', 'moviePartNm': '배우'}, 
#               {'movieCd': '20060018', 'movieNm': '빨간모자의 진실', 'moviePartNm': '배우'}, 
#               {'movieCd': '20050180', 'movieNm': '웰컴 투 동막골', 'moviePartNm': '배우'}, 
#               {'movieCd': '20050110', 'movieNm': '연애의 목적', 'moviePartNm': '배우'}, 
#               {'movieCd': '20139626', 'movieNm': "쓰리, 몬스터 '컷'", 'moviePartNm': '배우'}, 
#               {'movieCd': '20030350', 'movieNm': '올드보이', 'moviePartNm': '배우'}, 
#               {'movieCd': '20010094', 'movieNm': '나비', 'moviePartNm': '배우'}, 
#               {'movieCd': '20150405', 'movieNm': '가불병정', 'moviePartNm': '배우'}, 
#               {'movieCd': '20040637', 'movieNm': '쓰리 몬스터', 'moviePartNm': '배우'}
#               ]}, 
#         'source': '영화진흥위원회'}}