import requests
from bs4 import BeautifulSoup
import re
import pymongo
from urllib.parse import urljoin
 
actor_url = 'http://www.cine21.com/rank/person/content'
 
formdata_dict = dict()
formdata_dict['section'] = 'actor'
formdata_dict['period_start'] = '2020-07'
formdata_dict['gender'] = 'all'
formdata_dict['page'] = 1
formdata_dict
 
res = requests.post(actor_url, data=formdata_dict)
 
soup = BeautifulSoup(res.text,'html.parser')
 
actor_item_list = list()
 
actors = soup.select('li.people_li div.name')
hits = soup.select('ul.num_info > li > strong')
movies = soup.select('ul.mov_list')
rankings = soup.select('li.people_li span.grade ')
 
# actor변수가 <div class=name>를 의미함
for index, actor in enumerate(actors):
    actor_item_dict = dict()
    #print(actor)
    # 유아인(2편) 에서 (2편)을 제거하고 저장한다.
    actor_name = re.sub('\(\w*\)','',actor.text)
    actor_item_dict['배우이름'] = actor_name
    
    # 흥행지수 값에서 ,(콤마)를 제거하고 정수타입으로 변환해서 저장한다.
    actor_hit = int(hits[index].text.replace(',',''))
    actor_item_dict['흥행지수'] = actor_hit
    
    #출연작
    movie_titles = movies[index].select('li a span')    
    #출연작 여러개의 title을 저장하는 리스트
    movie_title_list = list()
    for movie_title in movie_titles:
        movie_title_list.append(movie_title.text)
    actor_item_dict['출연영화'] = movie_title_list
    
    #순위
    actor_ranking = rankings[index].text
    actor_item_dict['랭킹'] = int(actor_ranking)
    
    #배우의 상세정보를 보기 위해 http get다시 요청
    '''
        <div class=name>
            <a href="/db/person/info/?person_id=18040"> --> actor_detail_url
    '''
    actor_detail_url = actor.select_one('a').attrs['href']
    actor_detail_full_url = urljoin(actor_url,actor_detail_url)
    #print(actor_detail_full_url)
    
    res = requests.get(actor_detail_full_url)
    soup = BeautifulSoup(res.text,'html.parser')
       
    for li_tag in soup.select('ul.default_info li'):
        # dict의 key에 해당하는 값을 추출한다. dict['직업]'
        actor_item_field = li_tag.select_one('span.tit').text #직업
        
        #dict의 value에 해당하는 값을 추출. dict['직업'] == '배우'
        # <li><span class="tit">직업</span>배우</li>
        actor_item_value = re.sub('<span.*?>.*?</span>','',str(li_tag)) #<li>배우</li>
        actor_item_value = re.sub('<.*?>','',actor_item_value) #베우
        
        '''적용전
        \n https://gangdongwon.com
        '''
        regex = re.compile(r'[\n\r\t]')
        actor_item_value = regex.sub('',actor_item_value)
        '''적용후
        https://gangdongwon.com
        '''
        actor_item_dict[actor_item_field] = actor_item_value
    actor_item_list.append(actor_item_dict)
print(actor_item_list,len(actor_item_list))
