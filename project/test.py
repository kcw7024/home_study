import requests
from bs4 import BeautifulSoup
import csv
import pandas as pd


actor_code = "강동원"

url="https://play.google.com/store/search?q=" + actor_code +"&c=movies"
headers={ 
    "User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.50 Safari/537.36",
    "Accept-Language":"ko-KR,ko" 
}
res=requests.get(url,headers=headers)
res.raise_for_status()

soup=BeautifulSoup(res.text,"lxml")
movies=soup.find_all("div",attrs={"class":"ImZGtf mpg5gc"})
movie_list=[]
movie_list.append(['이름', '별점', '설명'])


for movie in movies:
    title=movie.find("div",attrs={"class":"WsMG1c nnK0zc"}).get_text()
    des=movie.find("div",attrs={"class":"b8cIId f5NCO"}).get_text()
    price=movie.find("span",attrs={"class":"LrNMN"}).get_text()
    

    print(len(movies))
    print(f"제목:{title},별점:{price},설명{des}")


    movie_list.append([title,price,des])
    f=open("practice.csv","w",encoding="utf-8",newline="")
    csvWriter=csv.writer(f)
    for i in movie_list:
        csvWriter.writerow(i)
    f.close()

