from re import M
from tkinter import Image
from urllib import response
from requests import request
import requests


headers = {
    'Host' : 'openapi.naver.com',
    'User-Agent' : 'curl/7.49.1',
    'Accept' : '*/*',
    'X-Naver-Client-Id' : 'dL152DAZ0zRBMN64XCor',
    'X-Naver-Client-Secret' : 'SNtgIMlyF3'    
    
}

url = 'https://openapi.naver.com/v1/search/movie.json'



params = {
    'query' : '관상',
    'items' : [
        "image":
        
    ]
}


response = requests.get(url, headers=headers, params=params)

print(response.text)
 
from PIL import Image

im = Image.open()
