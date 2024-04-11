import requests
import sys
import json
import os


assets_path = 'assets/'

def searchText(text):
    #input the query
    picquery = text
    pic_path = assets_path + picquery + ".jpg" 
    picparam = {'q':picquery,'media_type':'image'}
    response = requests.get("https://images-api.nasa.gov/search", params=picparam)
    response_dic = response.json()
    
    #find top-1 picture url in the nasa image api
    for piclink in response_dic["collection"]["items"]:
        if ('links' in piclink.keys()):
            for jpgurl in piclink["links"]:
                if ('href' in jpgurl.keys()):
                    #print(jpgurl["href"])
                    picrequest = requests.get(jpgurl["href"])
                    #download picture in pic_path
                    file = open(pic_path, "wb")
                    file.write(picrequest.content)
                    file.close()
                    return