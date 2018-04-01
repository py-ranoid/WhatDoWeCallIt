from bs4 import BeautifulSoup as soup
from urllib2 import urlopen
url = "http://yclist.com/"
s = soup(urlopen(url).read(), 'lxml')
yc = []
for i in s.find('table').find_all('tr')[1:]:
    try:
        yc.append(str(i.find_all('td')[1].text))
    except:
        pass

import pandas as pd
ind = list(pd.read_csv('data/Startups/startup_funding.csv')
           ['StartupName'].astype('str'))


all_names = set(ind).union(set(yc))
len(all_names)


def fetch():
    return all_names
# import requests
#
# response = requests.get("https://community-angellist.p.mashape.com/startups?filter=raising&access_token=kB0vTc8gJ8mshnYZg8Z2Gv2qBJmlp17oaZKjsngZ82UBIjXePv",
#                         headers={
#                             "X-Mashape-Key": "kB0vTc8gJ8mshnYZg8Z2Gv2qBJmlp17oaZKjsngZ82UBIjXePv",
#                             "Accept": "text/plain"
#                         }
#                         )
#
# response.content
