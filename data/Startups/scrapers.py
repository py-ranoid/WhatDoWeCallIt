from bs4 import BeautifulSoup as soup
from urllib2 import urlopen
import pandas as pd


def get_yc():
    url = "http://yclist.com/"
    s = soup(urlopen(url).read(), 'lxml')
    yc = []
    for i in s.find('table').find_all('tr')[1:]:
        try:
            yc.append(str(i.find_all('td')[1].text))
        except UnicodeError:
            pass
    return yc


def get_ind():
    df = pd.read_csv('data/Startups/startup_funding.csv')
    df.head()
    ind = list(df['StartupName'].astype('str'))
    return ind


def get_big():
    df = pd.read_csv('data/Startups/startups.csv')
    df.head()
    big = list(df.dropna()['name'])
    return big


def get_startups():
    funcs = [get_yc, get_ind, get_big]
    all = set()
    for f in funcs:
        all = all.union(set(f()))
    return list(all)
