
# **NOTE:** The folium map isn't visible in this markdown file. If you want to see it, go to the blog post.


```python
import requests
import pandas as pd
from pandas.io.json import json_normalize
import numpy as np 
import random
from tqdm import tqdm_notebook
import folium 
from geopy.geocoders import Nominatim
import urllib.request
from bs4 import BeautifulSoup
import matplotlib.cm as cm
import matplotlib.colors as colors
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


#from IPython.display import Image 
#from IPython.core.display import HTML 
#from pandas.io.json import json_normalize
```

# The Battle of Neighborhoods

This notebook is part 1 of the submissions for the final capstone project in the IBM Data Science Professional Certificate course on Coursera.

## Preamble

Mr. Nolan is going to be moving in to the city of Chennai, located at the edge part of the Indian subcontinent. He needs to find suitable housing. However, he frets. He doesn't worry about the locality one bit; except the fulfillment of one condition: there needs to be a lot of restaurants/food stalls nearby. You see, Mr. Nolan is a foodie. Not a day goes by where he does not make or savour a new dish. He has come to me for help, asking me to analyze the different areas in the city of Chennai and find which neighborhoods would be the best for a foodie like him to move in to. 

Me, being the perfect philanthropist as I am, have decided to help Mr. Nolan using a bit of data and a bit of science. 

Jokes aside, here's how I plan to do it. 



## Introduction

The aim here is to find neighborhoods with a high frequency of restaurants/food stalls/cafes. Firstly, the number of neighborhoods and their respective coordinates need to be retrieved, so that Foursquare can find nearby venues. Using this data, Foursquare should search for nearby venues and get their categories. 

These venues are then clustered using k-means. The cluster in which eateries are of the highest frequency will be the set of neighborhoods we are looking for. All of these neighborhoods are suitable for Mr.Nolan to move in.

This problem can also be easily extended to fit other requests, such as finding the neighborhoods with low real estate prices, neighborhoods with a wide variety of grocery shops, neighborhoods closes to public transport systems etc. 

The target audience here is people who are moving to a new city and require some knowledge about the neighborhoods beforehand so that they can decide the place they want to live in. 

## Data

### Wikipedia Scraper

Since available data for Chennai city was sparse, I've manually scraped the list of neighborhoods from [this](https://en.wikipedia.org/wiki/Areas_of_Chennai) Wikipedia page using bs4, and then grabbed all the hyperlinks. Using urllib, these links are visited individually and the coordinates and pincodes (neighborhoods identifiers) are scraped and put into a pandas dataframe.


```python
url = 'https://en.wikipedia.org/wiki/Areas_of_Chennai'
page_unparsed = urllib.request.urlopen(url)
soup = BeautifulSoup(page_unparsed, 'html.parser')
```


```python
wiki_rows = [] # each row in the wikipedia table
urls = []
names = []

wiki_table = soup.find_all("table", {"class": "wikitable"})
for row in wiki_table:
  wiki_rows.append(row.find_all('a', href=True))

# gets names and links of each neighborhood so that further scraping can be done
for i in range(len(wiki_rows[0])):
  urls.append('https://en.wikipedia.org' + wiki_rows[0][i]['href'])
  names.append(wiki_rows[0][i].text)
```


```python
# getting data from each neighborhood

latitudes = []
longitudes = []
pincodes = []

for url in tqdm_notebook(urls, total = len(urls), unit = 'url'):
  try: # because some links are broken
    page_unparsed = urllib.request.urlopen(url)
    soup = BeautifulSoup(page_unparsed, 'html.parser')
  except:
    continue

  coords = soup.find("span", {"class" : "geo-dec"})
  pincode = soup.find("div", {"class" : "postal-code"})

  if coords == None:  # because some pages do not have coordinates listed
    latitudes.append(np.nan)
    longitudes.append(np.nan)

  else:
    coords = coords.text.split()
    latitudes.append(float(coords[0].replace('N', '').replace('°', '')))
    longitudes.append(float(coords[1].replace('E', '').replace('°', '')))
```


    HBox(children=(IntProgress(value=0, max=157), HTML(value='')))


    
    


```python
neighborhoods = pd.DataFrame(list(zip(names, latitudes, longitudes)), columns =['Name', 'Latitude', 'Longitude']) 
neighborhoods = neighborhoods[neighborhoods['Latitude'].notnull()]
neighborhoods = neighborhoods[neighborhoods['Longitude'].notnull()]
neighborhoods.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Adambakkam</td>
      <td>12.9900</td>
      <td>80.2000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adyar</td>
      <td>13.0063</td>
      <td>80.2574</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Alandur</td>
      <td>13.0030</td>
      <td>80.2040</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Alapakkam</td>
      <td>13.0490</td>
      <td>80.1673</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Alwarpet</td>
      <td>13.0339</td>
      <td>80.2486</td>
    </tr>
  </tbody>
</table>
</div>



### Foursquare

Using Foursquare, individual neighboords are searched to find nearby venues and their categories withing a 500m radius of a randomnly chosen neighborhood, Choolaimedu.


```python
CLIENT_ID = 'BEYJXL240KTU54SBM4YR1FGO2LSOQ5LVFBRAYFW5YNP4YFI0' # your Foursquare ID
CLIENT_SECRET = 'XOEBQ2HUD5KVGZVLD1ETXN45MF4NQN1XBAJ0FNN4R1VZA5D0' # your Foursquare Secret
VERSION = '20190130'
```


```python
neighborhood_latitude = neighborhoods[neighborhoods['Name'] == 'Choolaimedu']['Latitude']
neighborhood_longitude = neighborhoods[neighborhoods['Name'] == 'Choolaimedu']['Longitude']

url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID[::-1], 
    CLIENT_SECRET[::-1], 
    VERSION, 
    neighborhood_latitude, 
    neighborhood_longitude, 
    radius, 
    LIMIT)

results = requests.get(url).json()
```


```python
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']
```


```python
LIMIT = 100 # limit of number of venues returned by Foursquare API
radius = 500 # 

def getNearbyVenues(names, latitudes, longitudes, radius=500):
    venues_list = []
    for name, lat, lng in tqdm_notebook(zip(names, latitudes, longitudes), total = neighborhoods.shape[0], unit = 'neighborhoods'):
        # print(name)

        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID[::-1], 
            CLIENT_SECRET[::-1], 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['categories'][0]['id'],
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue ID',
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)
```


```python
chennai_venues = getNearbyVenues(names = neighborhoods['Name'],
                                   latitudes = neighborhoods['Latitude'],
                                   longitudes = neighborhoods['Longitude'])
chennai_venues.head()
```


    HBox(children=(IntProgress(value=0, max=150), HTML(value='')))


    
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>Neighborhood Latitude</th>
      <th>Neighborhood Longitude</th>
      <th>Venue</th>
      <th>Venue ID</th>
      <th>Venue Latitude</th>
      <th>Venue Longitude</th>
      <th>Venue Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Adambakkam</td>
      <td>12.9900</td>
      <td>80.2000</td>
      <td>Pizza Republic</td>
      <td>4bf58dd8d48988d1ca941735</td>
      <td>12.990987</td>
      <td>80.198613</td>
      <td>Pizza Place</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adambakkam</td>
      <td>12.9900</td>
      <td>80.2000</td>
      <td>Loiee</td>
      <td>4bf58dd8d48988d16a941735</td>
      <td>12.992197</td>
      <td>80.199000</td>
      <td>Bakery</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Adambakkam</td>
      <td>12.9900</td>
      <td>80.2000</td>
      <td>Thalapakattu Hotel</td>
      <td>4bf58dd8d48988d142941735</td>
      <td>12.991979</td>
      <td>80.198937</td>
      <td>Asian Restaurant</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Adambakkam</td>
      <td>12.9900</td>
      <td>80.2000</td>
      <td>The Great Kabab Factory</td>
      <td>5283c7b4e4b094cb91ec88d7</td>
      <td>12.993796</td>
      <td>80.201702</td>
      <td>Kebab Restaurant</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Adyar</td>
      <td>13.0063</td>
      <td>80.2574</td>
      <td>Bombay Brassiere</td>
      <td>54135bf5e4b08f3d2429dfdd</td>
      <td>13.006961</td>
      <td>80.256419</td>
      <td>North Indian Restaurant</td>
    </tr>
  </tbody>
</table>
</div>



# Methodology

One hot encoding.


```python
chennai_onehot = pd.get_dummies(chennai_venues[['Venue Category']], prefix="", prefix_sep="")
chennai_onehot['Neighborhood'] = chennai_venues['Neighborhood'] 
fixed_columns = [chennai_onehot.columns[-1]] + list(chennai_onehot.columns[:-1])
chennai_onehot = chennai_onehot[fixed_columns]
chennai_onehot.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>ATM</th>
      <th>Accessories Store</th>
      <th>African Restaurant</th>
      <th>Airport Gate</th>
      <th>Airport Lounge</th>
      <th>Airport Terminal</th>
      <th>American Restaurant</th>
      <th>Antique Shop</th>
      <th>Arcade</th>
      <th>Art Gallery</th>
      <th>Arts &amp; Crafts Store</th>
      <th>Asian Restaurant</th>
      <th>Athletics &amp; Sports</th>
      <th>Auto Garage</th>
      <th>BBQ Joint</th>
      <th>Bakery</th>
      <th>Bar</th>
      <th>Beach</th>
      <th>Bed &amp; Breakfast</th>
      <th>Bike Rental / Bike Share</th>
      <th>Bistro</th>
      <th>Bookstore</th>
      <th>Boutique</th>
      <th>Bowling Alley</th>
      <th>Breakfast Spot</th>
      <th>Buffet</th>
      <th>Burger Joint</th>
      <th>Bus Line</th>
      <th>Bus Station</th>
      <th>Bus Stop</th>
      <th>Business Service</th>
      <th>Café</th>
      <th>Camera Store</th>
      <th>Chettinad Restaurant</th>
      <th>Chinese Restaurant</th>
      <th>Church</th>
      <th>Clothing Store</th>
      <th>Coffee Shop</th>
      <th>Concert Hall</th>
      <th>...</th>
      <th>Park</th>
      <th>Performing Arts Venue</th>
      <th>Pharmacy</th>
      <th>Pier</th>
      <th>Pizza Place</th>
      <th>Platform</th>
      <th>Playground</th>
      <th>Pool</th>
      <th>Pool Hall</th>
      <th>Print Shop</th>
      <th>Pub</th>
      <th>Ramen Restaurant</th>
      <th>Restaurant</th>
      <th>Road</th>
      <th>Rock Club</th>
      <th>Russian Restaurant</th>
      <th>Sandwich Place</th>
      <th>Sculpture Garden</th>
      <th>Shoe Store</th>
      <th>Shopping Mall</th>
      <th>Smoke Shop</th>
      <th>Snack Place</th>
      <th>Soccer Stadium</th>
      <th>South Indian Restaurant</th>
      <th>Spa</th>
      <th>Sports Bar</th>
      <th>Sports Club</th>
      <th>Sri Lankan Restaurant</th>
      <th>Stadium</th>
      <th>Supermarket</th>
      <th>Tea Room</th>
      <th>Tennis Court</th>
      <th>Thai Restaurant</th>
      <th>Theater</th>
      <th>Theme Park</th>
      <th>Tourist Information Center</th>
      <th>Train</th>
      <th>Train Station</th>
      <th>Vegetarian / Vegan Restaurant</th>
      <th>Women's Store</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Adambakkam</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adambakkam</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Adambakkam</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Adambakkam</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Adyar</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 148 columns</p>
</div>




```python
chennai_grouped = chennai_onehot.groupby('Neighborhood').mean().reset_index()
chennai_grouped
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>ATM</th>
      <th>Accessories Store</th>
      <th>African Restaurant</th>
      <th>Airport Gate</th>
      <th>Airport Lounge</th>
      <th>Airport Terminal</th>
      <th>American Restaurant</th>
      <th>Antique Shop</th>
      <th>Arcade</th>
      <th>Art Gallery</th>
      <th>Arts &amp; Crafts Store</th>
      <th>Asian Restaurant</th>
      <th>Athletics &amp; Sports</th>
      <th>Auto Garage</th>
      <th>BBQ Joint</th>
      <th>Bakery</th>
      <th>Bar</th>
      <th>Beach</th>
      <th>Bed &amp; Breakfast</th>
      <th>Bike Rental / Bike Share</th>
      <th>Bistro</th>
      <th>Bookstore</th>
      <th>Boutique</th>
      <th>Bowling Alley</th>
      <th>Breakfast Spot</th>
      <th>Buffet</th>
      <th>Burger Joint</th>
      <th>Bus Line</th>
      <th>Bus Station</th>
      <th>Bus Stop</th>
      <th>Business Service</th>
      <th>Café</th>
      <th>Camera Store</th>
      <th>Chettinad Restaurant</th>
      <th>Chinese Restaurant</th>
      <th>Church</th>
      <th>Clothing Store</th>
      <th>Coffee Shop</th>
      <th>Concert Hall</th>
      <th>...</th>
      <th>Park</th>
      <th>Performing Arts Venue</th>
      <th>Pharmacy</th>
      <th>Pier</th>
      <th>Pizza Place</th>
      <th>Platform</th>
      <th>Playground</th>
      <th>Pool</th>
      <th>Pool Hall</th>
      <th>Print Shop</th>
      <th>Pub</th>
      <th>Ramen Restaurant</th>
      <th>Restaurant</th>
      <th>Road</th>
      <th>Rock Club</th>
      <th>Russian Restaurant</th>
      <th>Sandwich Place</th>
      <th>Sculpture Garden</th>
      <th>Shoe Store</th>
      <th>Shopping Mall</th>
      <th>Smoke Shop</th>
      <th>Snack Place</th>
      <th>Soccer Stadium</th>
      <th>South Indian Restaurant</th>
      <th>Spa</th>
      <th>Sports Bar</th>
      <th>Sports Club</th>
      <th>Sri Lankan Restaurant</th>
      <th>Stadium</th>
      <th>Supermarket</th>
      <th>Tea Room</th>
      <th>Tennis Court</th>
      <th>Thai Restaurant</th>
      <th>Theater</th>
      <th>Theme Park</th>
      <th>Tourist Information Center</th>
      <th>Train</th>
      <th>Train Station</th>
      <th>Vegetarian / Vegan Restaurant</th>
      <th>Women's Store</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Adambakkam</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.250000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adyar</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.032258</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.032258</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.032258</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.032258</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.032258</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.032258</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.032258</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.032258</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.064516</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Alandur</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.250000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Alapakkam</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Alwarpet</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.037037</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.037037</td>
      <td>0.0</td>
      <td>0.037037</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.037037</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.037037</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.074074</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.037037</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.074074</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>126</th>
      <td>Vadapalani</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>127</th>
      <td>Valasaravakkam</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.333333</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.333333</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>128</th>
      <td>Vallalar Nagar</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.166667</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.166667</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.166667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>129</th>
      <td>Vanagaram</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.030303</td>
      <td>0.030303</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.060606</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.030303</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.242424</td>
      <td>0.030303</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.030303</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.030303</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.030303</td>
      <td>0.0</td>
      <td>0.030303</td>
      <td>0.0</td>
      <td>0.030303</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.030303</td>
      <td>0.030303</td>
    </tr>
    <tr>
      <th>130</th>
      <td>Vandalur</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>131 rows × 148 columns</p>
</div>



Getting the most frequent venues in each neighborhood.


```python
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]

num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = chennai_grouped['Neighborhood']

for ind in np.arange(chennai_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(chennai_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Adambakkam</td>
      <td>Pizza Place</td>
      <td>Bakery</td>
      <td>Kebab Restaurant</td>
      <td>Asian Restaurant</td>
      <td>Women's Store</td>
      <td>Electronics Store</td>
      <td>Fishing Store</td>
      <td>Fish Market</td>
      <td>Fast Food Restaurant</td>
      <td>Farmers Market</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adyar</td>
      <td>Indian Restaurant</td>
      <td>North Indian Restaurant</td>
      <td>Vegetarian / Vegan Restaurant</td>
      <td>Electronics Store</td>
      <td>Juice Bar</td>
      <td>Coffee Shop</td>
      <td>Dessert Shop</td>
      <td>Movie Theater</td>
      <td>Café</td>
      <td>Fast Food Restaurant</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Alandur</td>
      <td>Hotel</td>
      <td>Fish Market</td>
      <td>South Indian Restaurant</td>
      <td>Movie Theater</td>
      <td>Donut Shop</td>
      <td>Fast Food Restaurant</td>
      <td>Farmers Market</td>
      <td>Farm</td>
      <td>Electronics Store</td>
      <td>Diner</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Alapakkam</td>
      <td>Indian Restaurant</td>
      <td>Fast Food Restaurant</td>
      <td>Women's Store</td>
      <td>Donut Shop</td>
      <td>Flea Market</td>
      <td>Fishing Store</td>
      <td>Fish Market</td>
      <td>Farmers Market</td>
      <td>Farm</td>
      <td>Electronics Store</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Alwarpet</td>
      <td>Indian Restaurant</td>
      <td>Lounge</td>
      <td>Hotel</td>
      <td>Japanese Restaurant</td>
      <td>Restaurant</td>
      <td>Thai Restaurant</td>
      <td>Italian Restaurant</td>
      <td>South Indian Restaurant</td>
      <td>Pizza Place</td>
      <td>Burger Joint</td>
    </tr>
  </tbody>
</table>
</div>




```python
# set number of clusters
kclusters = 4

chennai_grouped_clustering = chennai_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(chennai_grouped_clustering)

# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)
chennai_merged = neighborhoods

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
chennai_merged = chennai_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Name')

chennai_merged.head() # check the last columns!
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Adambakkam</td>
      <td>12.9900</td>
      <td>80.2000</td>
      <td>1.0</td>
      <td>Pizza Place</td>
      <td>Bakery</td>
      <td>Kebab Restaurant</td>
      <td>Asian Restaurant</td>
      <td>Women's Store</td>
      <td>Electronics Store</td>
      <td>Fishing Store</td>
      <td>Fish Market</td>
      <td>Fast Food Restaurant</td>
      <td>Farmers Market</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adyar</td>
      <td>13.0063</td>
      <td>80.2574</td>
      <td>0.0</td>
      <td>Indian Restaurant</td>
      <td>North Indian Restaurant</td>
      <td>Vegetarian / Vegan Restaurant</td>
      <td>Electronics Store</td>
      <td>Juice Bar</td>
      <td>Coffee Shop</td>
      <td>Dessert Shop</td>
      <td>Movie Theater</td>
      <td>Café</td>
      <td>Fast Food Restaurant</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Alandur</td>
      <td>13.0030</td>
      <td>80.2040</td>
      <td>1.0</td>
      <td>Hotel</td>
      <td>Fish Market</td>
      <td>South Indian Restaurant</td>
      <td>Movie Theater</td>
      <td>Donut Shop</td>
      <td>Fast Food Restaurant</td>
      <td>Farmers Market</td>
      <td>Farm</td>
      <td>Electronics Store</td>
      <td>Diner</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Alapakkam</td>
      <td>13.0490</td>
      <td>80.1673</td>
      <td>0.0</td>
      <td>Indian Restaurant</td>
      <td>Fast Food Restaurant</td>
      <td>Women's Store</td>
      <td>Donut Shop</td>
      <td>Flea Market</td>
      <td>Fishing Store</td>
      <td>Fish Market</td>
      <td>Farmers Market</td>
      <td>Farm</td>
      <td>Electronics Store</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Alwarpet</td>
      <td>13.0339</td>
      <td>80.2486</td>
      <td>0.0</td>
      <td>Indian Restaurant</td>
      <td>Lounge</td>
      <td>Hotel</td>
      <td>Japanese Restaurant</td>
      <td>Restaurant</td>
      <td>Thai Restaurant</td>
      <td>Italian Restaurant</td>
      <td>South Indian Restaurant</td>
      <td>Pizza Place</td>
      <td>Burger Joint</td>
    </tr>
  </tbody>
</table>
</div>




```python
chennai_merged = chennai_merged[chennai_merged['Cluster Labels'].notnull()]

# create map
map_clusters = folium.Map(location=[13.067439, 80.237617], zoom_start=11)

# set color scheme for the clusters
'''
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters+1)]
colors_array = cm.hsv(np.linspace(0, 1, len(ys)))
hsv = [colors.rgb2hex(i) for i in colors_array]
'''

colors = ["#ff0000", "#3d84ad", "#000000", "#ffff00"]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(chennai_merged['Latitude'], chennai_merged['Longitude'], chennai_merged['Name'], chennai_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=colors[int(cluster)],
        fill=True,
        fill_color=colors[int(cluster)],
        fill_opacity=0.7).add_to(map_clusters)
       

map_clusters
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><iframe src="data:text/html;charset=utf-8;base64,PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgPHNjcmlwdD5MX1BSRUZFUl9DQU5WQVM9ZmFsc2U7IExfTk9fVE9VQ0g9ZmFsc2U7IExfRElTQUJMRV8zRD1mYWxzZTs8L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS40LjAvZGlzdC9sZWFmbGV0LmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2NvZGUuanF1ZXJ5LmNvbS9qcXVlcnktMS4xMi40Lm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvanMvYm9vdHN0cmFwLm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvTGVhZmxldC5hd2Vzb21lLW1hcmtlcnMvMi4wLjIvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuanMiPjwvc2NyaXB0PgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS40LjAvZGlzdC9sZWFmbGV0LmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvY3NzL2Jvb3RzdHJhcC10aGVtZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vZm9udC1hd2Vzb21lLzQuNi4zL2Nzcy9mb250LWF3ZXNvbWUubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9yYXdjZG4uZ2l0aGFjay5jb20vcHl0aG9uLXZpc3VhbGl6YXRpb24vZm9saXVtL21hc3Rlci9mb2xpdW0vdGVtcGxhdGVzL2xlYWZsZXQuYXdlc29tZS5yb3RhdGUuY3NzIi8+CiAgICA8c3R5bGU+aHRtbCwgYm9keSB7d2lkdGg6IDEwMCU7aGVpZ2h0OiAxMDAlO21hcmdpbjogMDtwYWRkaW5nOiAwO308L3N0eWxlPgogICAgPHN0eWxlPiNtYXAge3Bvc2l0aW9uOmFic29sdXRlO3RvcDowO2JvdHRvbTowO3JpZ2h0OjA7bGVmdDowO308L3N0eWxlPgogICAgCiAgICA8bWV0YSBuYW1lPSJ2aWV3cG9ydCIgY29udGVudD0id2lkdGg9ZGV2aWNlLXdpZHRoLAogICAgICAgIGluaXRpYWwtc2NhbGU9MS4wLCBtYXhpbXVtLXNjYWxlPTEuMCwgdXNlci1zY2FsYWJsZT1ubyIgLz4KICAgIDxzdHlsZT4jbWFwXzAyMzdjOTQxNzdhMzQ4ZmY5MzQzMWQxYTg4NmQwMWUxIHsKICAgICAgICBwb3NpdGlvbjogcmVsYXRpdmU7CiAgICAgICAgd2lkdGg6IDEwMC4wJTsKICAgICAgICBoZWlnaHQ6IDEwMC4wJTsKICAgICAgICBsZWZ0OiAwLjAlOwogICAgICAgIHRvcDogMC4wJTsKICAgICAgICB9CiAgICA8L3N0eWxlPgo8L2hlYWQ+Cjxib2R5PiAgICAKICAgIAogICAgPGRpdiBjbGFzcz0iZm9saXVtLW1hcCIgaWQ9Im1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSIgPjwvZGl2Pgo8L2JvZHk+CjxzY3JpcHQ+ICAgIAogICAgCiAgICAKICAgICAgICB2YXIgYm91bmRzID0gbnVsbDsKICAgIAoKICAgIHZhciBtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEgPSBMLm1hcCgKICAgICAgICAnbWFwXzAyMzdjOTQxNzdhMzQ4ZmY5MzQzMWQxYTg4NmQwMWUxJywgewogICAgICAgIGNlbnRlcjogWzEzLjA2NzQzOSwgODAuMjM3NjE3XSwKICAgICAgICB6b29tOiAxMSwKICAgICAgICBtYXhCb3VuZHM6IGJvdW5kcywKICAgICAgICBsYXllcnM6IFtdLAogICAgICAgIHdvcmxkQ29weUp1bXA6IGZhbHNlLAogICAgICAgIGNyczogTC5DUlMuRVBTRzM4NTcsCiAgICAgICAgem9vbUNvbnRyb2w6IHRydWUsCiAgICAgICAgfSk7CgoKICAgIAogICAgdmFyIHRpbGVfbGF5ZXJfZThjNzEzNTMyNzg1NDhiNjlmOGExOGY5MGU2ZjM4YTkgPSBMLnRpbGVMYXllcigKICAgICAgICAnaHR0cHM6Ly97c30udGlsZS5vcGVuc3RyZWV0bWFwLm9yZy97en0ve3h9L3t5fS5wbmcnLAogICAgICAgIHsKICAgICAgICAiYXR0cmlidXRpb24iOiBudWxsLAogICAgICAgICJkZXRlY3RSZXRpbmEiOiBmYWxzZSwKICAgICAgICAibWF4TmF0aXZlWm9vbSI6IDE4LAogICAgICAgICJtYXhab29tIjogMTgsCiAgICAgICAgIm1pblpvb20iOiAwLAogICAgICAgICJub1dyYXAiOiBmYWxzZSwKICAgICAgICAib3BhY2l0eSI6IDEsCiAgICAgICAgInN1YmRvbWFpbnMiOiAiYWJjIiwKICAgICAgICAidG1zIjogZmFsc2UKfSkuYWRkVG8obWFwXzAyMzdjOTQxNzdhMzQ4ZmY5MzQzMWQxYTg4NmQwMWUxKTsKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81MTA1MDRlYWNkYWY0MjgxYjliNjY2YjQyODcyNmI3MyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEyLjk5LCA4MC4yXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMzZDg0YWQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjM2Q4NGFkIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNjAzMTE1M2RkZGI1NDgyNTk1NWZiOWQ1NmNhZTQ3MjQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOWEyZDZjYjM3MTJjNGZhYmI2YWNlZDk1YWE2MzNhZDEgPSAkKGA8ZGl2IGlkPSJodG1sXzlhMmQ2Y2IzNzEyYzRmYWJiNmFjZWQ5NWFhNjMzYWQxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5BZGFtYmFra2FtIENsdXN0ZXIgMS4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82MDMxMTUzZGRkYjU0ODI1OTU1ZmI5ZDU2Y2FlNDcyNC5zZXRDb250ZW50KGh0bWxfOWEyZDZjYjM3MTJjNGZhYmI2YWNlZDk1YWE2MzNhZDEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNTEwNTA0ZWFjZGFmNDI4MWI5YjY2NmI0Mjg3MjZiNzMuYmluZFBvcHVwKHBvcHVwXzYwMzExNTNkZGRiNTQ4MjU5NTVmYjlkNTZjYWU0NzI0KQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xY2Y1ZTRjYTBlNmU0YTQxYWVlZmY2NWEzNTcwMzM5OCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEzLjAwNjMsIDgwLjI1NzRdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8wMWEzMTI0MjdlODg0NDIxYjYzNjliNTJlNzZkMzAxMCA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lZjY2OTZiNmFmOTk0NmIzYThiMTMxZjVjZmJiNTk1OCA9ICQoYDxkaXYgaWQ9Imh0bWxfZWY2Njk2YjZhZjk5NDZiM2E4YjEzMWY1Y2ZiYjU5NTgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkFkeWFyIENsdXN0ZXIgMC4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wMWEzMTI0MjdlODg0NDIxYjYzNjliNTJlNzZkMzAxMC5zZXRDb250ZW50KGh0bWxfZWY2Njk2YjZhZjk5NDZiM2E4YjEzMWY1Y2ZiYjU5NTgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMWNmNWU0Y2EwZTZlNGE0MWFlZWZmNjVhMzU3MDMzOTguYmluZFBvcHVwKHBvcHVwXzAxYTMxMjQyN2U4ODQ0MjFiNjM2OWI1MmU3NmQzMDEwKQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kZDE0NDM1ZjJmNmU0MzE5OTQzODk0ZTc0MGI5ZDk2MSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEzLjAwMywgODAuMjA0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMzZDg0YWQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjM2Q4NGFkIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNjVhMzNmZjY3NDhmNGQ3NmEzNjlhYTQ0MDIxNDJmMmEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYjdhY2YyNGQ2NDA4NDc3YmIzMjdhMWU2MWE2MzZjYzUgPSAkKGA8ZGl2IGlkPSJodG1sX2I3YWNmMjRkNjQwODQ3N2JiMzI3YTFlNjFhNjM2Y2M1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5BbGFuZHVyIENsdXN0ZXIgMS4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82NWEzM2ZmNjc0OGY0ZDc2YTM2OWFhNDQwMjE0MmYyYS5zZXRDb250ZW50KGh0bWxfYjdhY2YyNGQ2NDA4NDc3YmIzMjdhMWU2MWE2MzZjYzUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZGQxNDQzNWYyZjZlNDMxOTk0Mzg5NGU3NDBiOWQ5NjEuYmluZFBvcHVwKHBvcHVwXzY1YTMzZmY2NzQ4ZjRkNzZhMzY5YWE0NDAyMTQyZjJhKQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81OTE4ODkyZGQxMzc0Yjc1YmExOWY0MzkwZmI3NDIzNCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEzLjA0OSwgODAuMTY3M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICAuYWRkVG8obWFwXzAyMzdjOTQxNzdhMzQ4ZmY5MzQzMWQxYTg4NmQwMWUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzc0ZTU4N2VhYzM5YTQxNjk4NWUzNDllY2RlYzI4MzI5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICcxMDAlJwogICAgICAgICAgICAKICAgICAgICAgICAgfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzMzMDA0MDVlOWQzMTRlMjk4M2FlNzk4NjJmMjBkMmRiID0gJChgPGRpdiBpZD0iaHRtbF8zMzAwNDA1ZTlkMzE0ZTI5ODNhZTc5ODYyZjIwZDJkYiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QWxhcGFra2FtIENsdXN0ZXIgMC4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83NGU1ODdlYWMzOWE0MTY5ODVlMzQ5ZWNkZWMyODMyOS5zZXRDb250ZW50KGh0bWxfMzMwMDQwNWU5ZDMxNGUyOTgzYWU3OTg2MmYyMGQyZGIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNTkxODg5MmRkMTM3NGI3NWJhMTlmNDM5MGZiNzQyMzQuYmluZFBvcHVwKHBvcHVwXzc0ZTU4N2VhYzM5YTQxNjk4NWUzNDllY2RlYzI4MzI5KQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81NmZiN2EzZDNkYjU0NzJhOTZmMjEzMWZmYjc5ZDAzZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEzLjAzMzksIDgwLjI0ODZdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lNGQyOWZkMTg4ZmU0ODk1ODI2OTIxMTZjMjQzOGVjOCA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80YmY1ZmM5NmQ3N2Y0NzUyYmU1MThjNDUzMDdlZmU3OSA9ICQoYDxkaXYgaWQ9Imh0bWxfNGJmNWZjOTZkNzdmNDc1MmJlNTE4YzQ1MzA3ZWZlNzkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkFsd2FycGV0IENsdXN0ZXIgMC4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lNGQyOWZkMTg4ZmU0ODk1ODI2OTIxMTZjMjQzOGVjOC5zZXRDb250ZW50KGh0bWxfNGJmNWZjOTZkNzdmNDc1MmJlNTE4YzQ1MzA3ZWZlNzkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNTZmYjdhM2QzZGI1NDcyYTk2ZjIxMzFmZmI3OWQwM2UuYmluZFBvcHVwKHBvcHVwX2U0ZDI5ZmQxODhmZTQ4OTU4MjY5MjExNmMyNDM4ZWM4KQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83NDljZjBkZTRmZWM0YmM5OTFjZjZkZDk1YTRkMWUzZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEzLjA0NzU0NSwgODAuMTg3MjkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMzZDg0YWQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjM2Q4NGFkIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNmJkNTQzYTQyZDc3NGFmNWJkZmE0YjRhNzkzNTc4MjMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYTIwZDkwZGJhYTllNDI1YzhlNDM0ZGFhN2VjOWVhMGMgPSAkKGA8ZGl2IGlkPSJodG1sX2EyMGQ5MGRiYWE5ZTQyNWM4ZTQzNGRhYTdlYzllYTBjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5BbHdhcnRoaXJ1bmFnYXIgQ2x1c3RlciAxLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzZiZDU0M2E0MmQ3NzRhZjViZGZhNGI0YTc5MzU3ODIzLnNldENvbnRlbnQoaHRtbF9hMjBkOTBkYmFhOWU0MjVjOGU0MzRkYWE3ZWM5ZWEwYyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83NDljZjBkZTRmZWM0YmM5OTFjZjZkZDk1YTRkMWUzZC5iaW5kUG9wdXAocG9wdXBfNmJkNTQzYTQyZDc3NGFmNWJkZmE0YjRhNzkzNTc4MjMpCiAgICAgICAgICAgIDsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2ZkMDRiNjVlODViNjRlYmU5NzMyOTg0NWUzYjg0NjRkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMTMuMDk4MywgODAuMTYyMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjM2Q4NGFkIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzNkODRhZCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICAuYWRkVG8obWFwXzAyMzdjOTQxNzdhMzQ4ZmY5MzQzMWQxYTg4NmQwMWUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzYxNTM1YjI4ZDY2MTQwOGI5NDkyNWYzNmU5YWRhNjI3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICcxMDAlJwogICAgICAgICAgICAKICAgICAgICAgICAgfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2MzZDM5ZTUyMTc1NTQyYTk4YjQwNTA5NTQ3MGJhOTI1ID0gJChgPGRpdiBpZD0iaHRtbF9jM2QzOWU1MjE3NTU0MmE5OGI0MDUwOTU0NzBiYTkyNSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QW1iYXR0dXIgQ2x1c3RlciAxLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzYxNTM1YjI4ZDY2MTQwOGI5NDkyNWYzNmU5YWRhNjI3LnNldENvbnRlbnQoaHRtbF9jM2QzOWU1MjE3NTU0MmE5OGI0MDUwOTU0NzBiYTkyNSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mZDA0YjY1ZTg1YjY0ZWJlOTczMjk4NDVlM2I4NDY0ZC5iaW5kUG9wdXAocG9wdXBfNjE1MzViMjhkNjYxNDA4Yjk0OTI1ZjM2ZTlhZGE2MjcpCiAgICAgICAgICAgIDsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzJiZWM0N2Q3Mzc5MjQ5YjZiYWMzNTJiZmM3NmRjZTc0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMTMuMDgzLCA4MC4yMzNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mNjk5MmUyNWZmMDc0MzIzOTcyOGM2ZGY0YmZhMmE0NSA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zZDE2YmJmZTcwZjY0NzAzOGNiNzMwODU5OTYyN2Y1YiA9ICQoYDxkaXYgaWQ9Imh0bWxfM2QxNmJiZmU3MGY2NDcwMzhjYjczMDg1OTk2MjdmNWIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkFtaW5qaWthcmFpIENsdXN0ZXIgMC4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mNjk5MmUyNWZmMDc0MzIzOTcyOGM2ZGY0YmZhMmE0NS5zZXRDb250ZW50KGh0bWxfM2QxNmJiZmU3MGY2NDcwMzhjYjczMDg1OTk2MjdmNWIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMmJlYzQ3ZDczNzkyNDliNmJhYzM1MmJmYzc2ZGNlNzQuYmluZFBvcHVwKHBvcHVwX2Y2OTkyZTI1ZmYwNzQzMjM5NzI4YzZkZjRiZmEyYTQ1KQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80NzllMTQ2Nzc3ZTI0MmUwOWE4M2MxZDhiMGZmNjE2OCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEzLjA4NDYsIDgwLjIxNzldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jMTAwOGExNDIyZjA0MWVlYmY3YTY5MmY5MGIzNTBlNCA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hNjM2YzFhYmFiYjU0MTMyODM5N2M1ZjljYmI4MDY2NyA9ICQoYDxkaXYgaWQ9Imh0bWxfYTYzNmMxYWJhYmI1NDEzMjgzOTdjNWY5Y2JiODA2NjciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkFubmEgTmFnYXIgQ2x1c3RlciAwLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2MxMDA4YTE0MjJmMDQxZWViZjdhNjkyZjkwYjM1MGU0LnNldENvbnRlbnQoaHRtbF9hNjM2YzFhYmFiYjU0MTMyODM5N2M1ZjljYmI4MDY2Nyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80NzllMTQ2Nzc3ZTI0MmUwOWE4M2MxZDhiMGZmNjE2OC5iaW5kUG9wdXAocG9wdXBfYzEwMDhhMTQyMmYwNDFlZWJmN2E2OTJmOTBiMzUwZTQpCiAgICAgICAgICAgIDsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzUyZTJkNDEwMjk2YjRhMzg5YjY2NjU4ZWQ1YmJmOGRmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMTMuMDczNTQsIDgwLjIwNjM5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMzZDg0YWQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjM2Q4NGFkIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZWZhNWUwMDM0YTA4NDFmM2E1ZDI3YThiZGFhOTc0YTIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZjUyYzJiODM3NjQzNDlkMWE1MTY2YWQ3YmQ2ZTNjY2MgPSAkKGA8ZGl2IGlkPSJodG1sX2Y1MmMyYjgzNzY0MzQ5ZDFhNTE2NmFkN2JkNmUzY2NjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5BcnVtYmFra2FtIENsdXN0ZXIgMS4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lZmE1ZTAwMzRhMDg0MWYzYTVkMjdhOGJkYWE5NzRhMi5zZXRDb250ZW50KGh0bWxfZjUyYzJiODM3NjQzNDlkMWE1MTY2YWQ3YmQ2ZTNjY2MpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNTJlMmQ0MTAyOTZiNGEzODliNjY2NThlZDViYmY4ZGYuYmluZFBvcHVwKHBvcHVwX2VmYTVlMDAzNGEwODQxZjNhNWQyN2E4YmRhYTk3NGEyKQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jYWI2NjU2ODNmMzQ0NGNmOTY4OWYyZTY3MjE1NWViNCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEzLjAzNTEsIDgwLjIwOTVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzNkODRhZCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzZDg0YWQiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iZjFmNzc3ZmEyOTM0M2VkYmJmZThkNmQyMDIwMDA5NyA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84NWExNTk3Mzg2ZTY0MjIzOTc5NzMyMWIxNTE1MDUyZSA9ICQoYDxkaXYgaWQ9Imh0bWxfODVhMTU5NzM4NmU2NDIyMzk3OTczMjFiMTUxNTA1MmUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkFzaG9rIE5hZ2FyIENsdXN0ZXIgMS4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iZjFmNzc3ZmEyOTM0M2VkYmJmZThkNmQyMDIwMDA5Ny5zZXRDb250ZW50KGh0bWxfODVhMTU5NzM4NmU2NDIyMzk3OTczMjFiMTUxNTA1MmUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfY2FiNjY1NjgzZjM0NDRjZjk2ODlmMmU2NzIxNTVlYjQuYmluZFBvcHVwKHBvcHVwX2JmMWY3NzdmYTI5MzQzZWRiYmZlOGQ2ZDIwMjAwMDk3KQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83MjUxZGMxNjE0YmU0MWU1YTg0OTY1YmQ2YTRmMjFhNyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEzLjEyLCA4MC4xXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZTMxNjAxNGY5MjJiNGQ0NThiZGY1MWU2OGRmZWE5NTYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNWFmZGZmMDY0ZWVkNGNkN2EwNjk4YjQxMmU1YThjNmYgPSAkKGA8ZGl2IGlkPSJodG1sXzVhZmRmZjA2NGVlZDRjZDdhMDY5OGI0MTJlNWE4YzZmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5BdmFkaSBDbHVzdGVyIDAuMDwvZGl2PmApWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZTMxNjAxNGY5MjJiNGQ0NThiZGY1MWU2OGRmZWE5NTYuc2V0Q29udGVudChodG1sXzVhZmRmZjA2NGVlZDRjZDdhMDY5OGI0MTJlNWE4YzZmKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzcyNTFkYzE2MTRiZTQxZTVhODQ5NjViZDZhNGYyMWE3LmJpbmRQb3B1cChwb3B1cF9lMzE2MDE0ZjkyMmI0ZDQ1OGJkZjUxZTY4ZGZlYTk1NikKICAgICAgICAgICAgOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMzExZGNlYjgyMzY4NDk3OTkxNzdkNjgzYjA4ZTlkMjAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFsxMy4xLCA4MC4yMzNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzNkODRhZCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzZDg0YWQiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iYTgxMzNlMzI1N2E0NDUzYmJlNGIwODAwZTI4MjMxNCA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lMzQ2MGZlOTFkODE0NmRjYmQ3YmU2MGQxMGY1NWI2MSA9ICQoYDxkaXYgaWQ9Imh0bWxfZTM0NjBmZTkxZDgxNDZkY2JkN2JlNjBkMTBmNTViNjEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkF5YW5hdmFyYW0gQ2x1c3RlciAxLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2JhODEzM2UzMjU3YTQ0NTNiYmU0YjA4MDBlMjgyMzE0LnNldENvbnRlbnQoaHRtbF9lMzQ2MGZlOTFkODE0NmRjYmQ3YmU2MGQxMGY1NWI2MSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zMTFkY2ViODIzNjg0OTc5OTE3N2Q2ODNiMDhlOWQyMC5iaW5kUG9wdXAocG9wdXBfYmE4MTMzZTMyNTdhNDQ1M2JiZTRiMDgwMGUyODIzMTQpCiAgICAgICAgICAgIDsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzI3Nzc2MmM5Y2U1ZjQ3NDZiZjFiYzU3YWQ2ZWNhY2E5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMTMuMDAwMiwgODAuMjY2OF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICAuYWRkVG8obWFwXzAyMzdjOTQxNzdhMzQ4ZmY5MzQzMWQxYTg4NmQwMWUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzMzOTQ3MzBhYTYzODRhOTQ5NDE0ZDk2YzdmNTMyY2MxID0gTC5wb3B1cCh7bWF4V2lkdGg6ICcxMDAlJwogICAgICAgICAgICAKICAgICAgICAgICAgfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzY3MzI2NjU5MTcyMDQ2YTc4ZmVhODk1OGE1N2U0ZjI1ID0gJChgPGRpdiBpZD0iaHRtbF82NzMyNjY1OTE3MjA0NmE3OGZlYTg5NThhNTdlNGYyNSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QmVzYW50IE5hZ2FyIENsdXN0ZXIgMC4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zMzk0NzMwYWE2Mzg0YTk0OTQxNGQ5NmM3ZjUzMmNjMS5zZXRDb250ZW50KGh0bWxfNjczMjY2NTkxNzIwNDZhNzhmZWE4OTU4YTU3ZTRmMjUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMjc3NzYyYzljZTVmNDc0NmJmMWJjNTdhZDZlY2FjYTkuYmluZFBvcHVwKHBvcHVwXzMzOTQ3MzBhYTYzODRhOTQ5NDE0ZDk2YzdmNTMyY2MxKQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yMWI4ZGQyOTVmODk0NWQ2YmUwNTlkYTMzMWVhNjVmNyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEzLjEwMjMsIDgwLjI3MTQzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMzZDg0YWQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjM2Q4NGFkIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMWI5NTBlYzQ4ZmFlNGM1YTk1M2Q1YTg0MjdlYWM0MTQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfODlmMDg0ODliZGE2NGE3NWFiNjczZmZiMjI3OWI3OTIgPSAkKGA8ZGl2IGlkPSJodG1sXzg5ZjA4NDg5YmRhNjRhNzVhYjY3M2ZmYjIyNzliNzkyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5CYXNpbiBCcmlkZ2UgQ2x1c3RlciAxLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzFiOTUwZWM0OGZhZTRjNWE5NTNkNWE4NDI3ZWFjNDE0LnNldENvbnRlbnQoaHRtbF84OWYwODQ4OWJkYTY0YTc1YWI2NzNmZmIyMjc5Yjc5Mik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yMWI4ZGQyOTVmODk0NWQ2YmUwNTlkYTMzMWVhNjVmNy5iaW5kUG9wdXAocG9wdXBfMWI5NTBlYzQ4ZmFlNGM1YTk1M2Q1YTg0MjdlYWM0MTQpCiAgICAgICAgICAgIDsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzFjOTQxZmYxNzRlYjRlYWZhZmE4NTIzODY4Yzg4N2VjID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMTMuMDYxNywgODAuMjgwNF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICAuYWRkVG8obWFwXzAyMzdjOTQxNzdhMzQ4ZmY5MzQzMWQxYTg4NmQwMWUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzA2OTAwNGRhYzY1NjQ2OGU4NTZhNzJmNmFjZmI2YjczID0gTC5wb3B1cCh7bWF4V2lkdGg6ICcxMDAlJwogICAgICAgICAgICAKICAgICAgICAgICAgfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2RkNDkyYTY5MTAxZDRlZGRhMGYyODEzNDU2YTc2MDg1ID0gJChgPGRpdiBpZD0iaHRtbF9kZDQ5MmE2OTEwMWQ0ZWRkYTBmMjgxMzQ1NmE3NjA4NSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2hlcGF1ayBDbHVzdGVyIDAuMDwvZGl2PmApWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMDY5MDA0ZGFjNjU2NDY4ZTg1NmE3MmY2YWNmYjZiNzMuc2V0Q29udGVudChodG1sX2RkNDkyYTY5MTAxZDRlZGRhMGYyODEzNDU2YTc2MDg1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzFjOTQxZmYxNzRlYjRlYWZhZmE4NTIzODY4Yzg4N2VjLmJpbmRQb3B1cChwb3B1cF8wNjkwMDRkYWM2NTY0NjhlODU2YTcyZjZhY2ZiNmI3MykKICAgICAgICAgICAgOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYmQyODY0YmQ2NmMxNDA1NGIxNTg4ZjAxYzZhOWIxNTQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFsxMy4wNzQxMiwgODAuMjQyMzhdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmZmYwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZmZmMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lZjkyNTk3OTM2YTU0ODdmODkyYzI0YTA2MDMzN2U5MCA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jMGM0MGZiMjQ2YTI0ZThlOTU2M2JmOTliZmVjMTk2MiA9ICQoYDxkaXYgaWQ9Imh0bWxfYzBjNDBmYjI0NmEyNGU4ZTk1NjNiZjk5YmZlYzE5NjIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNoZXRwdXQgQ2x1c3RlciAzLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2VmOTI1OTc5MzZhNTQ4N2Y4OTJjMjRhMDYwMzM3ZTkwLnNldENvbnRlbnQoaHRtbF9jMGM0MGZiMjQ2YTI0ZThlOTU2M2JmOTliZmVjMTk2Mik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9iZDI4NjRiZDY2YzE0MDU0YjE1ODhmMDFjNmE5YjE1NC5iaW5kUG9wdXAocG9wdXBfZWY5MjU5NzkzNmE1NDg3Zjg5MmMyNGEwNjAzMzdlOTApCiAgICAgICAgICAgIDsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzVlZDNiNmM1MGVjNzQ1OTU5MjM4ZWJiOWE0ZTI1NzNlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMTMuMDY3LCA4MC4yNjddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85YzcwYTVkZWY0MTk0N2E4OWY1NDlmMzcwNzBmNDFiMiA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hOWFkYTE1NGRmNzY0ZmQ0OGNkYjUwNTZhYWZmNDc3MSA9ICQoYDxkaXYgaWQ9Imh0bWxfYTlhZGExNTRkZjc2NGZkNDhjZGI1MDU2YWFmZjQ3NzEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNoaW50YWRyaXBldCBDbHVzdGVyIDAuMDwvZGl2PmApWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOWM3MGE1ZGVmNDE5NDdhODlmNTQ5ZjM3MDcwZjQxYjIuc2V0Q29udGVudChodG1sX2E5YWRhMTU0ZGY3NjRmZDQ4Y2RiNTA1NmFhZmY0NzcxKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzVlZDNiNmM1MGVjNzQ1OTU5MjM4ZWJiOWE0ZTI1NzNlLmJpbmRQb3B1cChwb3B1cF85YzcwYTVkZWY0MTk0N2E4OWY1NDlmMzcwNzBmNDFiMikKICAgICAgICAgICAgOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYzc0ZDU1NGRlODg2NGZmNDhkMWYxZTgyY2VkNDE2ZjcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFsxMi45MzczOSwgODAuMTM4NzRdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzNkODRhZCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzZDg0YWQiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82ZTY3NjkzNmYwMDc0MzEzYTdkODVhZDI5NWE1OGMzNSA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lMzlhMzM2MDIxZmU0MmI1OGYyNWU5OWE0MDgyNjQ2NyA9ICQoYDxkaXYgaWQ9Imh0bWxfZTM5YTMzNjAyMWZlNDJiNThmMjVlOTlhNDA4MjY0NjciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNoaXRsYXBha2thbSBDbHVzdGVyIDEuMDwvZGl2PmApWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNmU2NzY5MzZmMDA3NDMxM2E3ZDg1YWQyOTVhNThjMzUuc2V0Q29udGVudChodG1sX2UzOWEzMzYwMjFmZTQyYjU4ZjI1ZTk5YTQwODI2NDY3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2M3NGQ1NTRkZTg4NjRmZjQ4ZDFmMWU4MmNlZDQxNmY3LmJpbmRQb3B1cChwb3B1cF82ZTY3NjkzNmYwMDc0MzEzYTdkODVhZDI5NWE1OGMzNSkKICAgICAgICAgICAgOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNDAwMjU4Mzg4YmQwNDJiYmIzMTM1YzJiOGJiY2Y3YmYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFsxMy4wNjI4LCA4MC4yMjc1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMzZDg0YWQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjM2Q4NGFkIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYTIyZjYzM2ZjYjkzNDM2MDkzM2U4MjMxYWJiNmE1ZWQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYTU4Mzk1MTNjZGRmNGEwYjkyMjc1ZTgzNTcxYzUzNmMgPSAkKGA8ZGl2IGlkPSJodG1sX2E1ODM5NTEzY2RkZjRhMGI5MjI3NWU4MzU3MWM1MzZjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DaG9vbGFpbWVkdSBDbHVzdGVyIDEuMDwvZGl2PmApWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYTIyZjYzM2ZjYjkzNDM2MDkzM2U4MjMxYWJiNmE1ZWQuc2V0Q29udGVudChodG1sX2E1ODM5NTEzY2RkZjRhMGI5MjI3NWU4MzU3MWM1MzZjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzQwMDI1ODM4OGJkMDQyYmJiMzEzNWMyYjhiYmNmN2JmLmJpbmRQb3B1cChwb3B1cF9hMjJmNjMzZmNiOTM0MzYwOTMzZTgyMzFhYmI2YTVlZCkKICAgICAgICAgICAgOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOTExZWJiNGZkYTVhNDIwMjg0ZDJjYzY3ODEwYjEwZTggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFsxMi45NTE2MSwgODAuMTQwOTddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83ZDA4YThiMTY0NGU0M2E2YjJhYzkxMzNhYTM3MGY3NyA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iMDFhMTRiOGRhZDQ0YzQ2OTE5NjJmZGVmZDFlY2I0MiA9ICQoYDxkaXYgaWQ9Imh0bWxfYjAxYTE0YjhkYWQ0NGM0NjkxOTYyZmRlZmQxZWNiNDIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNocm9tcGV0IENsdXN0ZXIgMC4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83ZDA4YThiMTY0NGU0M2E2YjJhYzkxMzNhYTM3MGY3Ny5zZXRDb250ZW50KGh0bWxfYjAxYTE0YjhkYWQ0NGM0NjkxOTYyZmRlZmQxZWNiNDIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOTExZWJiNGZkYTVhNDIwMjg0ZDJjYzY3ODEwYjEwZTguYmluZFBvcHVwKHBvcHVwXzdkMDhhOGIxNjQ0ZTQzYTZiMmFjOTEzM2FhMzcwZjc3KQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wMWZiMTg4ODFlMTM0MGRlYmIwYTFmZTczNzc0OGFiOCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEzLjA3OCwgODAuMjU5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMTA2YjI3ZWJlZjQwNDkxZmE0MTVkMTkwNmRlNjZhMWIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYmI5YmFjZDRhZWU4NDMxMjhlYzI4ZDYzMTM1OWQyNGQgPSAkKGA8ZGl2IGlkPSJodG1sX2JiOWJhY2Q0YWVlODQzMTI4ZWMyOGQ2MzEzNTlkMjRkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5FZ21vcmUgQ2x1c3RlciAwLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzEwNmIyN2ViZWY0MDQ5MWZhNDE1ZDE5MDZkZTY2YTFiLnNldENvbnRlbnQoaHRtbF9iYjliYWNkNGFlZTg0MzEyOGVjMjhkNjMxMzU5ZDI0ZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wMWZiMTg4ODFlMTM0MGRlYmIwYTFmZTczNzc0OGFiOC5iaW5kUG9wdXAocG9wdXBfMTA2YjI3ZWJlZjQwNDkxZmE0MTVkMTkwNmRlNjZhMWIpCiAgICAgICAgICAgIDsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2E5YzFhNjJhYzMxZDRiMzRiYzVjZDRjZTBiOWRmYWQ2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMTMuMDIzODksIDgwLjIwMDA1Nl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjM2Q4NGFkIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzNkODRhZCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICAuYWRkVG8obWFwXzAyMzdjOTQxNzdhMzQ4ZmY5MzQzMWQxYTg4NmQwMWUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzVlZWMwNjdhMjI3MTRlMGZhZWQ4NTQ5NzhlZTU5MWE5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICcxMDAlJwogICAgICAgICAgICAKICAgICAgICAgICAgfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzc4NTUwNTE0MzAzYzQxNWM4YmY0MTIwZWY5NjJiZTY3ID0gJChgPGRpdiBpZD0iaHRtbF83ODU1MDUxNDMwM2M0MTVjOGJmNDEyMGVmOTYyYmU2NyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RWtrYWR1dGhhbmdhbCBDbHVzdGVyIDEuMDwvZGl2PmApWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNWVlYzA2N2EyMjcxNGUwZmFlZDg1NDk3OGVlNTkxYTkuc2V0Q29udGVudChodG1sXzc4NTUwNTE0MzAzYzQxNWM4YmY0MTIwZWY5NjJiZTY3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2E5YzFhNjJhYzMxZDRiMzRiYzVjZDRjZTBiOWRmYWQ2LmJpbmRQb3B1cChwb3B1cF81ZWVjMDY3YTIyNzE0ZTBmYWVkODU0OTc4ZWU1OTFhOSkKICAgICAgICAgICAgOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNGNhYjAzYTAxZTg1NDE4OGIyYzA5NjgzMTgwZDJiZDYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFsxMy4yMTc1LCA4MC4zMjE1NV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmZmZjAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmZmYwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICAuYWRkVG8obWFwXzAyMzdjOTQxNzdhMzQ4ZmY5MzQzMWQxYTg4NmQwMWUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzQwOGE1MmZjZTRlZjQwYjQ4MTU1YTc5YjJiMjFmNGY1ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICcxMDAlJwogICAgICAgICAgICAKICAgICAgICAgICAgfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2JjNGFlODY0ZmU0ZjQzYzc5ZWY2NmUwNzY0YjQxOTk0ID0gJChgPGRpdiBpZD0iaHRtbF9iYzRhZTg2NGZlNGY0M2M3OWVmNjZlMDc2NGI0MTk5NCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RW5ub3JlIENsdXN0ZXIgMy4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80MDhhNTJmY2U0ZWY0MGI0ODE1NWE3OWIyYjIxZjRmNS5zZXRDb250ZW50KGh0bWxfYmM0YWU4NjRmZTRmNDNjNzllZjY2ZTA3NjRiNDE5OTQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNGNhYjAzYTAxZTg1NDE4OGIyYzA5NjgzMTgwZDJiZDYuYmluZFBvcHVwKHBvcHVwXzQwOGE1MmZjZTRlZjQwYjQ4MTU1YTc5YjJiMjFmNGY1KQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82MzhhYzdlZmY2MmY0N2Y3Yjk5YTAzNzg2NDRkNjMwNSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEzLjAyMjMsIDgwLjI3NjNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzNkODRhZCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzZDg0YWQiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hYmM3N2ViNWZiZWU0NzMyYmU2NTU2ZjZmZDgwMDliYSA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8xYzYwMGZlZTEwMzU0ZWVhYjE5NjllNTFiOWMxMzEyZSA9ICQoYDxkaXYgaWQ9Imh0bWxfMWM2MDBmZWUxMDM1NGVlYWIxOTY5ZTUxYjljMTMxMmUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkZvcmVzaG9yZSBFc3RhdGUgQ2x1c3RlciAxLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2FiYzc3ZWI1ZmJlZTQ3MzJiZTY1NTZmNmZkODAwOWJhLnNldENvbnRlbnQoaHRtbF8xYzYwMGZlZTEwMzU0ZWVhYjE5NjllNTFiOWMxMzEyZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82MzhhYzdlZmY2MmY0N2Y3Yjk5YTAzNzg2NDRkNjMwNS5iaW5kUG9wdXAocG9wdXBfYWJjNzdlYjVmYmVlNDczMmJlNjU1NmY2ZmQ4MDA5YmEpCiAgICAgICAgICAgIDsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2ExZjlkNmRhYzE1OTRmZGJhNTI2YWRjZTBiM2EyODQyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMTMuMDc5NzIyLCA4MC4yODY5NDRdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzNkODRhZCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzZDg0YWQiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81MjA3Mzc3YzE0Zjg0OWJkYmU5ZDlkY2QzYzg4NTc3MiA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yZjU4YTAwYTZmYWI0OTgyODdmNTFlNWNmNTgyYjA0NyA9ICQoYDxkaXYgaWQ9Imh0bWxfMmY1OGEwMGE2ZmFiNDk4Mjg3ZjUxZTVjZjU4MmIwNDciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkZvcnQgU3QuIEdlb3JnZSBDbHVzdGVyIDEuMDwvZGl2PmApWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNTIwNzM3N2MxNGY4NDliZGJlOWQ5ZGNkM2M4ODU3NzIuc2V0Q29udGVudChodG1sXzJmNThhMDBhNmZhYjQ5ODI4N2Y1MWU1Y2Y1ODJiMDQ3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2ExZjlkNmRhYzE1OTRmZGJhNTI2YWRjZTBiM2EyODQyLmJpbmRQb3B1cChwb3B1cF81MjA3Mzc3YzE0Zjg0OWJkYmU5ZDlkY2QzYzg4NTc3MikKICAgICAgICAgICAgOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOTkwOGRjMjIzZDJjNDJmYWExOTQ1ODQ0MGNmNjk1NGEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFsxMy4wOTM5LCA4MC4yODM5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZmZmMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmZmZjAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNDQ1ODgxMDllMWNiNDAyMmFmYjE5ZDM0MTkyY2RhY2QgPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYWJmMTEwY2MxMjI4NGRhZWJmMzUzNDFiOWY1NmM0YTQgPSAkKGA8ZGl2IGlkPSJodG1sX2FiZjExMGNjMTIyODRkYWViZjM1MzQxYjlmNTZjNGE0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5HZW9yZ2UgVG93biBDbHVzdGVyIDMuMDwvZGl2PmApWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNDQ1ODgxMDllMWNiNDAyMmFmYjE5ZDM0MTkyY2RhY2Quc2V0Q29udGVudChodG1sX2FiZjExMGNjMTIyODRkYWViZjM1MzQxYjlmNTZjNGE0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzk5MDhkYzIyM2QyYzQyZmFhMTk0NTg0NDBjZjY5NTRhLmJpbmRQb3B1cChwb3B1cF80NDU4ODEwOWUxY2I0MDIyYWZiMTlkMzQxOTJjZGFjZCkKICAgICAgICAgICAgOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMmQyNzIzZmJlM2Q2NDQ0YWFkOTNmNDk1MjdiODFlMjggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFsxMy4wNSwgODAuMjU4Ml0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICAuYWRkVG8obWFwXzAyMzdjOTQxNzdhMzQ4ZmY5MzQzMWQxYTg4NmQwMWUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2Q1YzNhNjc5NzY2ZjQ2YmU5NDUwYWUzNWJiM2NkMTQ3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICcxMDAlJwogICAgICAgICAgICAKICAgICAgICAgICAgfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzZjYWJiMDg0NjRlZTQ4N2E4NTA0MmJiY2RhODI0MDlmID0gJChgPGRpdiBpZD0iaHRtbF82Y2FiYjA4NDY0ZWU0ODdhODUwNDJiYmNkYTgyNDA5ZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+R29wYWxhcHVyYW0gQ2x1c3RlciAwLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2Q1YzNhNjc5NzY2ZjQ2YmU5NDUwYWUzNWJiM2NkMTQ3LnNldENvbnRlbnQoaHRtbF82Y2FiYjA4NDY0ZWU0ODdhODUwNDJiYmNkYTgyNDA5Zik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yZDI3MjNmYmUzZDY0NDRhYWQ5M2Y0OTUyN2I4MWUyOC5iaW5kUG9wdXAocG9wdXBfZDVjM2E2Nzk3NjZmNDZiZTk0NTBhZTM1YmIzY2QxNDcpCiAgICAgICAgICAgIDsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzY0MjkyNWJiY2UyYzQyYTRhYWIwYTJhYjgxY2Q0MTI0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMTMuMDgxNTM5LCA4MC4yODU3MThdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmZmYwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZmZmMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81Y2I4YWY5NDM4Yzk0YzUxOWNmYzFkZTdkMmRiZDg4OSA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83MGE2ODY0NTk0MTI0Mzc3Yjc1MzRlY2Q1MjkxY2MzZiA9ICQoYDxkaXYgaWQ9Imh0bWxfNzBhNjg2NDU5NDEyNDM3N2I3NTM0ZWNkNTI5MWNjM2YiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkdvdmVybm1lbnQgRXN0YXRlIENsdXN0ZXIgMy4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81Y2I4YWY5NDM4Yzk0YzUxOWNmYzFkZTdkMmRiZDg4OS5zZXRDb250ZW50KGh0bWxfNzBhNjg2NDU5NDEyNDM3N2I3NTM0ZWNkNTI5MWNjM2YpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNjQyOTI1YmJjZTJjNDJhNGFhYjBhMmFiODFjZDQxMjQuYmluZFBvcHVwKHBvcHVwXzVjYjhhZjk0MzhjOTRjNTE5Y2ZjMWRlN2QyZGJkODg5KQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hZTg4YjdjNmQ3ZmI0OWYwYWIwZDU1OTY1N2VmNWZlYyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEzLjAwODQxMjUsIDgwLjIxMjY4NzVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kMzVhZDkyZmRlYTk0M2IzOTU0MjFiYTgzYWNlZDFmMSA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82ZjNkMzhjODBmNjQ0N2E1YjI4NThlZWQ5NTMyOTMyZCA9ICQoYDxkaXYgaWQ9Imh0bWxfNmYzZDM4YzgwZjY0NDdhNWIyODU4ZWVkOTUzMjkzMmQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkd1aW5keSBDbHVzdGVyIDAuMDwvZGl2PmApWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZDM1YWQ5MmZkZWE5NDNiMzk1NDIxYmE4M2FjZWQxZjEuc2V0Q29udGVudChodG1sXzZmM2QzOGM4MGY2NDQ3YTViMjg1OGVlZDk1MzI5MzJkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2FlODhiN2M2ZDdmYjQ5ZjBhYjBkNTU5NjU3ZWY1ZmVjLmJpbmRQb3B1cChwb3B1cF9kMzVhZDkyZmRlYTk0M2IzOTU0MjFiYTgzYWNlZDFmMSkKICAgICAgICAgICAgOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMGEzYjc1YjM2ZWM0NDNmZGI4ZWZhNjI4ZmVjNjk5NGIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFsxMy4wMTMzMDQsIDgwLjE0MzAzN10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjM2Q4NGFkIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzNkODRhZCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICAuYWRkVG8obWFwXzAyMzdjOTQxNzdhMzQ4ZmY5MzQzMWQxYTg4NmQwMWUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2I0MjVmOGYzYzliMDQ5MWI4YmMwZWNlODQ1OGM2OGM1ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICcxMDAlJwogICAgICAgICAgICAKICAgICAgICAgICAgfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzFmYjk5M2I3NGJhODQxYWM5MGY0Yzk5MzYxYjI4ZWI0ID0gJChgPGRpdiBpZD0iaHRtbF8xZmI5OTNiNzRiYTg0MWFjOTBmNGM5OTM2MWIyOGViNCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+R2VydWdhbWJha2thbSBDbHVzdGVyIDEuMDwvZGl2PmApWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYjQyNWY4ZjNjOWIwNDkxYjhiYzBlY2U4NDU4YzY4YzUuc2V0Q29udGVudChodG1sXzFmYjk5M2I3NGJhODQxYWM5MGY0Yzk5MzYxYjI4ZWI0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzBhM2I3NWIzNmVjNDQzZmRiOGVmYTYyOGZlYzY5OTRiLmJpbmRQb3B1cChwb3B1cF9iNDI1ZjhmM2M5YjA0OTFiOGJjMGVjZTg0NThjNjhjNSkKICAgICAgICAgICAgOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYWU3NWUwZGYzNmY2NDAyYThhODdmY2MyYTJmZDk3MTQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFsxMi45OTE1MSwgODAuMjMzNjJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xNzk2ZDNjMWZlYTY0NTZiYmE0Y2JmZDNhOWRmZjdkNiA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iNjc4N2I3MWU5MTc0YTBmYTEyNWRmNDA3MTUwYTI0OCA9ICQoYDxkaXYgaWQ9Imh0bWxfYjY3ODdiNzFlOTE3NGEwZmExMjVkZjQwNzE1MGEyNDgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPklJVCBNYWRyYXMgQ2x1c3RlciAwLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzE3OTZkM2MxZmVhNjQ1NmJiYTRjYmZkM2E5ZGZmN2Q2LnNldENvbnRlbnQoaHRtbF9iNjc4N2I3MWU5MTc0YTBmYTEyNWRmNDA3MTUwYTI0OCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hZTc1ZTBkZjM2ZjY0MDJhOGE4N2ZjYzJhMmZkOTcxNC5iaW5kUG9wdXAocG9wdXBfMTc5NmQzYzFmZWE2NDU2YmJhNGNiZmQzYTlkZmY3ZDYpCiAgICAgICAgICAgIDsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzI4OWVkNTZiOWVkNjRlZTc4NDBjNjk2MjgzODRiMzBkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMTIuOTE2MiwgODAuMjQ4OF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICAuYWRkVG8obWFwXzAyMzdjOTQxNzdhMzQ4ZmY5MzQzMWQxYTg4NmQwMWUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2FiMzA3MWQ5NDVhMjRiNjA5OTBmNDNkYTc4NDZkMTNiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICcxMDAlJwogICAgICAgICAgICAKICAgICAgICAgICAgfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzNlMjJhMGU0ZTQ5ZTQ5NTk5YWY3ZjkwYmMxYzAyNDljID0gJChgPGRpdiBpZD0iaHRtbF8zZTIyYTBlNGU0OWU0OTU5OWFmN2Y5MGJjMWMwMjQ5YyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SW5qYW1iYWtrYW0gQ2x1c3RlciAwLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2FiMzA3MWQ5NDVhMjRiNjA5OTBmNDNkYTc4NDZkMTNiLnNldENvbnRlbnQoaHRtbF8zZTIyYTBlNGU0OWU0OTU5OWFmN2Y5MGJjMWMwMjQ5Yyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yODllZDU2YjllZDY0ZWU3ODQwYzY5NjI4Mzg0YjMwZC5iaW5kUG9wdXAocG9wdXBfYWIzMDcxZDk0NWEyNGI2MDk5MGY0M2RhNzg0NmQxM2IpCiAgICAgICAgICAgIDsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2I3YWViMDI1MjhhNTRjYmJhZjg3MzllNGMwZGYyODY4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMTMuMDM3OTYsIDgwLjEzNTEzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMzZDg0YWQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjM2Q4NGFkIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZjU4MGRkOTZmOTc5NDcwYTk1MTlhYTNjYzY2ZTExNTggPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMWU0YmIxMGM0M2E5NDVkYzgzM2ViMDliNTM5MDNmMDMgPSAkKGA8ZGl2IGlkPSJodG1sXzFlNGJiMTBjNDNhOTQ1ZGM4MzNlYjA5YjUzOTAzZjAzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5JeXlhcGFudGhhbmdhbCBDbHVzdGVyIDEuMDwvZGl2PmApWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZjU4MGRkOTZmOTc5NDcwYTk1MTlhYTNjYzY2ZTExNTguc2V0Q29udGVudChodG1sXzFlNGJiMTBjNDNhOTQ1ZGM4MzNlYjA5YjUzOTAzZjAzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2I3YWViMDI1MjhhNTRjYmJhZjg3MzllNGMwZGYyODY4LmJpbmRQb3B1cChwb3B1cF9mNTgwZGQ5NmY5Nzk0NzBhOTUxOWFhM2NjNjZlMTE1OCkKICAgICAgICAgICAgOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNjI0ODU1ODgzODc3NGFhNWIzMGZlNGQwNzUwMjU0NTQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFsxMy4wMjM1LCA4MC4yMjM3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOWNiN2ExNjNkN2VmNGUzNTkwMGRlNmUzNjczNmI0NTAgPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYTlkYzdlYmYzNzZjNGM4ZWE5MTEyMmNjZjk1OTkyZTUgPSAkKGA8ZGl2IGlkPSJodG1sX2E5ZGM3ZWJmMzc2YzRjOGVhOTExMjJjY2Y5NTk5MmU1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5KYWZmZXJraGFucGV0IENsdXN0ZXIgMC4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85Y2I3YTE2M2Q3ZWY0ZTM1OTAwZGU2ZTM2NzM2YjQ1MC5zZXRDb250ZW50KGh0bWxfYTlkYzdlYmYzNzZjNGM4ZWE5MTEyMmNjZjk1OTkyZTUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNjI0ODU1ODgzODc3NGFhNWIzMGZlNGQwNzUwMjU0NTQuYmluZFBvcHVwKHBvcHVwXzljYjdhMTYzZDdlZjRlMzU5MDBkZTZlMzY3MzZiNDUwKQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lNzk3OWFmMTdlYWU0YWYyODcwMmRlNGU0ODVkOTZiYyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEyLjkxNDE3LCA4MC4yMjkzOF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICAuYWRkVG8obWFwXzAyMzdjOTQxNzdhMzQ4ZmY5MzQzMWQxYTg4NmQwMWUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2E2ODY2MWNhN2ZiODRmNmRhNDRlMGVhMDFiYjc5NjVjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICcxMDAlJwogICAgICAgICAgICAKICAgICAgICAgICAgfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2RlYTMxMzQ0YWIzZDQ1MTRhNTdlMmI5NTBmNWU4ZGIwID0gJChgPGRpdiBpZD0iaHRtbF9kZWEzMTM0NGFiM2Q0NTE0YTU3ZTJiOTUwZjVlOGRiMCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+S2FyYXBha2thbSBDbHVzdGVyIDAuMDwvZGl2PmApWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYTY4NjYxY2E3ZmI4NGY2ZGE0NGUwZWEwMWJiNzk2NWMuc2V0Q29udGVudChodG1sX2RlYTMxMzQ0YWIzZDQ1MTRhNTdlMmI5NTBmNWU4ZGIwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2U3OTc5YWYxN2VhZTRhZjI4NzAyZGU0ZTQ4NWQ5NmJjLmJpbmRQb3B1cChwb3B1cF9hNjg2NjFjYTdmYjg0ZjZkYTQ0ZTBlYTAxYmI3OTY1YykKICAgICAgICAgICAgOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTY2YTViMTI0NGU1NDRlZGFmNTEyMzFlZGQyNGY5YTkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFsxMy4yMDQ2MDIsIDgwLjMxNjc0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZmZmMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmZmZjAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZDJmZDU2MjNlZjUyNDViZDgwMmY1ZTU4NjlkMDBmZjMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfODhiNzZlYjkxYTZiNGMwYmFmZDZkOWNiOTY5NWJmNTEgPSAkKGA8ZGl2IGlkPSJodG1sXzg4Yjc2ZWI5MWE2YjRjMGJhZmQ2ZDljYjk2OTViZjUxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5LYXR0aXZha2thbSBDbHVzdGVyIDMuMDwvZGl2PmApWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZDJmZDU2MjNlZjUyNDViZDgwMmY1ZTU4NjlkMDBmZjMuc2V0Q29udGVudChodG1sXzg4Yjc2ZWI5MWE2YjRjMGJhZmQ2ZDljYjk2OTViZjUxKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzU2NmE1YjEyNDRlNTQ0ZWRhZjUxMjMxZWRkMjRmOWE5LmJpbmRQb3B1cChwb3B1cF9kMmZkNTYyM2VmNTI0NWJkODAyZjVlNTg2OWQwMGZmMykKICAgICAgICAgICAgOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYTM1YmE5ZGY4Mzg1NDA5NWJiM2Y5YTQxYzUxNGE3OWMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFsxMy4wNDAwNSwgODAuMTk5MjldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iNDcwM2VmOWVlMmQ0NWI4YmU4MGZkNDAxYTJkMzM3NyA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hMDRmMDI1NTE5YTg0N2YzOTc4NzRlOTU5Yzk5ZTM2YiA9ICQoYDxkaXYgaWQ9Imh0bWxfYTA0ZjAyNTUxOWE4NDdmMzk3ODc0ZTk1OWM5OWUzNmIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPksuSy4gTmFnYXIgQ2x1c3RlciAwLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2I0NzAzZWY5ZWUyZDQ1YjhiZTgwZmQ0MDFhMmQzMzc3LnNldENvbnRlbnQoaHRtbF9hMDRmMDI1NTE5YTg0N2YzOTc4NzRlOTU5Yzk5ZTM2Yik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hMzViYTlkZjgzODU0MDk1YmIzZjlhNDFjNTE0YTc5Yy5iaW5kUG9wdXAocG9wdXBfYjQ3MDNlZjllZTJkNDViOGJlODBmZDQwMWEyZDMzNzcpCiAgICAgICAgICAgIDsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzNhOWU1ODc0ZDgwNDRhMzY4OTI4MzBhM2Q0ODJmMmYyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMTIuOTY0ODcsIDgwLjE5NjExXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMzZDg0YWQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjM2Q4NGFkIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOTYxZDM5ZmYxM2IwNDMyYmIzNTNlODQzMjhjOTAzYzMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNTA3MTA0YmQxZGY3NGVkOGEwYjRmZjg3NDBiYzUzZjcgPSAkKGA8ZGl2IGlkPSJodG1sXzUwNzEwNGJkMWRmNzRlZDhhMGI0ZmY4NzQwYmM1M2Y3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5LZWVsa2F0dGFsYWkgQ2x1c3RlciAxLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzk2MWQzOWZmMTNiMDQzMmJiMzUzZTg0MzI4YzkwM2MzLnNldENvbnRlbnQoaHRtbF81MDcxMDRiZDFkZjc0ZWQ4YTBiNGZmODc0MGJjNTNmNyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zYTllNTg3NGQ4MDQ0YTM2ODkyODMwYTNkNDgyZjJmMi5iaW5kUG9wdXAocG9wdXBfOTYxZDM5ZmYxM2IwNDMyYmIzNTNlODQzMjhjOTAzYzMpCiAgICAgICAgICAgIDsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzU5NjRkOGZjZGQ4NzRlMmRhNmI2MWU5OGMwODk2NWMyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMTMuMDg1NiwgODAuMjM3OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjM2Q4NGFkIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzNkODRhZCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICAuYWRkVG8obWFwXzAyMzdjOTQxNzdhMzQ4ZmY5MzQzMWQxYTg4NmQwMWUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2Y1YTliZWIzYTU1NTRhMTBiYTU4MDQyZGEzMjA5ZjdjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICcxMDAlJwogICAgICAgICAgICAKICAgICAgICAgICAgfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2VmNjdmODg0YjYyMjRlNDQ5ODA3NzhjYTkzNTZlOGI2ID0gJChgPGRpdiBpZD0iaHRtbF9lZjY3Zjg4NGI2MjI0ZTQ0OTgwNzc4Y2E5MzU2ZThiNiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+S2lscGF1ayBDbHVzdGVyIDEuMDwvZGl2PmApWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZjVhOWJlYjNhNTU1NGExMGJhNTgwNDJkYTMyMDlmN2Muc2V0Q29udGVudChodG1sX2VmNjdmODg0YjYyMjRlNDQ5ODA3NzhjYTkzNTZlOGI2KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzU5NjRkOGZjZGQ4NzRlMmRhNmI2MWU5OGMwODk2NWMyLmJpbmRQb3B1cChwb3B1cF9mNWE5YmViM2E1NTU0YTEwYmE1ODA0MmRhMzIwOWY3YykKICAgICAgICAgICAgOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYTU4N2VkNTcxZmM2NDU1ZTlhN2U3OGQyYmFjYTkzYTYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFsxMy4wNDgxLCA4MC4yMjE0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZmNhZjlhMmQzZDczNGJjMjk3YjQxZjljZTQ2ZWI1YjIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZjgyN2RkY2RjMjA2NGMzODgyOGRlNDMyZjIyZGJjMTcgPSAkKGA8ZGl2IGlkPSJodG1sX2Y4MjdkZGNkYzIwNjRjMzg4MjhkZTQzMmYyMmRiYzE3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Lb2RhbWJha2thbSBDbHVzdGVyIDAuMDwvZGl2PmApWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZmNhZjlhMmQzZDczNGJjMjk3YjQxZjljZTQ2ZWI1YjIuc2V0Q29udGVudChodG1sX2Y4MjdkZGNkYzIwNjRjMzg4MjhkZTQzMmYyMmRiYzE3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2E1ODdlZDU3MWZjNjQ1NWU5YTdlNzhkMmJhY2E5M2E2LmJpbmRQb3B1cChwb3B1cF9mY2FmOWEyZDNkNzM0YmMyOTdiNDFmOWNlNDZlYjViMikKICAgICAgICAgICAgOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNzkwYjAyNjAzZDA4NGRmNTk4OTlmNWI3NTg4MzZjMmUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFsxMy4xMTcwMTQsIDgwLjIxMTQzOV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjMDAwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzAwMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICAuYWRkVG8obWFwXzAyMzdjOTQxNzdhMzQ4ZmY5MzQzMWQxYTg4NmQwMWUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzlhNzEzOTg1YjE5MjQzNDQ4OTI5MGJmZmQwMmRmYjIzID0gTC5wb3B1cCh7bWF4V2lkdGg6ICcxMDAlJwogICAgICAgICAgICAKICAgICAgICAgICAgfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzQyNWEzYjA4M2QyYzRiY2NhMGIxNTQ1MGI5NjY1ZWZlID0gJChgPGRpdiBpZD0iaHRtbF80MjVhM2IwODNkMmM0YmNjYTBiMTU0NTBiOTY2NWVmZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+S29sYXRodXIgQ2x1c3RlciAyLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzlhNzEzOTg1YjE5MjQzNDQ4OTI5MGJmZmQwMmRmYjIzLnNldENvbnRlbnQoaHRtbF80MjVhM2IwODNkMmM0YmNjYTBiMTU0NTBiOTY2NWVmZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83OTBiMDI2MDNkMDg0ZGY1OTg5OWY1Yjc1ODgzNmMyZS5iaW5kUG9wdXAocG9wdXBfOWE3MTM5ODViMTkyNDM0NDg5MjkwYmZmZDAyZGZiMjMpCiAgICAgICAgICAgIDsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzRhZWVlN2FhMGRhOTQ5ZmJiYTk2MDk1MzY5M2U3NmE3ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMTMuMTExNDk3LCA4MC4xODQ0MTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzNkODRhZCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzZDg0YWQiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jNmQxOWFjZDI1ZGQ0OTljOWE0NjdiYzc5ZWYyZDA0OSA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hNDVlYjg3MGVmNzk0OTIyOTRhYjY4YWIwY2YxM2IzNyA9ICQoYDxkaXYgaWQ9Imh0bWxfYTQ1ZWI4NzBlZjc5NDkyMjk0YWI2OGFiMGNmMTNiMzciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPktvcmF0dHVyIENsdXN0ZXIgMS4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9jNmQxOWFjZDI1ZGQ0OTljOWE0NjdiYzc5ZWYyZDA0OS5zZXRDb250ZW50KGh0bWxfYTQ1ZWI4NzBlZjc5NDkyMjk0YWI2OGFiMGNmMTNiMzcpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNGFlZWU3YWEwZGE5NDlmYmJhOTYwOTUzNjkzZTc2YTcuYmluZFBvcHVwKHBvcHVwX2M2ZDE5YWNkMjVkZDQ5OWM5YTQ2N2JjNzllZjJkMDQ5KQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83NTBlNzUzYjY3ZGM0NGVkODMxOWQ1NjNlYzBjZGZiNyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEzLjExNywgODAuMjgzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMzZDg0YWQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjM2Q4NGFkIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMTc3ZmU2NDNmYmU4NDZiNTk2MTdiNTY5YjYwMGM3NzEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZWU2MzQ2YTA0NjVlNDZhZWFhMjliMWIzNjIzOWUwNmEgPSAkKGA8ZGl2IGlkPSJodG1sX2VlNjM0NmEwNDY1ZTQ2YWVhYTI5YjFiMzYyMzllMDZhIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Lb3J1a2t1cGV0IENsdXN0ZXIgMS4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8xNzdmZTY0M2ZiZTg0NmI1OTYxN2I1NjliNjAwYzc3MS5zZXRDb250ZW50KGh0bWxfZWU2MzQ2YTA0NjVlNDZhZWFhMjliMWIzNjIzOWUwNmEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNzUwZTc1M2I2N2RjNDRlZDgzMTlkNTYzZWMwY2RmYjcuYmluZFBvcHVwKHBvcHVwXzE3N2ZlNjQzZmJlODQ2YjU5NjE3YjU2OWI2MDBjNzcxKQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hNmY1OWY0ZTRkYmY0MjI2OTg2MjUzYTM2YTQ4YzlhOCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEyLjk3LCA4MC4yNThdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzNkODRhZCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzZDg0YWQiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80MjIzNGNkNGFlMTU0YmY2OWRmYjIxZmUwNzM3NDBmMiA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lNWNkZGY1ZjFhNGQ0YmQ1OGI5ODNiNGY5N2Y2YjllNiA9ICQoYDxkaXYgaWQ9Imh0bWxfZTVjZGRmNWYxYTRkNGJkNThiOTgzYjRmOTdmNmI5ZTYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPktvdHRpdmFra2FtIENsdXN0ZXIgMS4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80MjIzNGNkNGFlMTU0YmY2OWRmYjIxZmUwNzM3NDBmMi5zZXRDb250ZW50KGh0bWxfZTVjZGRmNWYxYTRkNGJkNThiOTgzYjRmOTdmNmI5ZTYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYTZmNTlmNGU0ZGJmNDIyNjk4NjI1M2EzNmE0OGM5YTguYmluZFBvcHVwKHBvcHVwXzQyMjM0Y2Q0YWUxNTRiZjY5ZGZiMjFmZTA3Mzc0MGYyKQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yMjU0MjEzYTU5OTU0OGJjYTgwOWU2YmJhMGVkNGU0MSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEzLjAxOTYsIDgwLjI0NThdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzNkODRhZCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzZDg0YWQiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yYzM0NWJkNTdjYWY0MDI2YWFhZDJiYTEyNThiYTZkNyA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83M2ZlY2Q2NGIwYjQ0NTg2OGRlOGMwMzgxNWVjMWJkNCA9ICQoYDxkaXYgaWQ9Imh0bWxfNzNmZWNkNjRiMGI0NDU4NjhkZThjMDM4MTVlYzFiZDQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPktvdHR1cnB1cmFtIENsdXN0ZXIgMS4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yYzM0NWJkNTdjYWY0MDI2YWFhZDJiYTEyNThiYTZkNy5zZXRDb250ZW50KGh0bWxfNzNmZWNkNjRiMGI0NDU4NjhkZThjMDM4MTVlYzFiZDQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMjI1NDIxM2E1OTk1NDhiY2E4MDllNmJiYTBlZDRlNDEuYmluZFBvcHVwKHBvcHVwXzJjMzQ1YmQ1N2NhZjQwMjZhYWFkMmJhMTI1OGJhNmQ3KQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85ZTk1YjlhMjQ0MDg0YzcyYWM5YzExN2I1MTBmOTQzZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEyLjk0NTczNTcsIDgwLjIwMTQ1NDNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kNzExY2Y2YmU0NGE0MWVjYmQ3YWJhYzM2MDc1YTM3ZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iM2Y3ZmU5ZGZlZTM0ZjMzYmU0ZjQ2MjIxYTk3ZjcxMCA9ICQoYDxkaXYgaWQ9Imh0bWxfYjNmN2ZlOWRmZWUzNGYzM2JlNGY0NjIyMWE5N2Y3MTAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPktvdmlsYW1iYWtrYW0gQ2x1c3RlciAwLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2Q3MTFjZjZiZTQ0YTQxZWNiZDdhYmFjMzYwNzVhMzdlLnNldENvbnRlbnQoaHRtbF9iM2Y3ZmU5ZGZlZTM0ZjMzYmU0ZjQ2MjIxYTk3ZjcxMCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85ZTk1YjlhMjQ0MDg0YzcyYWM5YzExN2I1MTBmOTQzZS5iaW5kUG9wdXAocG9wdXBfZDcxMWNmNmJlNDRhNDFlY2JkN2FiYWMzNjA3NWEzN2UpCiAgICAgICAgICAgIDsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzcyNzlmZGQ5ZmI3NjQzNWFhNTVkNjM0MDc2NzYwMDc1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMTMuMDY5MzYyLCA4MC4xOTc0MjddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmZmYwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZmZmMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jYzk3MzU0MjlkMzE0ZDQxODZhYjA4OWQ3NzNjMWNlOSA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82NzU4MzBlOTM2Mjc0N2E3ODVhNjdiYjg3MTAyNzhmMSA9ICQoYDxkaXYgaWQ9Imh0bWxfNjc1ODMwZTkzNjI3NDdhNzg1YTY3YmI4NzEwMjc4ZjEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPktveWFtYmVkdSBDbHVzdGVyIDMuMDwvZGl2PmApWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfY2M5NzM1NDI5ZDMxNGQ0MTg2YWIwODlkNzczYzFjZTkuc2V0Q29udGVudChodG1sXzY3NTgzMGU5MzYyNzQ3YTc4NWE2N2JiODcxMDI3OGYxKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzcyNzlmZGQ5ZmI3NjQzNWFhNTVkNjM0MDc2NzYwMDc1LmJpbmRQb3B1cChwb3B1cF9jYzk3MzU0MjlkMzE0ZDQxODZhYjA4OWQ3NzNjMWNlOSkKICAgICAgICAgICAgOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMzhhMGMwMzJhNTRmNDBhMWI4YTFkMGQ1NDE1YTFiOTggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFsxMi45OTc0LCA4MC4wOTY2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZmZmMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmZmZjAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZjhmMzUwYWFlOTc4NDRjNzliYTM1N2VmZTg4NTEyN2EgPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOWE4YWQ2ZjhkZmM1NGE0NWE3ZWM2MzdjODBhNmU3NmUgPSAkKGA8ZGl2IGlkPSJodG1sXzlhOGFkNmY4ZGZjNTRhNDVhN2VjNjM3YzgwYTZlNzZlIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5LdW5kcmF0aHVyIENsdXN0ZXIgMy4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mOGYzNTBhYWU5Nzg0NGM3OWJhMzU3ZWZlODg1MTI3YS5zZXRDb250ZW50KGh0bWxfOWE4YWQ2ZjhkZmM1NGE0NWE3ZWM2MzdjODBhNmU3NmUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMzhhMGMwMzJhNTRmNDBhMWI4YTFkMGQ1NDE1YTFiOTguYmluZFBvcHVwKHBvcHVwX2Y4ZjM1MGFhZTk3ODQ0Yzc5YmEzNTdlZmU4ODUxMjdhKQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83ODdiZWI1YzYwYzc0ZmM4YTFhZGEwY2YwMjUzYjIxOSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEzLjE1LCA4MC4yNF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjM2Q4NGFkIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzNkODRhZCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICAuYWRkVG8obWFwXzAyMzdjOTQxNzdhMzQ4ZmY5MzQzMWQxYTg4NmQwMWUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzQwMjYyZTJkYTZiYTQ3Njk4NzFkODYzNDFhOTk2ODc0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICcxMDAlJwogICAgICAgICAgICAKICAgICAgICAgICAgfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzM4YzM1MWRiN2U1YzRiNzk4NzViMzk5ZGJlNzMzM2I5ID0gJChgPGRpdiBpZD0iaHRtbF8zOGMzNTFkYjdlNWM0Yjc5ODc1YjM5OWRiZTczMzNiOSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TWFkaGF2YXJhbSBDbHVzdGVyIDEuMDwvZGl2PmApWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNDAyNjJlMmRhNmJhNDc2OTg3MWQ4NjM0MWE5OTY4NzQuc2V0Q29udGVudChodG1sXzM4YzM1MWRiN2U1YzRiNzk4NzViMzk5ZGJlNzMzM2I5KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzc4N2JlYjVjNjBjNzRmYzhhMWFkYTBjZjAyNTNiMjE5LmJpbmRQb3B1cChwb3B1cF80MDI2MmUyZGE2YmE0NzY5ODcxZDg2MzQxYTk5Njg3NCkKICAgICAgICAgICAgOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYzliZDZhOGY5YjQwNDY2Yjk1ZWFjYjZmMzQ3YjhiOTAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFsxMy4xNSwgODAuMjRdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzNkODRhZCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzZDg0YWQiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hNzhiMGRhMjkwMDI0YTRmYmQ4NzIxNWQ1MTUwOWMyNyA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9mNDY2MTAwZTMwZDE0ZDNlOGRhOTQ2NjQ1NDQ0Y2Q0NiA9ICQoYDxkaXYgaWQ9Imh0bWxfZjQ2NjEwMGUzMGQxNGQzZThkYTk0NjY0NTQ0NGNkNDYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk1hZGhhdmFyYW0gTWlsayBDb2xvbnkgQ2x1c3RlciAxLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2E3OGIwZGEyOTAwMjRhNGZiZDg3MjE1ZDUxNTA5YzI3LnNldENvbnRlbnQoaHRtbF9mNDY2MTAwZTMwZDE0ZDNlOGRhOTQ2NjQ1NDQ0Y2Q0Nik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jOWJkNmE4ZjliNDA0NjZiOTVlYWNiNmYzNDdiOGI5MC5iaW5kUG9wdXAocG9wdXBfYTc4YjBkYTI5MDAyNGE0ZmJkODcyMTVkNTE1MDljMjcpCiAgICAgICAgICAgIDsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzVlZjY2ODg4YzAwODQ0NzlhMjUyMjQwYzg5NjQyN2RiID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMTIuOTY0ODcsIDgwLjE5NjExXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMzZDg0YWQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjM2Q4NGFkIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYThiNjYwMjllNzc3NGM4N2IyYjRhMDgyMjEwNDFlNjYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMmRjNDU0MTFiOWU5NDY3OWE0ZGJlNTY5ZDA3YzllNDMgPSAkKGA8ZGl2IGlkPSJodG1sXzJkYzQ1NDExYjllOTQ2NzlhNGRiZTU2OWQwN2M5ZTQzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5NYWRpcGFra2FtIENsdXN0ZXIgMS4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hOGI2NjAyOWU3Nzc0Yzg3YjJiNGEwODIyMTA0MWU2Ni5zZXRDb250ZW50KGh0bWxfMmRjNDU0MTFiOWU5NDY3OWE0ZGJlNTY5ZDA3YzllNDMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNWVmNjY4ODhjMDA4NDQ3OWEyNTIyNDBjODk2NDI3ZGIuYmluZFBvcHVwKHBvcHVwX2E4YjY2MDI5ZTc3NzRjODdiMmI0YTA4MjIxMDQxZTY2KQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85ZDhlYjZjZDg5NWM0Y2VjYjExYjI3YjllMWFhNjBhZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEzLjA2NzUsIDgwLjE2MzJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzNkODRhZCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzZDg0YWQiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83MDI1NWUyY2M2MTM0Nzg0YmIyNTNkMTRiZjAyMDJmYiA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wZWM2YjAyNGRlZTE0NzViOGYyM2I4NjYyYjFiOTkxYyA9ICQoYDxkaXYgaWQ9Imh0bWxfMGVjNmIwMjRkZWUxNDc1YjhmMjNiODY2MmIxYjk5MWMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk1hZHVyYXZveWFsIENsdXN0ZXIgMS4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83MDI1NWUyY2M2MTM0Nzg0YmIyNTNkMTRiZjAyMDJmYi5zZXRDb250ZW50KGh0bWxfMGVjNmIwMjRkZWUxNDc1YjhmMjNiODY2MmIxYjk5MWMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOWQ4ZWI2Y2Q4OTVjNGNlY2IxMWIyN2I5ZTFhYTYwYWUuYmluZFBvcHVwKHBvcHVwXzcwMjU1ZTJjYzYxMzQ3ODRiYjI1M2QxNGJmMDIwMmZiKQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83NTJhNDBmMDI5ZDQ0NmU0OTg2MDQzYTZkOWU2ZjAyMiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEzLjE2MzYxLCA4MC4yNTg1Nl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjMDAwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzAwMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICAuYWRkVG8obWFwXzAyMzdjOTQxNzdhMzQ4ZmY5MzQzMWQxYTg4NmQwMWUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2U2OTY5MjYxZDdiMjQyOGRiMzM1MGFlMWJiMmQ5ZjJkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICcxMDAlJwogICAgICAgICAgICAKICAgICAgICAgICAgfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2I5NTQzNDg5OWFhNzRmNzhhNzg5M2JhZTIwYTgwNjczID0gJChgPGRpdiBpZD0iaHRtbF9iOTU0MzQ4OTlhYTc0Zjc4YTc4OTNiYWUyMGE4MDY3MyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TWFuYWxpIENsdXN0ZXIgMi4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lNjk2OTI2MWQ3YjI0MjhkYjMzNTBhZTFiYjJkOWYyZC5zZXRDb250ZW50KGh0bWxfYjk1NDM0ODk5YWE3NGY3OGE3ODkzYmFlMjBhODA2NzMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNzUyYTQwZjAyOWQ0NDZlNDk4NjA0M2E2ZDllNmYwMjIuYmluZFBvcHVwKHBvcHVwX2U2OTY5MjYxZDdiMjQyOGRiMzM1MGFlMWJiMmQ5ZjJkKQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lNmE3NTkzYmEwZDM0OTI3YWVjZTIxNTQxYjUzNmMxYyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEzLjIwODAyOCwgODAuMjcxNTg1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMzZDg0YWQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjM2Q4NGFkIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYjNiN2E4YTU5OTA5NDI3YWFlYjg3NGJmMTY2MzQ1MjYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMzMzY2UzN2YyY2UzNDYyNmE2N2RkYzZiNzk2YTAxNzkgPSAkKGA8ZGl2IGlkPSJodG1sXzMzM2NlMzdmMmNlMzQ2MjZhNjdkZGM2Yjc5NmEwMTc5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5NYW5hbGkgTmV3IFRvd24gQ2x1c3RlciAxLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2IzYjdhOGE1OTkwOTQyN2FhZWI4NzRiZjE2NjM0NTI2LnNldENvbnRlbnQoaHRtbF8zMzNjZTM3ZjJjZTM0NjI2YTY3ZGRjNmI3OTZhMDE3OSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lNmE3NTkzYmEwZDM0OTI3YWVjZTIxNTQxYjUzNmMxYy5iaW5kUG9wdXAocG9wdXBfYjNiN2E4YTU5OTA5NDI3YWFlYjg3NGJmMTY2MzQ1MjYpCiAgICAgICAgICAgIDsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzBmYjA2OWZiNjFhMTQ3MmM5MzYwY2Y3YjJjZWU5MDkxID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMTMuMDI3LCA4MC4yNjAyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMzZDg0YWQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjM2Q4NGFkIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfODQzYzAyNmQ1MzJiNDQzZmE1OTEyMzYzNjkxMjdkODAgPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOTYxMDBkNDRmZDlkNGJkYWI1ZDk3MGY3NDRiMjQ5MDEgPSAkKGA8ZGl2IGlkPSJodG1sXzk2MTAwZDQ0ZmQ5ZDRiZGFiNWQ5NzBmNzQ0YjI0OTAxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5NYW5kYXZlbGkgQ2x1c3RlciAxLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzg0M2MwMjZkNTMyYjQ0M2ZhNTkxMjM2MzY5MTI3ZDgwLnNldENvbnRlbnQoaHRtbF85NjEwMGQ0NGZkOWQ0YmRhYjVkOTcwZjc0NGIyNDkwMSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wZmIwNjlmYjYxYTE0NzJjOTM2MGNmN2IyY2VlOTA5MS5iaW5kUG9wdXAocG9wdXBfODQzYzAyNmQ1MzJiNDQzZmE1OTEyMzYzNjkxMjdkODApCiAgICAgICAgICAgIDsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzJjMjliZTNkMDZkOTRiYzg5NDNlNzdjYjEzNTA5OGE3ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMTMuMDM5Njc1LCA4MC4xMDk3MTddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzAwMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMwMDAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iMDU4N2NmNDRkMzI0MDNjYjkxOTdlMGI0ODY2YzYzZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hNjljOTcxMTlmZjU0NjI3OTBiZWUwZGJkNzlkMDVhNiA9ICQoYDxkaXYgaWQ9Imh0bWxfYTY5Yzk3MTE5ZmY1NDYyNzkwYmVlMGRiZDc5ZDA1YTYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk1hbmdhZHUgQ2x1c3RlciAyLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2IwNTg3Y2Y0NGQzMjQwM2NiOTE5N2UwYjQ4NjZjNjNmLnNldENvbnRlbnQoaHRtbF9hNjljOTcxMTlmZjU0NjI3OTBiZWUwZGJkNzlkMDVhNik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yYzI5YmUzZDA2ZDk0YmM4OTQzZTc3Y2IxMzUwOThhNy5iaW5kUG9wdXAocG9wdXBfYjA1ODdjZjQ0ZDMyNDAzY2I5MTk3ZTBiNDg2NmM2M2YpCiAgICAgICAgICAgIDsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzA4MjhkNjU3Yzc1MzQxYmFhYTFjZjA4NDYyYzgzYTA5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMTIuOTE3MTU4LCA4MC4xOTI4N10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICAuYWRkVG8obWFwXzAyMzdjOTQxNzdhMzQ4ZmY5MzQzMWQxYTg4NmQwMWUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzQ0Yjc4NDY2OGY3NjQwZjVhZWU1MjhlOTI1ODQyN2IxID0gTC5wb3B1cCh7bWF4V2lkdGg6ICcxMDAlJwogICAgICAgICAgICAKICAgICAgICAgICAgfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzdlOWRiODI0MjJkZjRlOWI4Y2NkNGI1ODVmODM5YTAwID0gJChgPGRpdiBpZD0iaHRtbF83ZTlkYjgyNDIyZGY0ZTliOGNjZDRiNTg1ZjgzOWEwMCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TWF0aHVyIENsdXN0ZXIgMC4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80NGI3ODQ2NjhmNzY0MGY1YWVlNTI4ZTkyNTg0MjdiMS5zZXRDb250ZW50KGh0bWxfN2U5ZGI4MjQyMmRmNGU5YjhjY2Q0YjU4NWY4MzlhMDApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMDgyOGQ2NTdjNzUzNDFiYWFhMWNmMDg0NjJjODNhMDkuYmluZFBvcHVwKHBvcHVwXzQ0Yjc4NDY2OGY3NjQwZjVhZWU1MjhlOTI1ODQyN2IxKQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83MjBiYThiZTQyZjg0NmQ4YTE1MTMzMzAzZTYxMzk5ZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEyLjk4NDU2LCA4MC4xNzQ3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZmZmMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmZmZjAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZmM5N2ZlOThiMjA2NGYwOTkwNWZiYjY4MjM0NTFiZWEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZTU4ZDM2NDZmZGQ0NDU5M2FhNjA1NzlkNDBkY2Q1ZWYgPSAkKGA8ZGl2IGlkPSJodG1sX2U1OGQzNjQ2ZmRkNDQ1OTNhYTYwNTc5ZDQwZGNkNWVmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5NZWRhdmFra2FtIENsdXN0ZXIgMy4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mYzk3ZmU5OGIyMDY0ZjA5OTA1ZmJiNjgyMzQ1MWJlYS5zZXRDb250ZW50KGh0bWxfZTU4ZDM2NDZmZGQ0NDU5M2FhNjA1NzlkNDBkY2Q1ZWYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNzIwYmE4YmU0MmY4NDZkOGExNTEzMzMwM2U2MTM5OWUuYmluZFBvcHVwKHBvcHVwX2ZjOTdmZTk4YjIwNjRmMDk5MDVmYmI2ODIzNDUxYmVhKQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80ZWQ4ZGVlMDY4Yzc0YTc1OTU0ODU0MzIzMjdiYzAwNSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEzLjAzNjMsIDgwLjIwMTRdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF84NzI4MWQwYjlmNmQ0YzgwOGEyYWY0OGEwOWFjNzhhMyA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82MDAzMTQyOGZmZjI0MTJlOTc4MGM0YWQzMDU2NWVkZSA9ICQoYDxkaXYgaWQ9Imh0bWxfNjAwMzE0MjhmZmYyNDEyZTk3ODBjNGFkMzA1NjVlZGUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk1lZW5hbWJha2thbSBDbHVzdGVyIDAuMDwvZGl2PmApWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfODcyODFkMGI5ZjZkNGM4MDhhMmFmNDhhMDlhYzc4YTMuc2V0Q29udGVudChodG1sXzYwMDMxNDI4ZmZmMjQxMmU5NzgwYzRhZDMwNTY1ZWRlKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzRlZDhkZWUwNjhjNzRhNzU5NTQ4NTQzMjMyN2JjMDA1LmJpbmRQb3B1cChwb3B1cF84NzI4MWQwYjlmNmQ0YzgwOGEyYWY0OGEwOWFjNzhhMykKICAgICAgICAgICAgOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYTBlZjYxN2VmYzA4NDMxNGJiNDMwZDhmNGE2ODFkNmEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFsxMy4wODA1LCA4MC4xODAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOWUxODkwZTUxNDQyNGEzNGJhNjQ2ODE3OTliMDkwMzcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMWFjM2IxNGM1YjMyNDk3MDg1ZGNhOTZjZTlkODRiYjEgPSAkKGA8ZGl2IGlkPSJodG1sXzFhYzNiMTRjNWIzMjQ5NzA4NWRjYTk2Y2U5ZDg0YmIxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5NaW5qdXIgQ2x1c3RlciAwLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzllMTg5MGU1MTQ0MjRhMzRiYTY0NjgxNzk5YjA5MDM3LnNldENvbnRlbnQoaHRtbF8xYWMzYjE0YzViMzI0OTcwODVkY2E5NmNlOWQ4NGJiMSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hMGVmNjE3ZWZjMDg0MzE0YmI0MzBkOGY0YTY4MWQ2YS5iaW5kUG9wdXAocG9wdXBfOWUxODkwZTUxNDQyNGEzNGJhNjQ2ODE3OTliMDkwMzcpCiAgICAgICAgICAgIDsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzhlZWNkZDFmYTkwMDQxNzhhZTJkNTJjZjZkOTllNmZmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMTMuMTIxMjc3LCA4MC4yNTg3MzJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzNkODRhZCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzZDg0YWQiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hZTI1MjJiOTM0YjQ0ODY5YjM2ZmJiOTE3ZjliZjQ1MyA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82ZDM5YWNlNDFhMDI0Njc5YmQ2MDRkMDg1ZWQ2YmY5MSA9ICQoYDxkaXYgaWQ9Imh0bWxfNmQzOWFjZTQxYTAyNDY3OWJkNjA0ZDA4NWVkNmJmOTEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk1vZ2FwcGFpciBDbHVzdGVyIDEuMDwvZGl2PmApWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYWUyNTIyYjkzNGI0NDg2OWIzNmZiYjkxN2Y5YmY0NTMuc2V0Q29udGVudChodG1sXzZkMzlhY2U0MWEwMjQ2NzliZDYwNGQwODVlZDZiZjkxKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzhlZWNkZDFmYTkwMDQxNzhhZTJkNTJjZjZkOTllNmZmLmJpbmRQb3B1cChwb3B1cF9hZTI1MjJiOTM0YjQ0ODY5YjM2ZmJiOTE3ZjliZjQ1MykKICAgICAgICAgICAgOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMTZmMzc5MmEzNTA1NGQ2Njk0Y2IzZWUzMGFmMGMwZDEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFsxMy4wNjQzNjksIDgwLjI2NTgwOF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICAuYWRkVG8obWFwXzAyMzdjOTQxNzdhMzQ4ZmY5MzQzMWQxYTg4NmQwMWUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzcxNjNjZWRhMWFkNzRjZDBhOTdlMzcyNjdlNDc3Yjg4ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICcxMDAlJwogICAgICAgICAgICAKICAgICAgICAgICAgfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2E1NzFmM2ZjM2E0ZTRjMTlhNTZhZWQ5YTgxNTY2NjM4ID0gJChgPGRpdiBpZD0iaHRtbF9hNTcxZjNmYzNhNGU0YzE5YTU2YWVkOWE4MTU2NjYzOCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TUtCIE5hZ2FyIENsdXN0ZXIgMC4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83MTYzY2VkYTFhZDc0Y2QwYTk3ZTM3MjY3ZTQ3N2I4OC5zZXRDb250ZW50KGh0bWxfYTU3MWYzZmMzYTRlNGMxOWE1NmFlZDlhODE1NjY2MzgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMTZmMzc5MmEzNTA1NGQ2Njk0Y2IzZWUzMGFmMGMwZDEuYmluZFBvcHVwKHBvcHVwXzcxNjNjZWRhMWFkNzRjZDBhOTdlMzcyNjdlNDc3Yjg4KQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80MDgxYWQzYTA2Y2Y0ZmQ1YjBlMjNlYjQ2OGM3MGE5NiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEzLjEyODkzNywgODAuMjQxNDUzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMzZDg0YWQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjM2Q4NGFkIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfY2JmZWFjNjhhYjVlNDk0Mjk4NDRlZDhjNTdhMmNiNjYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOTlhN2RiYTY5OTBlNDk4Y2IwMGFmM2JiYWI4MjFiNGEgPSAkKGA8ZGl2IGlkPSJodG1sXzk5YTdkYmE2OTkwZTQ5OGNiMDBhZjNiYmFiODIxYjRhIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Nb3VudCBSb2FkIENsdXN0ZXIgMS4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9jYmZlYWM2OGFiNWU0OTQyOTg0NGVkOGM1N2EyY2I2Ni5zZXRDb250ZW50KGh0bWxfOTlhN2RiYTY5OTBlNDk4Y2IwMGFmM2JiYWI4MjFiNGEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNDA4MWFkM2EwNmNmNGZkNWIwZTIzZWI0NjhjNzBhOTYuYmluZFBvcHVwKHBvcHVwX2NiZmVhYzY4YWI1ZTQ5NDI5ODQ0ZWQ4YzU3YTJjYjY2KQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zYzJmYzYyODgxMGE0ZDBjODc2Njc5Yzk0ODc0YmQwYyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEzLjAyMDI2OSwgODAuMTQzNTc4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZGMzNGM0ODEzOTA2NDM4NDg0ZDY4NTI0YzQ3MWE4Y2YgPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNzM0NTAwYTkyYzU2NDg1Yjg4YmVlNTdjYWE4MjM3Y2EgPSAkKGA8ZGl2IGlkPSJodG1sXzczNDUwMGE5MmM1NjQ4NWI4OGJlZTU3Y2FhODIzN2NhIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Nb29sYWthZGFpIENsdXN0ZXIgMC4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kYzM0YzQ4MTM5MDY0Mzg0ODRkNjg1MjRjNDcxYThjZi5zZXRDb250ZW50KGh0bWxfNzM0NTAwYTkyYzU2NDg1Yjg4YmVlNTdjYWE4MjM3Y2EpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfM2MyZmM2Mjg4MTBhNGQwYzg3NjY3OWM5NDg3NGJkMGMuYmluZFBvcHVwKHBvcHVwX2RjMzRjNDgxMzkwNjQzODQ4NGQ2ODUyNGM0NzFhOGNmKQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iZGMyYzBlOTA3MGQ0ZDFkYjJmOGJmMzE0NGE0OWMwZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEzLjAyMTU2NTMsIDgwLjE2MDUyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMzZDg0YWQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjM2Q4NGFkIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMjc2Yzg2YzYxZjI1NGM0Y2JkZDMxMGYwNDFkMmFkZjkgPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMDYxMDliNWEwNmZmNDAwZGFlY2E2ZDg3ZGZmNzY3OWQgPSAkKGA8ZGl2IGlkPSJodG1sXzA2MTA5YjVhMDZmZjQwMGRhZWNhNmQ4N2RmZjc2NzlkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Nb3VsaXZha2thbSBDbHVzdGVyIDEuMDwvZGl2PmApWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMjc2Yzg2YzYxZjI1NGM0Y2JkZDMxMGYwNDFkMmFkZjkuc2V0Q29udGVudChodG1sXzA2MTA5YjVhMDZmZjQwMGRhZWNhNmQ4N2RmZjc2NzlkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2JkYzJjMGU5MDcwZDRkMWRiMmY4YmYzMTQ0YTQ5YzBmLmJpbmRQb3B1cChwb3B1cF8yNzZjODZjNjFmMjU0YzRjYmRkMzEwZjA0MWQyYWRmOSkKICAgICAgICAgICAgOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZTY4MjMzY2NhYzYwNDgwMGEyYjY1MzZlMzk3YTc4MDEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFsxMy4wMzM2LCA4MC4yNjg3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYjFkZTcwYjI0ZTA1NGJkNzg2YzU0OThhYTZhMzE4MjcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZGQ1MTAzOTIxMzZmNGMyN2I3M2FlODkzYTM3Njg4ODEgPSAkKGA8ZGl2IGlkPSJodG1sX2RkNTEwMzkyMTM2ZjRjMjdiNzNhZTg5M2EzNzY4ODgxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5NdWRpY2h1ciBDbHVzdGVyIDAuMDwvZGl2PmApWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYjFkZTcwYjI0ZTA1NGJkNzg2YzU0OThhYTZhMzE4Mjcuc2V0Q29udGVudChodG1sX2RkNTEwMzkyMTM2ZjRjMjdiNzNhZTg5M2EzNzY4ODgxKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2U2ODIzM2NjYWM2MDQ4MDBhMmI2NTM2ZTM5N2E3ODAxLmJpbmRQb3B1cChwb3B1cF9iMWRlNzBiMjRlMDU0YmQ3ODZjNTQ5OGFhNmEzMTgyNykKICAgICAgICAgICAgOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOTg0YjMxMmZjMjFjNGQwNmExYjcyMTRmNDJlNGY5MmEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFsxMy4wMzAxLCA4MC4yNDE2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMzZDg0YWQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjM2Q4NGFkIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfM2I3Y2QwMGQ3YzNiNDdmMTkxNmM0ZjkxODA4MzI3MzMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMjcyNjcyYTEzMDM1NGIyNDhhMGZhMmY2NGVjZWEyMjAgPSAkKGA8ZGl2IGlkPSJodG1sXzI3MjY3MmExMzAzNTRiMjQ4YTBmYTJmNjRlY2VhMjIwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5NeWxhcG9yZSBDbHVzdGVyIDEuMDwvZGl2PmApWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfM2I3Y2QwMGQ3YzNiNDdmMTkxNmM0ZjkxODA4MzI3MzMuc2V0Q29udGVudChodG1sXzI3MjY3MmExMzAzNTRiMjQ4YTBmYTJmNjRlY2VhMjIwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzk4NGIzMTJmYzIxYzRkMDZhMWI3MjE0ZjQyZTRmOTJhLmJpbmRQb3B1cChwb3B1cF8zYjdjZDAwZDdjM2I0N2YxOTE2YzRmOTE4MDgzMjczMykKICAgICAgICAgICAgOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNWQ5NmE3MGUzOTI5NGU4YmE2ZTkyNTczNjBkN2U4ZGQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFsxMi45NzUyLCA4MC4xOTIzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMzZDg0YWQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjM2Q4NGFkIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMzExNzI4MzhjZWViNDIxNzhjNTZjNzgyMTZhZDE3NzEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNjQxMzFiZDljOGVkNGY3MzgwNTcxN2MwYzAwNzMxYTMgPSAkKGA8ZGl2IGlkPSJodG1sXzY0MTMxYmQ5YzhlZDRmNzM4MDU3MTdjMGMwMDczMWEzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5OYW5kYW5hbSBDbHVzdGVyIDEuMDwvZGl2PmApWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMzExNzI4MzhjZWViNDIxNzhjNTZjNzgyMTZhZDE3NzEuc2V0Q29udGVudChodG1sXzY0MTMxYmQ5YzhlZDRmNzM4MDU3MTdjMGMwMDczMWEzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzVkOTZhNzBlMzkyOTRlOGJhNmU5MjU3MzYwZDdlOGRkLmJpbmRQb3B1cChwb3B1cF8zMTE3MjgzOGNlZWI0MjE3OGM1NmM3ODIxNmFkMTc3MSkKICAgICAgICAgICAgOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZGZlZDhhNjI5NTUwNDE3NWJjYzQ3YmY5YTI0ZTFjZGYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFsxMi45NDk1LCA4MC4yNTkyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMzZDg0YWQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjM2Q4NGFkIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYjEzZjdlYmJjODMyNGIyZmFiYjdmNDY3NTU4MzcxZWUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNDk3M2Q0YWRhMTcxNGJhMzhlOGMzNzI2ODU4NjNlMDcgPSAkKGA8ZGl2IGlkPSJodG1sXzQ5NzNkNGFkYTE3MTRiYTM4ZThjMzcyNjg1ODYzZTA3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5OYW5nYW5hbGx1ciBDbHVzdGVyIDEuMDwvZGl2PmApWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYjEzZjdlYmJjODMyNGIyZmFiYjdmNDY3NTU4MzcxZWUuc2V0Q29udGVudChodG1sXzQ5NzNkNGFkYTE3MTRiYTM4ZThjMzcyNjg1ODYzZTA3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2RmZWQ4YTYyOTU1MDQxNzViY2M0N2JmOWEyNGUxY2RmLmJpbmRQb3B1cChwb3B1cF9iMTNmN2ViYmM4MzI0YjJmYWJiN2Y0Njc1NTgzNzFlZSkKICAgICAgICAgICAgOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMDBlNTMyYjE0YjU5NDZkYjk2Yzg5M2M3ZmMyMjAxNjggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFsxMy4wNDAwNSwgODAuMTk5MjhdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jNzBkMmE2YTE3MDY0MTBhYjI3MjE4OTkxY2FjMTJmZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iZDQwZjc1ODI1Yzk0Y2IzOTkxMjcyZjQxZDFmM2FhYiA9ICQoYDxkaXYgaWQ9Imh0bWxfYmQ0MGY3NTgyNWM5NGNiMzk5MTI3MmY0MWQxZjNhYWIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk5hbm1hbmdhbGFtIENsdXN0ZXIgMC4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9jNzBkMmE2YTE3MDY0MTBhYjI3MjE4OTkxY2FjMTJmZi5zZXRDb250ZW50KGh0bWxfYmQ0MGY3NTgyNWM5NGNiMzk5MTI3MmY0MWQxZjNhYWIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMDBlNTMyYjE0YjU5NDZkYjk2Yzg5M2M3ZmMyMjAxNjguYmluZFBvcHVwKHBvcHVwX2M3MGQyYTZhMTcwNjQxMGFiMjcyMTg5OTFjYWMxMmZmKQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82NmQxNTc0YjhlOTg0ZTkzYjI2ZGU5MzI0YmZjYjBmNSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEzLjA4MDUsIDgwLjE4MDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85YWY1YmZjZjJiZmM0ZGJhOTZmZGE2MjYyZmFjY2EzYyA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9mMGRkMzExM2Q2ODk0Y2IwOGNjYWYwZjFiY2JiNThkYyA9ICQoYDxkaXYgaWQ9Imh0bWxfZjBkZDMxMTNkNjg5NGNiMDhjY2FmMGYxYmNiYjU4ZGMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk5lZWxhbmthcmFpIENsdXN0ZXIgMC4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85YWY1YmZjZjJiZmM0ZGJhOTZmZGE2MjYyZmFjY2EzYy5zZXRDb250ZW50KGh0bWxfZjBkZDMxMTNkNjg5NGNiMDhjY2FmMGYxYmNiYjU4ZGMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNjZkMTU3NGI4ZTk4NGU5M2IyNmRlOTMyNGJmY2IwZjUuYmluZFBvcHVwKHBvcHVwXzlhZjViZmNmMmJmYzRkYmE5NmZkYTYyNjJmYWNjYTNjKQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iNWFmMGY1MTVjNjg0NTU0YWFjM2IyZGZkNTU0M2ViYSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEzLjA2NTQsIDgwLjIzMjZdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzNkODRhZCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzZDg0YWQiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81ZDM5MmZlNDVhYTg0ZGM4OTk0MjQ1YjE2MTg4MWMxMCA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8xNWI4NjRjM2FjZGI0MjA0YWEzYjkzMDhmYTJkNDMwYyA9ICQoYDxkaXYgaWQ9Imh0bWxfMTViODY0YzNhY2RiNDIwNGFhM2I5MzA4ZmEyZDQzMGMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk5lbWlsaWNoZXJ5IENsdXN0ZXIgMS4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81ZDM5MmZlNDVhYTg0ZGM4OTk0MjQ1YjE2MTg4MWMxMC5zZXRDb250ZW50KGh0bWxfMTViODY0YzNhY2RiNDIwNGFhM2I5MzA4ZmEyZDQzMGMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYjVhZjBmNTE1YzY4NDU1NGFhYzNiMmRmZDU1NDNlYmEuYmluZFBvcHVwKHBvcHVwXzVkMzkyZmU0NWFhODRkYzg5OTQyNDViMTYxODgxYzEwKQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81Yjc3NGExOTE5NWU0ODNiYWU3NjhmYTA0ZGYwOGFkYiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEzLjA5ODAzOCwgODAuMjUxMTk3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMzZDg0YWQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjM2Q4NGFkIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfN2QwZjc0ZWVhZjA2NDY4ZWE3OWMzNTNlODhiODAxNjEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfY2MyZTMyNGY3OWJjNDM2Zjg4ZGQ5MzAzYTc3OGU2YjUgPSAkKGA8ZGl2IGlkPSJodG1sX2NjMmUzMjRmNzliYzQzNmY4OGRkOTMwM2E3NzhlNmI1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5OZXNhcGFra2FtIENsdXN0ZXIgMS4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83ZDBmNzRlZWFmMDY0NjhlYTc5YzM1M2U4OGI4MDE2MS5zZXRDb250ZW50KGh0bWxfY2MyZTMyNGY3OWJjNDM2Zjg4ZGQ5MzAzYTc3OGU2YjUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNWI3NzRhMTkxOTVlNDgzYmFlNzY4ZmEwNGRmMDhhZGIuYmluZFBvcHVwKHBvcHVwXzdkMGY3NGVlYWYwNjQ2OGVhNzljMzUzZTg4YjgwMTYxKQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mYTk0NzAxMzRiZjc0NmMyYmZmMDBmNjY1YjZhNmQ5YyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEzLjEwMzcsIDgwLjE5NDddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzNkODRhZCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzZDg0YWQiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8wMzY5MTE3NjQ3ZWU0ZGQ2ODZlMjQ5MjAxNTY1ZjA5YiA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yZmNiZGY0MmNhN2I0ODJlYThjMGRmNmU1NWFlMWNmZCA9ICQoYDxkaXYgaWQ9Imh0bWxfMmZjYmRmNDJjYTdiNDgyZWE4YzBkZjZlNTVhZTFjZmQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk5vbGFtYnVyIENsdXN0ZXIgMS4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wMzY5MTE3NjQ3ZWU0ZGQ2ODZlMjQ5MjAxNTY1ZjA5Yi5zZXRDb250ZW50KGh0bWxfMmZjYmRmNDJjYTdiNDgyZWE4YzBkZjZlNTVhZTFjZmQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZmE5NDcwMTM0YmY3NDZjMmJmZjAwZjY2NWI2YTZkOWMuYmluZFBvcHVwKHBvcHVwXzAzNjkxMTc2NDdlZTRkZDY4NmUyNDkyMDE1NjVmMDliKQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jMWIxYjY0M2I2YjU0MTZhOWExNDZkNjQxNzc4YTYzOCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEyLjk1MzUsIDgwLjI1NzJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lODEyYmExYmZjOTE0MTAzYTZkNTczZGZlODJiYmRhZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85NGZhMjZmNWE4NzA0NDZkOTQ5YjFlYzllMTQwNTQ0YyA9ICQoYDxkaXYgaWQ9Imh0bWxfOTRmYTI2ZjVhODcwNDQ2ZDk0OWIxZWM5ZTE0MDU0NGMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk51bmdhbWJha2thbSBDbHVzdGVyIDAuMDwvZGl2PmApWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZTgxMmJhMWJmYzkxNDEwM2E2ZDU3M2RmZTgyYmJkYWUuc2V0Q29udGVudChodG1sXzk0ZmEyNmY1YTg3MDQ0NmQ5NDliMWVjOWUxNDA1NDRjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2MxYjFiNjQzYjZiNTQxNmE5YTE0NmQ2NDE3NzhhNjM4LmJpbmRQb3B1cChwb3B1cF9lODEyYmExYmZjOTE0MTAzYTZkNTczZGZlODJiYmRhZSkKICAgICAgICAgICAgOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfM2E0M2I0MGJhZDNhNGNkYzg5NTA1NWRiNDE2NjU2OTcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFsxMi45OCwgODAuMThdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzAwMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMwMDAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80MmRmZjEyMGRjMGI0N2NhYTAzMTMwOGU1NmRlMzAyNyA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wOGM4NGNhY2FmM2Q0OGRjOGQ4YmNjOTM4MTY2MWYyOCA9ICQoYDxkaXYgaWQ9Imh0bWxfMDhjODRjYWNhZjNkNDhkYzhkOGJjYzkzODE2NjFmMjgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk90dGVyaSBDbHVzdGVyIDIuMDwvZGl2PmApWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNDJkZmYxMjBkYzBiNDdjYWEwMzEzMDhlNTZkZTMwMjcuc2V0Q29udGVudChodG1sXzA4Yzg0Y2FjYWYzZDQ4ZGM4ZDhiY2M5MzgxNjYxZjI4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzNhNDNiNDBiYWQzYTRjZGM4OTUwNTVkYjQxNjY1Njk3LmJpbmRQb3B1cChwb3B1cF80MmRmZjEyMGRjMGI0N2NhYTAzMTMwOGU1NmRlMzAyNykKICAgICAgICAgICAgOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMmE5MjVhYTQ0YmJkNDQyOWJmMTk5OTYyODI4ODUxZjIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFsxMi45NzUsIDgwLjEzNDcyMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjM2Q4NGFkIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzNkODRhZCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICAuYWRkVG8obWFwXzAyMzdjOTQxNzdhMzQ4ZmY5MzQzMWQxYTg4NmQwMWUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2I0ZDY1YmM1NzE3MDQ1NTZhY2Y1N2Y4ODUwYjRhM2I0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICcxMDAlJwogICAgICAgICAgICAKICAgICAgICAgICAgfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzk2MzVhZWEzNzMwYzRiZDQ4ZGNkZjQzZGQ0YWJhYzZhID0gJChgPGRpdiBpZD0iaHRtbF85NjM1YWVhMzczMGM0YmQ0OGRjZGY0M2RkNGFiYWM2YSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UGFra2FtIENsdXN0ZXIgMS4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iNGQ2NWJjNTcxNzA0NTU2YWNmNTdmODg1MGI0YTNiNC5zZXRDb250ZW50KGh0bWxfOTYzNWFlYTM3MzBjNGJkNDhkY2RmNDNkZDRhYmFjNmEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMmE5MjVhYTQ0YmJkNDQyOWJmMTk5OTYyODI4ODUxZjIuYmluZFBvcHVwKHBvcHVwX2I0ZDY1YmM1NzE3MDQ1NTZhY2Y1N2Y4ODUwYjRhM2I0KQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83ZjI2ZjZkNzI1ODQ0NjBkYWI5NjZmZjdiZTVkYmEyNiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEzLjA3OTgsIDgwLjI3NzZdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmZmYwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZmZmMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hN2E5NTA2NmI5ZmQ0YmY0OTYyMWViMTg5MWRkNzY5YiA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9mMWQwZTZjNTQyMDU0ZTY4YTM2ZmYyZWM5YzUyMDU2MyA9ICQoYDxkaXYgaWQ9Imh0bWxfZjFkMGU2YzU0MjA1NGU2OGEzNmZmMmVjOWM1MjA1NjMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlBhbGF2YWtrYW0gQ2x1c3RlciAzLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2E3YTk1MDY2YjlmZDRiZjQ5NjIxZWIxODkxZGQ3NjliLnNldENvbnRlbnQoaHRtbF9mMWQwZTZjNTQyMDU0ZTY4YTM2ZmYyZWM5YzUyMDU2Myk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83ZjI2ZjZkNzI1ODQ0NjBkYWI5NjZmZjdiZTVkYmEyNi5iaW5kUG9wdXAocG9wdXBfYTdhOTUwNjZiOWZkNGJmNDk2MjFlYjE4OTFkZDc2OWIpCiAgICAgICAgICAgIDsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzM0Y2EzYzM0N2VmNTQ0MGZiOTMxNzNmYjU1NzUxNDg4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMTMuMDkzOSwgODAuMjgzOV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmZmZjAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmZmYwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICAuYWRkVG8obWFwXzAyMzdjOTQxNzdhMzQ4ZmY5MzQzMWQxYTg4NmQwMWUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzQ2MzNhZWFlY2Y2MDRlMzQ5MTk5NGFhMGZkY2ZlZThiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICcxMDAlJwogICAgICAgICAgICAKICAgICAgICAgICAgfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2I4ODg1YjYyODIwYzQxMmRiZWM4NzVkYzYxMWVjYzU0ID0gJChgPGRpdiBpZD0iaHRtbF9iODg4NWI2MjgyMGM0MTJkYmVjODc1ZGM2MTFlY2M1NCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UGFsbGF2YXJhbSBDbHVzdGVyIDMuMDwvZGl2PmApWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNDYzM2FlYWVjZjYwNGUzNDkxOTk0YWEwZmRjZmVlOGIuc2V0Q29udGVudChodG1sX2I4ODg1YjYyODIwYzQxMmRiZWM4NzVkYzYxMWVjYzU0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzM0Y2EzYzM0N2VmNTQ0MGZiOTMxNzNmYjU1NzUxNDg4LmJpbmRQb3B1cChwb3B1cF80NjMzYWVhZWNmNjA0ZTM0OTE5OTRhYTBmZGNmZWU4YikKICAgICAgICAgICAgOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYTY4M2E5YzgzMjVkNGNmNmJkMmJlMzkxZmNmYTAzMjQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFsxMy4xMjM2LCA4MC4wNl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjMDAwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzAwMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICAuYWRkVG8obWFwXzAyMzdjOTQxNzdhMzQ4ZmY5MzQzMWQxYTg4NmQwMWUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzAzYzc5NDgwZDc2NzQ1YzJhMGZhYjc0NmQyNjhlYTkyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICcxMDAlJwogICAgICAgICAgICAKICAgICAgICAgICAgfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzExODY0ZWIwNjQzNjQzMjY5MjAzMzU1YTVhNmExYTEyID0gJChgPGRpdiBpZD0iaHRtbF8xMTg2NGViMDY0MzY0MzI2OTIwMzM1NWE1YTZhMWExMiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UGFsbGlrYXJhbmFpIENsdXN0ZXIgMi4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wM2M3OTQ4MGQ3Njc0NWMyYTBmYWI3NDZkMjY4ZWE5Mi5zZXRDb250ZW50KGh0bWxfMTE4NjRlYjA2NDM2NDMyNjkyMDMzNTVhNWE2YTFhMTIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYTY4M2E5YzgzMjVkNGNmNmJkMmJlMzkxZmNmYTAzMjQuYmluZFBvcHVwKHBvcHVwXzAzYzc5NDgwZDc2NzQ1YzJhMGZhYjc0NmQyNjhlYTkyKQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lNTJhODdmNjU1NGE0YjlmOWE3NWM2ZTAyZmYzMGU1ZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEzLjExNDYsIDgwLjE2NThdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzNkODRhZCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzZDg0YWQiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xM2Q2ZDNjMjE5YWQ0NzAzOTVmMmM1NjlhZTQxNzNkZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kZGMyZmRmMGUzZmI0YWRiOGU3ZjI4M2UyY2YyZjM0OCA9ICQoYDxkaXYgaWQ9Imh0bWxfZGRjMmZkZjBlM2ZiNGFkYjhlN2YyODNlMmNmMmYzNDgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlBhbW1hbCBDbHVzdGVyIDEuMDwvZGl2PmApWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMTNkNmQzYzIxOWFkNDcwMzk1ZjJjNTY5YWU0MTczZGYuc2V0Q29udGVudChodG1sX2RkYzJmZGYwZTNmYjRhZGI4ZTdmMjgzZTJjZjJmMzQ4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2U1MmE4N2Y2NTU0YTRiOWY5YTc1YzZlMDJmZjMwZTVmLmJpbmRQb3B1cChwb3B1cF8xM2Q2ZDNjMjE5YWQ0NzAzOTVmMmM1NjlhZTQxNzNkZikKICAgICAgICAgICAgOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMzQ0YThmZWEyYjZiNDdmN2E0MDdjZmM2ZWJmMDY0MjQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFsxMi45ODk1MzcsIDgwLjE4NjI5MV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmZmZjAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmZmYwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICAuYWRkVG8obWFwXzAyMzdjOTQxNzdhMzQ4ZmY5MzQzMWQxYTg4NmQwMWUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2E5ZWU1Mzc4YTNkMDQxODg4YWI1MTlmZDM1OWFiNzExID0gTC5wb3B1cCh7bWF4V2lkdGg6ICcxMDAlJwogICAgICAgICAgICAKICAgICAgICAgICAgfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzg5NTg0NGJmZmRjMzQxMjU4MzIyYWJmOWI1MDc1MWI3ID0gJChgPGRpdiBpZD0iaHRtbF84OTU4NDRiZmZkYzM0MTI1ODMyMmFiZjliNTA3NTFiNyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UGFyayBUb3duIENsdXN0ZXIgMy4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hOWVlNTM3OGEzZDA0MTg4OGFiNTE5ZmQzNTlhYjcxMS5zZXRDb250ZW50KGh0bWxfODk1ODQ0YmZmZGMzNDEyNTgzMjJhYmY5YjUwNzUxYjcpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMzQ0YThmZWEyYjZiNDdmN2E0MDdjZmM2ZWJmMDY0MjQuYmluZFBvcHVwKHBvcHVwX2E5ZWU1Mzc4YTNkMDQxODg4YWI1MTlmZDM1OWFiNzExKQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hY2Q2NmFlZjZiMjM0M2QyYjBjYjMyN2M4Y2MzMzc5MSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEyLjkxMTEyNiwgODAuMTAzNjM2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMzZDg0YWQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjM2Q4NGFkIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMDMyZGE5NzI4Yjg1NGRlNGI5ZjYyZjUwNDQ3NTJmN2EgPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfN2VjOWFkNzg0NzFkNGM3NGFkOTU2ZWQ1NDkwMTA4N2YgPSAkKGA8ZGl2IGlkPSJodG1sXzdlYzlhZDc4NDcxZDRjNzRhZDk1NmVkNTQ5MDEwODdmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5QYXJyeSYjMzk7cyBDb3JuZXIgQ2x1c3RlciAxLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzAzMmRhOTcyOGI4NTRkZTRiOWY2MmY1MDQ0NzUyZjdhLnNldENvbnRlbnQoaHRtbF83ZWM5YWQ3ODQ3MWQ0Yzc0YWQ5NTZlZDU0OTAxMDg3Zik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hY2Q2NmFlZjZiMjM0M2QyYjBjYjMyN2M4Y2MzMzc5MS5iaW5kUG9wdXAocG9wdXBfMDMyZGE5NzI4Yjg1NGRlNGI5ZjYyZjUwNDQ3NTJmN2EpCiAgICAgICAgICAgIDsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2I3ZmVhMjUwOTM4ZTRkZGY4MjMzYzZjNDRiOTYzZDQ1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMTMuMTA3MjcsIDgwLjI0NDQ4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMzZDg0YWQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjM2Q4NGFkIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMGQxYjQ4ZTQ1MWMxNGU2ZDkyNDNlYzI1ZmY3N2Y4ODMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZTI4YWM4NjJkOTI2NDFhYTg2NDQ5YTQ2ZWRmYmUxNzEgPSAkKGA8ZGl2IGlkPSJodG1sX2UyOGFjODYyZDkyNjQxYWE4NjQ0OWE0NmVkZmJlMTcxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5QYXR0YWJpcmFtIENsdXN0ZXIgMS4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wZDFiNDhlNDUxYzE0ZTZkOTI0M2VjMjVmZjc3Zjg4My5zZXRDb250ZW50KGh0bWxfZTI4YWM4NjJkOTI2NDFhYTg2NDQ5YTQ2ZWRmYmUxNzEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYjdmZWEyNTA5MzhlNGRkZjgyMzNjNmM0NGI5NjNkNDUuYmluZFBvcHVwKHBvcHVwXzBkMWI0OGU0NTFjMTRlNmQ5MjQzZWMyNWZmNzdmODgzKQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hZjRjY2MyMDUwZjQ0MDQxOGUxOGFmOWY4ZGU4M2JlMSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEzLjExNjAxMSwgODAuMjMxNjAzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMzZDg0YWQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjM2Q4NGFkIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMjRiMjM0ZTQzMmRjNGRmZmJjYjRiNzhiNDNmOTY3ZWEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOWY5MTcxZWE5YThlNDJiYWE1MmNiOWE2MmYyYWVmYjMgPSAkKGA8ZGl2IGlkPSJodG1sXzlmOTE3MWVhOWE4ZTQyYmFhNTJjYjlhNjJmMmFlZmIzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5QYXR0YXJhdmFra2FtIENsdXN0ZXIgMS4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yNGIyMzRlNDMyZGM0ZGZmYmNiNGI3OGI0M2Y5NjdlYS5zZXRDb250ZW50KGh0bWxfOWY5MTcxZWE5YThlNDJiYWE1MmNiOWE2MmYyYWVmYjMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYWY0Y2NjMjA1MGY0NDA0MThlMThhZjlmOGRlODNiZTEuYmluZFBvcHVwKHBvcHVwXzI0YjIzNGU0MzJkYzRkZmZiY2I0Yjc4YjQzZjk2N2VhKQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kNzZiYjYxZmZhMTg0ZTllOTJhOTRlOTE0NjkwM2FlNSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEyLjkwNTMsIDgwLjE5ODZdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzNkODRhZCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzZDg0YWQiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF84ZWQwNjlhZGU5MWY0Zjg5OTYxZjQ1MDI2MWNiMWJkMyA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iZGUyZTllYTk2MDM0N2Y1OTg0MTIzZjJhNGY0MDdhZCA9ICQoYDxkaXYgaWQ9Imh0bWxfYmRlMmU5ZWE5NjAzNDdmNTk4NDEyM2YyYTRmNDA3YWQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlBhemhhdmFudGhhbmdhbCBDbHVzdGVyIDEuMDwvZGl2PmApWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOGVkMDY5YWRlOTFmNGY4OTk2MWY0NTAyNjFjYjFiZDMuc2V0Q29udGVudChodG1sX2JkZTJlOWVhOTYwMzQ3ZjU5ODQxMjNmMmE0ZjQwN2FkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2Q3NmJiNjFmZmExODRlOWU5MmE5NGU5MTQ2OTAzYWU1LmJpbmRQb3B1cChwb3B1cF84ZWQwNjlhZGU5MWY0Zjg5OTYxZjQ1MDI2MWNiMWJkMykKICAgICAgICAgICAgOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZjUxM2QwM2NlMGE2NGQ2Yjg0N2FhZGFlZGIwYjYwMGUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFsxMi45MDUxLCA4MC4wOTQ4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZmZmMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmZmZjAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYTFhYmEwNTM4ZjJlNDhmMzhhYzExNWMwOTYzYWYwMTMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMWYwZTIxMzJlZDIzNDY5OGJlNjZlY2Q3NjQ1NmVlZjMgPSAkKGA8ZGl2IGlkPSJodG1sXzFmMGUyMTMyZWQyMzQ2OThiZTY2ZWNkNzY0NTZlZWYzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5QZWVya2Fua2FyYW5haSBDbHVzdGVyIDMuMDwvZGl2PmApWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYTFhYmEwNTM4ZjJlNDhmMzhhYzExNWMwOTYzYWYwMTMuc2V0Q29udGVudChodG1sXzFmMGUyMTMyZWQyMzQ2OThiZTY2ZWNkNzY0NTZlZWYzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2Y1MTNkMDNjZTBhNjRkNmI4NDdhYWRhZWRiMGI2MDBlLmJpbmRQb3B1cChwb3B1cF9hMWFiYTA1MzhmMmU0OGYzOGFjMTE1YzA5NjNhZjAxMykKICAgICAgICAgICAgOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMjI3NWMzZDkyYzgzNDBkMjk2MjE3YTMwYTU3NWQ4MjEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFsxMi45NywgODAuMjVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lN2M0OWZiYzc3OTk0OTM2OTM4NDYyYzcyMGFlNGQyZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lZjdkZWQzNGY0MzU0ODUzOTM0NTA3YmIwZmQ0N2I3YyA9ICQoYDxkaXYgaWQ9Imh0bWxfZWY3ZGVkMzRmNDM1NDg1MzkzNDUwN2JiMGZkNDdiN2MiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlBlcmFtYnVyIENsdXN0ZXIgMC4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lN2M0OWZiYzc3OTk0OTM2OTM4NDYyYzcyMGFlNGQyZC5zZXRDb250ZW50KGh0bWxfZWY3ZGVkMzRmNDM1NDg1MzkzNDUwN2JiMGZkNDdiN2MpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMjI3NWMzZDkyYzgzNDBkMjk2MjE3YTMwYTU3NWQ4MjEuYmluZFBvcHVwKHBvcHVwX2U3YzQ5ZmJjNzc5OTQ5MzY5Mzg0NjJjNzIwYWU0ZDJkKQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84NTI4MjcxNmY0ZGE0Y2JjYWM3YjQxYTEyZTQ0MDdmMyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEyLjk4NTM1OSwgODAuMTM5ODM1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMzZDg0YWQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjM2Q4NGFkIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNzI3NDNmODI2NmI3NDM3ODhiN2ZjNTM3MzRmMTEwOTcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfODQ5ZGFmOTg2N2E4NGRkZDljZWQxZTlhOGRiNmE0MTAgPSAkKGA8ZGl2IGlkPSJodG1sXzg0OWRhZjk4NjdhODRkZGQ5Y2VkMWU5YThkYjZhNDEwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5QZXJhdmFsbHVyIENsdXN0ZXIgMS4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83Mjc0M2Y4MjY2Yjc0Mzc4OGI3ZmM1MzczNGYxMTA5Ny5zZXRDb250ZW50KGh0bWxfODQ5ZGFmOTg2N2E4NGRkZDljZWQxZTlhOGRiNmE0MTApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfODUyODI3MTZmNGRhNGNiY2FjN2I0MWExMmU0NDA3ZjMuYmluZFBvcHVwKHBvcHVwXzcyNzQzZjgyNjZiNzQzNzg4YjdmYzUzNzM0ZjExMDk3KQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lZDAxM2M2NzA3Yjk0NjA1YjUxMDBiNTIwMDdkOWQ2ZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEzLjA1LCA4MC4xMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjM2Q4NGFkIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzNkODRhZCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICAuYWRkVG8obWFwXzAyMzdjOTQxNzdhMzQ4ZmY5MzQzMWQxYTg4NmQwMWUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzI2OWQzZWY5YWMxZjRhODI4ZjZiNDhlMWMwNTNlZmFmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICcxMDAlJwogICAgICAgICAgICAKICAgICAgICAgICAgfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2E2ZGYyMWQ2YTRiYTQyN2JiYTE2NmZhYWFiYTZkZDkxID0gJChgPGRpdiBpZD0iaHRtbF9hNmRmMjFkNmE0YmE0MjdiYmExNjZmYWFhYmE2ZGQ5MSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UGVydW1iYWtrYW0gQ2x1c3RlciAxLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzI2OWQzZWY5YWMxZjRhODI4ZjZiNDhlMWMwNTNlZmFmLnNldENvbnRlbnQoaHRtbF9hNmRmMjFkNmE0YmE0MjdiYmExNjZmYWFhYmE2ZGQ5MSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lZDAxM2M2NzA3Yjk0NjA1YjUxMDBiNTIwMDdkOWQ2ZC5iaW5kUG9wdXAocG9wdXBfMjY5ZDNlZjlhYzFmNGE4MjhmNmI0OGUxYzA1M2VmYWYpCiAgICAgICAgICAgIDsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzIwNmY4NTcxOGU2NTQyNjk4YmNjNzM4YzRhMWI4ZTAzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMTMuMDM0NzgsIDgwLjE1NTg2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfODk1ZmNmMjQyMzNlNDMzMDliOTNlMThiMTM4ZWMzMWQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNDFjOWZmMDk4ZTZiNDM0NmFlZmY5ZTVlNDk0Mjc3YzkgPSAkKGA8ZGl2IGlkPSJodG1sXzQxYzlmZjA5OGU2YjQzNDZhZWZmOWU1ZTQ5NDI3N2M5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5QZXJ1bmdhbGF0aHVyIENsdXN0ZXIgMC4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF84OTVmY2YyNDIzM2U0MzMwOWI5M2UxOGIxMzhlYzMxZC5zZXRDb250ZW50KGh0bWxfNDFjOWZmMDk4ZTZiNDM0NmFlZmY5ZTVlNDk0Mjc3YzkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMjA2Zjg1NzE4ZTY1NDI2OThiY2M3MzhjNGExYjhlMDMuYmluZFBvcHVwKHBvcHVwXzg5NWZjZjI0MjMzZTQzMzA5YjkzZTE4YjEzOGVjMzFkKQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iNzY1Y2NhZWIzMjc0ZGY0OWI4YTg5N2ZkODcxNTVhZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEzLjA5OTA1LCA4MC4yNjQwNjVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzNkODRhZCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzZDg0YWQiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xNDIwODUzN2VjN2I0NjJhOWI1NDcyOTdmZDg4OThiZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82Y2YwYTRiNDQ4ZGU0NzM1YWRjMGE4YjVlYzc1YjY3MiA9ICQoYDxkaXYgaWQ9Imh0bWxfNmNmMGE0YjQ0OGRlNDczNWFkYzBhOGI1ZWM3NWI2NzIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlBlcnVuZ3VkaSBDbHVzdGVyIDEuMDwvZGl2PmApWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMTQyMDg1MzdlYzdiNDYyYTliNTQ3Mjk3ZmQ4ODk4YmQuc2V0Q29udGVudChodG1sXzZjZjBhNGI0NDhkZTQ3MzVhZGMwYThiNWVjNzViNjcyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2I3NjVjY2FlYjMyNzRkZjQ5YjhhODk3ZmQ4NzE1NWFkLmJpbmRQb3B1cChwb3B1cF8xNDIwODUzN2VjN2I0NjJhOWI1NDcyOTdmZDg4OThiZCkKICAgICAgICAgICAgOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNDMzNDhjZDdlZDcwNDNiNmFhNTJhNjQzMmY4ZGZmNTEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFsxMy4wODU5MywgODAuMjUwNDddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kMTg0NTE4ZDAyYzM0MWEzOWUyYzk2YzRkODhlMDE0MiA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zYzM2NGYwNzE5YjY0NGRhYTg4MjgyZmUwYjJhMWE1YSA9ICQoYDxkaXYgaWQ9Imh0bWxfM2MzNjRmMDcxOWI2NDRkYWE4ODI4MmZlMGIyYTFhNWEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlBvemhpY2hhbHVyIENsdXN0ZXIgMC4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kMTg0NTE4ZDAyYzM0MWEzOWUyYzk2YzRkODhlMDE0Mi5zZXRDb250ZW50KGh0bWxfM2MzNjRmMDcxOWI2NDRkYWE4ODI4MmZlMGIyYTFhNWEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNDMzNDhjZDdlZDcwNDNiNmFhNTJhNjQzMmY4ZGZmNTEuYmluZFBvcHVwKHBvcHVwX2QxODQ1MThkMDJjMzQxYTM5ZTJjOTZjNGQ4OGUwMTQyKQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83ZDA1MzI2NjExMmQ0NjU4OTBiMTdjYTI4OTBmYjMzNiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEzLjE0MDI1MSwgODAuMTg4MDhdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzAwMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMwMDAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81ZGJkYjMzNzNiZmQ0OTQ2OGM1NTFhMzFmNDQ1ZmJhMSA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lZjQ1YzlhMDBhN2I0Zjg1YTk4YjU3NTAwYmZhNjUwOSA9ICQoYDxkaXYgaWQ9Imh0bWxfZWY0NWM5YTAwYTdiNGY4NWE5OGI1NzUwMGJmYTY1MDkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlBvb25hbWFsbGVlIENsdXN0ZXIgMi4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81ZGJkYjMzNzNiZmQ0OTQ2OGM1NTFhMzFmNDQ1ZmJhMS5zZXRDb250ZW50KGh0bWxfZWY0NWM5YTAwYTdiNGY4NWE5OGI1NzUwMGJmYTY1MDkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfN2QwNTMyNjYxMTJkNDY1ODkwYjE3Y2EyODkwZmIzMzYuYmluZFBvcHVwKHBvcHVwXzVkYmRiMzM3M2JmZDQ5NDY4YzU1MWEzMWY0NDVmYmExKQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yZjVkODcxYWZhN2Y0YzkyOWU3NWFlNDMwOTIwZjM4MCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEzLjE2OTcyLCA4MC4yMDcyMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjM2Q4NGFkIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzNkODRhZCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICAuYWRkVG8obWFwXzAyMzdjOTQxNzdhMzQ4ZmY5MzQzMWQxYTg4NmQwMWUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzVjZWNlZDM1Y2U5ODQ0Mjc5NjgzODM0OTg5ZTVjOWZmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICcxMDAlJwogICAgICAgICAgICAKICAgICAgICAgICAgfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzI5NWM1NGI1MTAxODRmMDZhOWMyMDQxMzIyMTVkMDcyID0gJChgPGRpdiBpZD0iaHRtbF8yOTVjNTRiNTEwMTg0ZjA2YTljMjA0MTMyMjE1ZDA3MiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UG9ydXIgQ2x1c3RlciAxLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzVjZWNlZDM1Y2U5ODQ0Mjc5NjgzODM0OTg5ZTVjOWZmLnNldENvbnRlbnQoaHRtbF8yOTVjNTRiNTEwMTg0ZjA2YTljMjA0MTMyMjE1ZDA3Mik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yZjVkODcxYWZhN2Y0YzkyOWU3NWFlNDMwOTIwZjM4MC5iaW5kUG9wdXAocG9wdXBfNWNlY2VkMzVjZTk4NDQyNzk2ODM4MzQ5ODllNWM5ZmYpCiAgICAgICAgICAgIDsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzA4OTZiYmNhN2E4ZjQ5ZjQ4ZTI5MGQ5OWZiZDU1NGQ5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMTMuMDI1LCA4MC4xODQ3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMzZDg0YWQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjM2Q4NGFkIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMGVhOTM5OTBhZDEwNDU0MmFhMzAzMjY1MTJjYTM0ZjAgPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMWE5YjdmMjZkY2MzNDUyNWFmOTU0ZDIzNDNiM2Q1ODUgPSAkKGA8ZGl2IGlkPSJodG1sXzFhOWI3ZjI2ZGNjMzQ1MjVhZjk1NGQyMzQzYjNkNTg1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5QdWxpYW50aG9wZSBDbHVzdGVyIDEuMDwvZGl2PmApWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMGVhOTM5OTBhZDEwNDU0MmFhMzAzMjY1MTJjYTM0ZjAuc2V0Q29udGVudChodG1sXzFhOWI3ZjI2ZGNjMzQ1MjVhZjk1NGQyMzQzYjNkNTg1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzA4OTZiYmNhN2E4ZjQ5ZjQ4ZTI5MGQ5OWZiZDU1NGQ5LmJpbmRQb3B1cChwb3B1cF8wZWE5Mzk5MGFkMTA0NTQyYWEzMDMyNjUxMmNhMzRmMCkKICAgICAgICAgICAgOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYTQwYTZkOTk0YjI3NDMyNGE4YmYyOWFiY2U4YjczYTEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFsxMy4wODMsIDgwLjI2N10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjM2Q4NGFkIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzNkODRhZCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICAuYWRkVG8obWFwXzAyMzdjOTQxNzdhMzQ4ZmY5MzQzMWQxYTg4NmQwMWUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzExYjdkN2UwNzRlNTQ0NGY4N2NkNjI5Y2RjZWRiZDcwID0gTC5wb3B1cCh7bWF4V2lkdGg6ICcxMDAlJwogICAgICAgICAgICAKICAgICAgICAgICAgfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzk2OWQwN2I0ODFlZDQ2ZDFiZjM5YjJjZjA3YjE2MzUwID0gJChgPGRpdiBpZD0iaHRtbF85NjlkMDdiNDgxZWQ0NmQxYmYzOWIyY2YwN2IxNjM1MCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UHV0aGFnYXJhbSBDbHVzdGVyIDEuMDwvZGl2PmApWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMTFiN2Q3ZTA3NGU1NDQ0Zjg3Y2Q2MjljZGNlZGJkNzAuc2V0Q29udGVudChodG1sXzk2OWQwN2I0ODFlZDQ2ZDFiZjM5YjJjZjA3YjE2MzUwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2E0MGE2ZDk5NGIyNzQzMjRhOGJmMjlhYmNlOGI3M2ExLmJpbmRQb3B1cChwb3B1cF8xMWI3ZDdlMDc0ZTU0NDRmODdjZDYyOWNkY2VkYmQ3MCkKICAgICAgICAgICAgOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZGY0MTc4ZjBlMTM2NDRhZjk4MWNmYjQwMTUwZjNmMDMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFsxMy4xMDQwMywgODAuMjkzNjZdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzNkODRhZCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzZDg0YWQiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8wODIxNWU4Yzc0YTg0MmE5YmIyZWE0MDE2ZjM3N2ZkMiA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kOThiZGE3OGM0ZmI0OTNlOGE2M2YyZDYzZjI1MDVmZCA9ICQoYDxkaXYgaWQ9Imh0bWxfZDk4YmRhNzhjNGZiNDkzZThhNjNmMmQ2M2YyNTA1ZmQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlB1emhhbCBDbHVzdGVyIDEuMDwvZGl2PmApWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMDgyMTVlOGM3NGE4NDJhOWJiMmVhNDAxNmYzNzdmZDIuc2V0Q29udGVudChodG1sX2Q5OGJkYTc4YzRmYjQ5M2U4YTYzZjJkNjNmMjUwNWZkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2RmNDE3OGYwZTEzNjQ0YWY5ODFjZmI0MDE1MGYzZjAzLmJpbmRQb3B1cChwb3B1cF8wODIxNWU4Yzc0YTg0MmE5YmIyZWE0MDE2ZjM3N2ZkMikKICAgICAgICAgICAgOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMmMzMGIyNzA4ZDgwNDNjNWI1Y2VlYzJlY2Q3MTdkNzUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFsxMy4wMjM1LCA4MC4yMjM3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNjI0NDIyZWJkYzY2NDkwNmIzMmYwMzM1MDI0Y2E1MDAgPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYzcxYjJhMTk4YmNiNGE2ZDlhNzBkYjJiMmY4MGVmMmMgPSAkKGA8ZGl2IGlkPSJodG1sX2M3MWIyYTE5OGJjYjRhNmQ5YTcwZGIyYjJmODBlZjJjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5QdXpodXRoaXZha2thbS8gVWxsYWdhcmFtIENsdXN0ZXIgMC4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82MjQ0MjJlYmRjNjY0OTA2YjMyZjAzMzUwMjRjYTUwMC5zZXRDb250ZW50KGh0bWxfYzcxYjJhMTk4YmNiNGE2ZDlhNzBkYjJiMmY4MGVmMmMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMmMzMGIyNzA4ZDgwNDNjNWI1Y2VlYzJlY2Q3MTdkNzUuYmluZFBvcHVwKHBvcHVwXzYyNDQyMmViZGM2NjQ5MDZiMzJmMDMzNTAyNGNhNTAwKQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80MGQ4Zjc0Yjg5Nzg0YjYzYTAwMjkzZGVhYjQxNmNjZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEzLjA0ODEsIDgwLjIwMTRdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85ZTk4YzAxZDQ3NzE0NWE3OTU3MzhlYTczYWQ2YjAxNSA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83Y2FlMmNiZjU2MGU0MDliYWUxZDg5MDIzMjk4YTJjMiA9ICQoYDxkaXYgaWQ9Imh0bWxfN2NhZTJjYmY1NjBlNDA5YmFlMWQ4OTAyMzI5OGEyYzIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJhaiBCaGF2YW4gQ2x1c3RlciAwLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzllOThjMDFkNDc3MTQ1YTc5NTczOGVhNzNhZDZiMDE1LnNldENvbnRlbnQoaHRtbF83Y2FlMmNiZjU2MGU0MDliYWUxZDg5MDIzMjk4YTJjMik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80MGQ4Zjc0Yjg5Nzg0YjYzYTAwMjkzZGVhYjQxNmNjZC5iaW5kUG9wdXAocG9wdXBfOWU5OGMwMWQ0NzcxNDVhNzk1NzM4ZWE3M2FkNmIwMTUpCiAgICAgICAgICAgIDsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzFjMTM4ZDIwYzg2NzQ3YTdiMGIwYTM2ZWMyN2FmMGMzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMTMuMDMwMiwgODAuMjc4N10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjM2Q4NGFkIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzNkODRhZCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICAuYWRkVG8obWFwXzAyMzdjOTQxNzdhMzQ4ZmY5MzQzMWQxYTg4NmQwMWUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2E5ODE5MGUwN2Q3NzRjNTQ4YjI3MTViOTM5ODViZDQ0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICcxMDAlJwogICAgICAgICAgICAKICAgICAgICAgICAgfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2NhZWYyNWVkMDRmNDRmNzM4NzZlZTgwNWI1YjEyMTE2ID0gJChgPGRpdiBpZD0iaHRtbF9jYWVmMjVlZDA0ZjQ0ZjczODc2ZWU4MDViNWIxMjExNiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UmFtYXZhcmFtIENsdXN0ZXIgMS4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hOTgxOTBlMDdkNzc0YzU0OGIyNzE1YjkzOTg1YmQ0NC5zZXRDb250ZW50KGh0bWxfY2FlZjI1ZWQwNGY0NGY3Mzg3NmVlODA1YjViMTIxMTYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMWMxMzhkMjBjODY3NDdhN2IwYjBhMzZlYzI3YWYwYzMuYmluZFBvcHVwKHBvcHVwX2E5ODE5MGUwN2Q3NzRjNTQ4YjI3MTViOTM5ODViZDQ0KQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mNmFlNDk4ZjQ4ZDA0ZGY4ODlkYjllYWQyMDRmZGU4NiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEyLjkyNjM5LCA4MC4xNjVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzNkODRhZCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzZDg0YWQiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mMzhjODU4NDE3ZTQ0ZjhkODEwYzU2Y2ExODM2YWU3MyA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wMjNkNjhjYjBlNDA0NjNlYjY4NzVjNWNmMmM2MjBkNCA9ICQoYDxkaXYgaWQ9Imh0bWxfMDIzZDY4Y2IwZTQwNDYzZWI2ODc1YzVjZjJjNjIwZDQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJlZCBIaWxscyBDbHVzdGVyIDEuMDwvZGl2PmApWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZjM4Yzg1ODQxN2U0NGY4ZDgxMGM1NmNhMTgzNmFlNzMuc2V0Q29udGVudChodG1sXzAyM2Q2OGNiMGU0MDQ2M2ViNjg3NWM1Y2YyYzYyMGQ0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2Y2YWU0OThmNDhkMDRkZjg4OWRiOWVhZDIwNGZkZTg2LmJpbmRQb3B1cChwb3B1cF9mMzhjODU4NDE3ZTQ0ZjhkODEwYzU2Y2ExODM2YWU3MykKICAgICAgICAgICAgOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOTEzMGIzZmFmYzRlNDExZmIxNTY4NTVjMjA5MWJmYmUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFsxMi45MywgODAuMTRdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzNkODRhZCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzZDg0YWQiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80MDY5YmViOTUwMzE0YTk0Yjg1ZTNiMGQ0M2JiODgyOSA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zYWRkZDUxZTNhYjM0ZDg2YmI2NTlkYzI0NDkzNTJjNiA9ICQoYDxkaXYgaWQ9Imh0bWxfM2FkZGQ1MWUzYWIzNGQ4NmJiNjU5ZGMyNDQ5MzUyYzYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJveWFwZXR0YWggQ2x1c3RlciAxLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzQwNjliZWI5NTAzMTRhOTRiODVlM2IwZDQzYmI4ODI5LnNldENvbnRlbnQoaHRtbF8zYWRkZDUxZTNhYjM0ZDg2YmI2NTlkYzI0NDkzNTJjNik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85MTMwYjNmYWZjNGU0MTFmYjE1Njg1NWMyMDkxYmZiZS5iaW5kUG9wdXAocG9wdXBfNDA2OWJlYjk1MDMxNGE5NGI4NWUzYjBkNDNiYjg4MjkpCiAgICAgICAgICAgIDsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQyNWZhNDIwZThmYTQzMjc5ZWFkZjA4YzIxYzNkNzU4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMTMuMDc5NCwgODAuMjI4OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjM2Q4NGFkIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzNkODRhZCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICAuYWRkVG8obWFwXzAyMzdjOTQxNzdhMzQ4ZmY5MzQzMWQxYTg4NmQwMWUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzUyZjZhZmZhNzJjOTRmYTJhNDg0YWI4Njk1OTg0ZWU4ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICcxMDAlJwogICAgICAgICAgICAKICAgICAgICAgICAgfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2E1NmYzOWM0NTM5ODQzNWRhNDM5YTQ0ZjM4MmU0MGJkID0gJChgPGRpdiBpZD0iaHRtbF9hNTZmMzljNDUzOTg0MzVkYTQzOWE0NGYzODJlNDBiZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Um95YXB1cmFtIENsdXN0ZXIgMS4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81MmY2YWZmYTcyYzk0ZmEyYTQ4NGFiODY5NTk4NGVlOC5zZXRDb250ZW50KGh0bWxfYTU2ZjM5YzQ1Mzk4NDM1ZGE0MzlhNDRmMzgyZTQwYmQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNDI1ZmE0MjBlOGZhNDMyNzllYWRmMDhjMjFjM2Q3NTguYmluZFBvcHVwKHBvcHVwXzUyZjZhZmZhNzJjOTRmYTJhNDg0YWI4Njk1OTg0ZWU4KQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jM2NkYjhkZTNjOGY0ZTdiYTdlZDY5YjE4YjU1MmY2MSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEyLjkwMDE1LCA4MC4yMjc5MV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICAuYWRkVG8obWFwXzAyMzdjOTQxNzdhMzQ4ZmY5MzQzMWQxYTg4NmQwMWUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzdlOGFmMzE3ODMxMTQ1OWM5YWFlZGJmZGJiODZjYTExID0gTC5wb3B1cCh7bWF4V2lkdGg6ICcxMDAlJwogICAgICAgICAgICAKICAgICAgICAgICAgfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzUzODA1NGE0MTNkMTRkZTRhM2I4MjBlYzhmODkxYTE5ID0gJChgPGRpdiBpZD0iaHRtbF81MzgwNTRhNDEzZDE0ZGU0YTNiODIwZWM4Zjg5MWExOSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U2FsaWdyYW1hbSBDbHVzdGVyIDAuMDwvZGl2PmApWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfN2U4YWYzMTc4MzExNDU5YzlhYWVkYmZkYmI4NmNhMTEuc2V0Q29udGVudChodG1sXzUzODA1NGE0MTNkMTRkZTRhM2I4MjBlYzhmODkxYTE5KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2MzY2RiOGRlM2M4ZjRlN2JhN2VkNjliMThiNTUyZjYxLmJpbmRQb3B1cChwb3B1cF83ZThhZjMxNzgzMTE0NTljOWFhZWRiZmRiYjg2Y2ExMSkKICAgICAgICAgICAgOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZTg5NmU3ZDAxZTZiNDA1MGI4NmIzNGQ5NWE4MDdkYTcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFsxMy4wODk1LCA4MC4yNzg5N10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICAuYWRkVG8obWFwXzAyMzdjOTQxNzdhMzQ4ZmY5MzQzMWQxYTg4NmQwMWUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzRhZmY0NWUwZjQwYTRkY2NiYjkxYjMxNzhkZjI2NTFmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICcxMDAlJwogICAgICAgICAgICAKICAgICAgICAgICAgfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzlkNmU1ZWEwMDdmOTQ1ZTc4MTFlODI3MmU5NGIxMGMyID0gJChgPGRpdiBpZD0iaHRtbF85ZDZlNWVhMDA3Zjk0NWU3ODExZTgyNzJlOTRiMTBjMiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U2VtYmFra2FtIENsdXN0ZXIgMC4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80YWZmNDVlMGY0MGE0ZGNjYmI5MWIzMTc4ZGYyNjUxZi5zZXRDb250ZW50KGh0bWxfOWQ2ZTVlYTAwN2Y5NDVlNzgxMWU4MjcyZTk0YjEwYzIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZTg5NmU3ZDAxZTZiNDA1MGI4NmIzNGQ5NWE4MDdkYTcuYmluZFBvcHVwKHBvcHVwXzRhZmY0NWUwZjQwYTRkY2NiYjkxYjMxNzhkZjI2NTFmKQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84NDA0OGEyZjM2MTI0ZWViYmE4MTVjZjdiYTU4MGMzZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEzLjAwNTA1NiwgODAuMTkzMzA2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZmZmMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmZmZjAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZWRmNjFhMzQyYWIzNGY5YTkxZGIzMDExNjc0NjhhMDMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOWQ5NjBjZDE5ZjQ5NDUwZWJjM2QyZTBhM2E5ZWVhNGMgPSAkKGA8ZGl2IGlkPSJodG1sXzlkOTYwY2QxOWY0OTQ1MGViYzNkMmUwYTNhOWVlYTRjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TZWxhaXl1ciBDbHVzdGVyIDMuMDwvZGl2PmApWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZWRmNjFhMzQyYWIzNGY5YTkxZGIzMDExNjc0NjhhMDMuc2V0Q29udGVudChodG1sXzlkOTYwY2QxOWY0OTQ1MGViYzNkMmUwYTNhOWVlYTRjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzg0MDQ4YTJmMzYxMjRlZWJiYTgxNWNmN2JhNTgwYzNmLmJpbmRQb3B1cChwb3B1cF9lZGY2MWEzNDJhYjM0ZjlhOTFkYjMwMTE2NzQ2OGEwMykKICAgICAgICAgICAgOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYzZiNWJlZmI0MDNkNGJkZjkwZGY2MWU3MjUwMTQwZTQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFsxMi45MywgODAuMTFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzNkODRhZCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzZDg0YWQiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hNzE0MDQ1M2I4M2Q0ODBjOGEyZmQyNDhhZTQ4Yzc4ZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84MDdhZjdkNDJmMDk0NTUxYTRlMWNlMzgyNGQ5ZGRiZCA9ICQoYDxkaXYgaWQ9Imh0bWxfODA3YWY3ZDQyZjA5NDU1MWE0ZTFjZTM4MjRkOWRkYmQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlNoZW5veSBOYWdhciBDbHVzdGVyIDEuMDwvZGl2PmApWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYTcxNDA0NTNiODNkNDgwYzhhMmZkMjQ4YWU0OGM3OGUuc2V0Q29udGVudChodG1sXzgwN2FmN2Q0MmYwOTQ1NTFhNGUxY2UzODI0ZDlkZGJkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2M2YjViZWZiNDAzZDRiZGY5MGRmNjFlNzI1MDE0MGU0LmJpbmRQb3B1cChwb3B1cF9hNzE0MDQ1M2I4M2Q0ODBjOGEyZmQyNDhhZTQ4Yzc4ZSkKICAgICAgICAgICAgOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMzEyOGJmNzJlYjBlNGE1Zjg0OTI0YzRhMGJlYmJiMTAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFsxMy4wNDM3LCA4MC4yNTIzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNjYyMTI0M2M0NzdiNGY5ZDlmMDdhOTgxNzljY2VlNWUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfODAzOTg0ZTU4ODdkNDYwNmIwMDZmZjYwZDk2ZDc3MjMgPSAkKGA8ZGl2IGlkPSJodG1sXzgwMzk4NGU1ODg3ZDQ2MDZiMDA2ZmY2MGQ5NmQ3NzIzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TaG9sYXZhcmFtIENsdXN0ZXIgMC4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82NjIxMjQzYzQ3N2I0ZjlkOWYwN2E5ODE3OWNjZWU1ZS5zZXRDb250ZW50KGh0bWxfODAzOTg0ZTU4ODdkNDYwNmIwMDZmZjYwZDk2ZDc3MjMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMzEyOGJmNzJlYjBlNGE1Zjg0OTI0YzRhMGJlYmJiMTAuYmluZFBvcHVwKHBvcHVwXzY2MjEyNDNjNDc3YjRmOWQ5ZjA3YTk4MTc5Y2NlZTVlKQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82N2E5YzkyZDRkNzk0MDc0OGIxMTlhMDEzYjgwYzk0ZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEyLjk3ODYsIDgwLjI0MDldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8wNjI0NTZlMmEzNWI0MzI5OTYxNmFkN2I1ZjNmMmY2ZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iNTJlMDNkMGQ1ZWM0NDNiYWE3ZDdiYjI2N2RlMGE0YSA9ICQoYDxkaXYgaWQ9Imh0bWxfYjUyZTAzZDBkNWVjNDQzYmFhN2Q3YmIyNjdkZTBhNGEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlNob2xpbmdhbmFsbHVyIENsdXN0ZXIgMC4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wNjI0NTZlMmEzNWI0MzI5OTYxNmFkN2I1ZjNmMmY2ZS5zZXRDb250ZW50KGh0bWxfYjUyZTAzZDBkNWVjNDQzYmFhN2Q3YmIyNjdkZTBhNGEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNjdhOWM5MmQ0ZDc5NDA3NDhiMTE5YTAxM2I4MGM5NGQuYmluZFBvcHVwKHBvcHVwXzA2MjQ1NmUyYTM1YjQzMjk5NjE2YWQ3YjVmM2YyZjZlKQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85MjMyZjllY2Y0MzE0ZDEyYWViZWE0ZWE0MzRjNTViOCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEzLjAzNDE2LCA4MC4yMzAwNl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjM2Q4NGFkIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzNkODRhZCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICAuYWRkVG8obWFwXzAyMzdjOTQxNzdhMzQ4ZmY5MzQzMWQxYTg4NmQwMWUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2IyMTEyYWFkNGEzOTRiM2M5MGQyZmYzMWYyMjYwYmEyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICcxMDAlJwogICAgICAgICAgICAKICAgICAgICAgICAgfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzYxYjE2NzQ2MDFjNjRhNzQ4NjVmOWFjMTBmMzJiZDljID0gJChgPGRpdiBpZD0iaHRtbF82MWIxNjc0NjAxYzY0YTc0ODY1ZjlhYzEwZjMyYmQ5YyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U2l0aGFsYXBha2thbSBDbHVzdGVyIDEuMDwvZGl2PmApWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYjIxMTJhYWQ0YTM5NGIzYzkwZDJmZjMxZjIyNjBiYTIuc2V0Q29udGVudChodG1sXzYxYjE2NzQ2MDFjNjRhNzQ4NjVmOWFjMTBmMzJiZDljKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzkyMzJmOWVjZjQzMTRkMTJhZWJlYTRlYTQzNGM1NWI4LmJpbmRQb3B1cChwb3B1cF9iMjExMmFhZDRhMzk0YjNjOTBkMmZmMzFmMjI2MGJhMikKICAgICAgICAgICAgOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZTM2NDc3ZjJiOTdiNDE4ODljNzAzNDM5MTA3YjJmNzIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFsxMy4wODU0MSwgODAuMTk4NjVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85ZjUxYzQ4N2E3NjA0MDcxYThjYmI1NmY3ZWU3ZTAwZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zMDMyODFmOTVhY2Y0MmUxYjZkYjIzODdmY2M4ZTYwMSA9ICQoYDxkaXYgaWQ9Imh0bWxfMzAzMjgxZjk1YWNmNDJlMWI2ZGIyMzg3ZmNjOGU2MDEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlNvd2NhcnBldCBDbHVzdGVyIDAuMDwvZGl2PmApWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOWY1MWM0ODdhNzYwNDA3MWE4Y2JiNTZmN2VlN2UwMGYuc2V0Q29udGVudChodG1sXzMwMzI4MWY5NWFjZjQyZTFiNmRiMjM4N2ZjYzhlNjAxKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2UzNjQ3N2YyYjk3YjQxODg5YzcwMzQzOTEwN2IyZjcyLmJpbmRQb3B1cChwb3B1cF85ZjUxYzQ4N2E3NjA0MDcxYThjYmI1NmY3ZWU3ZTAwZikKICAgICAgICAgICAgOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfN2EwOWNlMDFkOTdiNDBmMmJmMTA0YjgxNzcwYWJmNDYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFsxMy4xMzE4MywgODAuMTMwODJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzNkODRhZCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzZDg0YWQiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF84ODQ5ZDdlNTRiZjA0ZDMyODU5ZGFjZmUzYzY3ODdmNCA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yM2Q5NGIwOGVlNWE0ZmEzOWRiNTRiNGE1ZjE5YmNjZCA9ICQoYDxkaXYgaWQ9Imh0bWxfMjNkOTRiMDhlZTVhNGZhMzlkYjU0YjRhNWYxOWJjY2QiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN0LlRob21hcyBNb3VudCBDbHVzdGVyIDEuMDwvZGl2PmApWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfODg0OWQ3ZTU0YmYwNGQzMjg1OWRhY2ZlM2M2Nzg3ZjQuc2V0Q29udGVudChodG1sXzIzZDk0YjA4ZWU1YTRmYTM5ZGI1NGI0YTVmMTliY2NkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzdhMDljZTAxZDk3YjQwZjJiZjEwNGI4MTc3MGFiZjQ2LmJpbmRQb3B1cChwb3B1cF84ODQ5ZDdlNTRiZjA0ZDMyODU5ZGFjZmUzYzY3ODdmNCkKICAgICAgICAgICAgOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTcyODFhZTQyNWIwNDdjZGE1NzU4MmI4Yjk5OTBiMWIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFsxMy4xMjM2LCA4MC4wMjc5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZmZmMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmZmZjAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZDBjZDE5MTQ0YmU4NDRlZGFiMDNjM2YxZWRmZDE0MjQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNjk3ZTZmZTkxOWVhNDk1MGEyM2Q3NTcyMTc5Y2FiNWQgPSAkKGA8ZGl2IGlkPSJodG1sXzY5N2U2ZmU5MTllYTQ5NTBhMjNkNzU3MjE3OWNhYjVkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5UZXluYW1wZXQgQ2x1c3RlciAzLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2QwY2QxOTE0NGJlODQ0ZWRhYjAzYzNmMWVkZmQxNDI0LnNldENvbnRlbnQoaHRtbF82OTdlNmZlOTE5ZWE0OTUwYTIzZDc1NzIxNzljYWI1ZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81NzI4MWFlNDI1YjA0N2NkYTU3NTgyYjhiOTk5MGIxYi5iaW5kUG9wdXAocG9wdXBfZDBjZDE5MTQ0YmU4NDRlZGFiMDNjM2YxZWRmZDE0MjQpCiAgICAgICAgICAgIDsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2I1ZDYyYWEyOTg5MTQ3ZGRhM2E4YzM0MTRlY2Q0NDM0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMTIuOTg1NiwgODAuMjYxNF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjM2Q4NGFkIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzNkODRhZCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICAuYWRkVG8obWFwXzAyMzdjOTQxNzdhMzQ4ZmY5MzQzMWQxYTg4NmQwMWUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2EzMjkwYzA0ZDVjNDRlZWY4YzFiZDc0MmM2NzgyOTk4ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICcxMDAlJwogICAgICAgICAgICAKICAgICAgICAgICAgfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzMwMWFjOTA2OGU1YzQ5M2Y4ZjUyNWFiMWQ2MjdmNTM4ID0gJChgPGRpdiBpZD0iaHRtbF8zMDFhYzkwNjhlNWM0OTNmOGY1MjVhYjFkNjI3ZjUzOCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VGhhcmFtYW5pIENsdXN0ZXIgMS4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hMzI5MGMwNGQ1YzQ0ZWVmOGMxYmQ3NDJjNjc4Mjk5OC5zZXRDb250ZW50KGh0bWxfMzAxYWM5MDY4ZTVjNDkzZjhmNTI1YWIxZDYyN2Y1MzgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYjVkNjJhYTI5ODkxNDdkZGEzYThjMzQxNGVjZDQ0MzQuYmluZFBvcHVwKHBvcHVwX2EzMjkwYzA0ZDVjNDRlZWY4YzFiZDc0MmM2NzgyOTk4KQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jZTNhNWEwMDBlNWE0NDY3YjhhZTViN2Q5YzQ5ZTI2MiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEzLjE2LCA4MC4zXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNTJlNjgzNDY2OWMzNDRlYmEwNTYzOWRkMmFhNjM0NWQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMTY0NzRlNWJkYzZlNDVlZTgxN2U2OWY4NDU5OWUyZTAgPSAkKGA8ZGl2IGlkPSJodG1sXzE2NDc0ZTViZGM2ZTQ1ZWU4MTdlNjlmODQ1OTllMmUwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5UaGlydW1hbmdhbGFtIENsdXN0ZXIgMC4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81MmU2ODM0NjY5YzM0NGViYTA1NjM5ZGQyYWE2MzQ1ZC5zZXRDb250ZW50KGh0bWxfMTY0NzRlNWJkYzZlNDVlZTgxN2U2OWY4NDU5OWUyZTApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfY2UzYTVhMDAwZTVhNDQ2N2I4YWU1YjdkOWM0OWUyNjIuYmluZFBvcHVwKHBvcHVwXzUyZTY4MzQ2NjljMzQ0ZWJhMDU2MzlkZDJhYTYzNDVkKQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xMjY5ODIwYzJkZTU0Njg2YjU3MmZhNDkwN2RkOTJiYyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEyLjk4MDE1LCA4MC4xNjU2Nl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjM2Q4NGFkIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzNkODRhZCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICAuYWRkVG8obWFwXzAyMzdjOTQxNzdhMzQ4ZmY5MzQzMWQxYTg4NmQwMWUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzE2YzM4Y2IzZmY3NTQyOTQ5OWM2NTAyZGQ3YmJiNDY3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICcxMDAlJwogICAgICAgICAgICAKICAgICAgICAgICAgfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzRlODBjZGQzOWZmMjQ4YzFiMWFiYjBiNDg4NzU5ZGVhID0gJChgPGRpdiBpZD0iaHRtbF80ZTgwY2RkMzlmZjI0OGMxYjFhYmIwYjQ4ODc1OWRlYSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VGhpcnVtdWxsYWl2b3lhbCBDbHVzdGVyIDEuMDwvZGl2PmApWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMTZjMzhjYjNmZjc1NDI5NDk5YzY1MDJkZDdiYmI0Njcuc2V0Q29udGVudChodG1sXzRlODBjZGQzOWZmMjQ4YzFiMWFiYjBiNDg4NzU5ZGVhKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzEyNjk4MjBjMmRlNTQ2ODZiNTcyZmE0OTA3ZGQ5MmJjLmJpbmRQb3B1cChwb3B1cF8xNmMzOGNiM2ZmNzU0Mjk0OTljNjUwMmRkN2JiYjQ2NykKICAgICAgICAgICAgOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYWY2NzcxMTA0ZjlhNDM0Mjk2YTFiZDQ4MTA0NjUyMGYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFsxMy4wNTUzLCA4MC4yODA3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZmZmMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmZmZjAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZmM5ODFiODEyYzhmNDFlNGI0N2ZjOGFhOTVhMmI2Y2YgPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOWZlNzYzZGYyZWE0NGVjZDllMjBkMWNlNGE0MWFmOGIgPSAkKGA8ZGl2IGlkPSJodG1sXzlmZTc2M2RmMmVhNDRlY2Q5ZTIwZDFjZTRhNDFhZjhiIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5UaGlydW5lZXJtYWxhaSBDbHVzdGVyIDMuMDwvZGl2PmApWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZmM5ODFiODEyYzhmNDFlNGI0N2ZjOGFhOTVhMmI2Y2Yuc2V0Q29udGVudChodG1sXzlmZTc2M2RmMmVhNDRlY2Q5ZTIwZDFjZTRhNDFhZjhiKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2FmNjc3MTEwNGY5YTQzNDI5NmExYmQ0ODEwNDY1MjBmLmJpbmRQb3B1cChwb3B1cF9mYzk4MWI4MTJjOGY0MWU0YjQ3ZmM4YWE5NWEyYjZjZikKICAgICAgICAgICAgOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZGFkODAzNmQzYWViNGU3NDhiYWE1MGRkNGIyYTBhMzYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFsxMy4xMjc1LCA4MC4yODE2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMzZDg0YWQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjM2Q4NGFkIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYjkyMGNlNjYyOWQxNDQ4Mzg5ODY2ZWY2ZDI5MzlkYjMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMWRiYTkxOWQ1MDYwNDgxYWFiYWUzZWMzMDNmMjZjMTQgPSAkKGA8ZGl2IGlkPSJodG1sXzFkYmE5MTlkNTA2MDQ4MWFhYmFlM2VjMzAzZjI2YzE0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5UaGlydW5pbnJhdnVyIENsdXN0ZXIgMS4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iOTIwY2U2NjI5ZDE0NDgzODk4NjZlZjZkMjkzOWRiMy5zZXRDb250ZW50KGh0bWxfMWRiYTkxOWQ1MDYwNDgxYWFiYWUzZWMzMDNmMjZjMTQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZGFkODAzNmQzYWViNGU3NDhiYWE1MGRkNGIyYTBhMzYuYmluZFBvcHVwKHBvcHVwX2I5MjBjZTY2MjlkMTQ0ODM4OTg2NmVmNmQyOTM5ZGIzKQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85Yzg0MTE0ZjY2OTU0ZWZmYjk0ZTUzYWJlMjE4YzM5NSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEzLjA1MjcxMywgODAuMjI2MDk1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMzZDg0YWQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjM2Q4NGFkIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMTllZjFkYjZkOGM1NGJmZWIwZjA4MmJkYjM0ZmJlZTEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNTNlYjc2Y2I5MzJmNGZhMmI0Njg1YjQ5NzYyY2Q4YzYgPSAkKGA8ZGl2IGlkPSJodG1sXzUzZWI3NmNiOTMyZjRmYTJiNDY4NWI0OTc2MmNkOGM2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5UaGlydXZhbm1peXVyIENsdXN0ZXIgMS4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8xOWVmMWRiNmQ4YzU0YmZlYjBmMDgyYmRiMzRmYmVlMS5zZXRDb250ZW50KGh0bWxfNTNlYjc2Y2I5MzJmNGZhMmI0Njg1YjQ5NzYyY2Q4YzYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOWM4NDExNGY2Njk1NGVmZmI5NGU1M2FiZTIxOGMzOTUuYmluZFBvcHVwKHBvcHVwXzE5ZWYxZGI2ZDhjNTRiZmViMGYwODJiZGIzNGZiZWUxKQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xNjIxOTBhYjE4YTM0YzRjYmI0ODJiZTEwY2VhMzlkYSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEzLjA0OTcxMywgODAuMjEyNTU1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMzZDg0YWQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjM2Q4NGFkIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYThkOTgxN2NiZjllNGNlNmI4YTUzNzFhY2I3N2NiNDggPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNGFhZDkwYjg3NzJhNGI5Yzk0NGJlMGUyNDBmYTMyODIgPSAkKGA8ZGl2IGlkPSJodG1sXzRhYWQ5MGI4NzcyYTRiOWM5NDRiZTBlMjQwZmEzMjgyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5UaGlydXZvdHJpeXVyIENsdXN0ZXIgMS4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hOGQ5ODE3Y2JmOWU0Y2U2YjhhNTM3MWFjYjc3Y2I0OC5zZXRDb250ZW50KGh0bWxfNGFhZDkwYjg3NzJhNGI5Yzk0NGJlMGUyNDBmYTMyODIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMTYyMTkwYWIxOGEzNGM0Y2JiNDgyYmUxMGNlYTM5ZGEuYmluZFBvcHVwKHBvcHVwX2E4ZDk4MTdjYmY5ZTRjZTZiOGE1MzcxYWNiNzdjYjQ4KQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lYThhOGQ3N2U4YTQ0NjVjYTExOGI5ZmZlY2M3YWEyMiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEzLjA0Mzk0LCA4MC4xNzI1MV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjM2Q4NGFkIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzNkODRhZCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICAuYWRkVG8obWFwXzAyMzdjOTQxNzdhMzQ4ZmY5MzQzMWQxYTg4NmQwMWUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2ZjMjFmN2FjZmEzNTQ1YTc5OWQ5YjhmZDQwZjM1ODExID0gTC5wb3B1cCh7bWF4V2lkdGg6ICcxMDAlJwogICAgICAgICAgICAKICAgICAgICAgICAgfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzUyMTY4NThkYmRkZDRjNTE4NWM4ZjIyZjg5YWYyMWU1ID0gJChgPGRpdiBpZD0iaHRtbF81MjE2ODU4ZGJkZGQ0YzUxODVjOGYyMmY4OWFmMjFlNSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VGlydXN1bGFtIENsdXN0ZXIgMS4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mYzIxZjdhY2ZhMzU0NWE3OTlkOWI4ZmQ0MGYzNTgxMS5zZXRDb250ZW50KGh0bWxfNTIxNjg1OGRiZGRkNGM1MTg1YzhmMjJmODlhZjIxZTUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZWE4YThkNzdlOGE0NDY1Y2ExMThiOWZmZWNjN2FhMjIuYmluZFBvcHVwKHBvcHVwX2ZjMjFmN2FjZmEzNTQ1YTc5OWQ5YjhmZDQwZjM1ODExKQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jMTFlZDZlMzI5Njk0NjU3OTAxNjAxMzk1YjkyNjk2NiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEzLjEwNTYsIDgwLjI3OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmZmYwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZmZmMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xNDRlMTI5MTllYmM0ZThmOWM1MTg3YmE5YjhjMWQ0OSA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kOWU4ODA3MTYwZGY0ZjJjODVhOGIxNDAxYmI0ZmFhOCA9ICQoYDxkaXYgaWQ9Imh0bWxfZDllODgwNzE2MGRmNGYyYzg1YThiMTQwMWJiNGZhYTgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRpcnV2YWxsaWtlbmkgQ2x1c3RlciAzLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzE0NGUxMjkxOWViYzRlOGY5YzUxODdiYTliOGMxZDQ5LnNldENvbnRlbnQoaHRtbF9kOWU4ODA3MTYwZGY0ZjJjODVhOGIxNDAxYmI0ZmFhOCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jMTFlZDZlMzI5Njk0NjU3OTAxNjAxMzk1YjkyNjk2Ni5iaW5kUG9wdXAocG9wdXBfMTQ0ZTEyOTE5ZWJjNGU4ZjljNTE4N2JhOWI4YzFkNDkpCiAgICAgICAgICAgIDsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2M1ZGFkMzM1YjEyODQ0YjJhZjA2MzU0YTI4ZTljMGRkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMTMuMDUzMzMsIDgwLjE2MTExXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMzZDg0YWQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjM2Q4NGFkIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIC5hZGRUbyhtYXBfMDIzN2M5NDE3N2EzNDhmZjkzNDMxZDFhODg2ZDAxZTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfN2ViMjMyNDczYTgzNDg4NjkzOTgxZTViN2U4OGJhMzUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzEwMCUnCiAgICAgICAgICAgIAogICAgICAgICAgICB9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfODQ5NTIxOWJiNWYxNGFlNGJkMTUyZmZkMmZmN2RlMGYgPSAkKGA8ZGl2IGlkPSJodG1sXzg0OTUyMTliYjVmMTRhZTRiZDE1MmZmZDJmZjdkZTBmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Ub25kaWFycGV0IENsdXN0ZXIgMS4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83ZWIyMzI0NzNhODM0ODg2OTM5ODFlNWI3ZTg4YmEzNS5zZXRDb250ZW50KGh0bWxfODQ5NTIxOWJiNWYxNGFlNGJkMTUyZmZkMmZmN2RlMGYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYzVkYWQzMzViMTI4NDRiMmFmMDYzNTRhMjhlOWMwZGQuYmluZFBvcHVwKHBvcHVwXzdlYjIzMjQ3M2E4MzQ4ODY5Mzk4MWU1YjdlODhiYTM1KQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zNzBkZmJjMThhYzg0NWRhYTVmOGZhNTZmODBkZWRiZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEyLjk3NTgsIDgwLjIyMDVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81OGJhN2M0ZjEzMzQ0YmZhYmE4NzMxZjdmMGFjMWE4YiA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iODE2YzcxNjI0ZGU0ZjkwYTBlNDc2MzAxNDQyMjY0NiA9ICQoYDxkaXYgaWQ9Imh0bWxfYjgxNmM3MTYyNGRlNGY5MGEwZTQ3NjMwMTQ0MjI2NDYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlVuaXRlZCBJbmRpYSBDb2xvbnkgQ2x1c3RlciAwLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzU4YmE3YzRmMTMzNDRiZmFiYTg3MzFmN2YwYWMxYThiLnNldENvbnRlbnQoaHRtbF9iODE2YzcxNjI0ZGU0ZjkwYTBlNDc2MzAxNDQyMjY0Nik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zNzBkZmJjMThhYzg0NWRhYTVmOGZhNTZmODBkZWRiZS5iaW5kUG9wdXAocG9wdXBfNThiYTdjNGYxMzM0NGJmYWJhODczMWY3ZjBhYzFhOGIpCiAgICAgICAgICAgIDsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQzNTFhNzBjM2Y0YTQ0MGVhMzRhY2I0YWEzMzcxOTRhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMTMuMTEwMSwgODAuMjA5NV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmZmZjAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmZmYwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICAuYWRkVG8obWFwXzAyMzdjOTQxNzdhMzQ4ZmY5MzQzMWQxYTg4NmQwMWUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2ViNWI2ZTY5NGEzNTRlNjRhYTg4ODYzNDI2MzQyZGE4ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICcxMDAlJwogICAgICAgICAgICAKICAgICAgICAgICAgfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2VlYTNhNTY1YmIxYzQ1ZjI4ZWNjMzNiZGI4OGQ1ZTQ2ID0gJChgPGRpdiBpZD0iaHRtbF9lZWEzYTU2NWJiMWM0NWYyOGVjYzMzYmRiODhkNWU0NiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VmFuZGFsdXIgQ2x1c3RlciAzLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2ViNWI2ZTY5NGEzNTRlNjRhYTg4ODYzNDI2MzQyZGE4LnNldENvbnRlbnQoaHRtbF9lZWEzYTU2NWJiMWM0NWYyOGVjYzMzYmRiODhkNWU0Nik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80MzUxYTcwYzNmNGE0NDBlYTM0YWNiNGFhMzM3MTk0YS5iaW5kUG9wdXAocG9wdXBfZWI1YjZlNjk0YTM1NGU2NGFhODg4NjM0MjYzNDJkYTgpCiAgICAgICAgICAgIDsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2MzZDdjOTcxYjU0ZDQwY2ZiZmNlZmRiNTBmMDkzYWZjID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMTMuMDQ5NTU3LCA4MC4xODQ5MjhdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzNkODRhZCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzZDg0YWQiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jNDUzMzYwZDQxM2U0OGMxOWVmNjc4MGQ2MTFiNGJhMSA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83MjcyMjRjM2FmZGE0ZTJlOTAwOTk1N2Y3OGRlZmNjZSA9ICQoYDxkaXYgaWQ9Imh0bWxfNzI3MjI0YzNhZmRhNGUyZTkwMDk5NTdmNzhkZWZjY2UiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlZhZGFwYWxhbmkgQ2x1c3RlciAxLjA8L2Rpdj5gKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2M0NTMzNjBkNDEzZTQ4YzE5ZWY2NzgwZDYxMWI0YmExLnNldENvbnRlbnQoaHRtbF83MjcyMjRjM2FmZGE0ZTJlOTAwOTk1N2Y3OGRlZmNjZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jM2Q3Yzk3MWI1NGQ0MGNmYmZjZWZkYjUwZjA5M2FmYy5iaW5kUG9wdXAocG9wdXBfYzQ1MzM2MGQ0MTNlNDhjMTllZjY3ODBkNjExYjRiYTEpCiAgICAgICAgICAgIDsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2Y1Mjg2MzM3NTY1YTRhNzk4NTkxY2YyNjNmOGJmNTIwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMTMuMTA5MDMsIDgwLjI1NzRdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmZmYwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZmZmMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yNmFjOTYxNDYyMjc0NmY5YmY1NzhmZjQ4M2JkZjg3MiA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hYzEyNTI3Mjg5ZDU0N2IzOTNjZGM0MTI5YjMzMzVhYiA9ICQoYDxkaXYgaWQ9Imh0bWxfYWMxMjUyNzI4OWQ1NDdiMzkzY2RjNDEyOWIzMzM1YWIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlZhbGFzYXJhdmFra2FtIENsdXN0ZXIgMy4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yNmFjOTYxNDYyMjc0NmY5YmY1NzhmZjQ4M2JkZjg3Mi5zZXRDb250ZW50KGh0bWxfYWMxMjUyNzI4OWQ1NDdiMzkzY2RjNDEyOWIzMzM1YWIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZjUyODYzMzc1NjVhNGE3OTg1OTFjZjI2M2Y4YmY1MjAuYmluZFBvcHVwKHBvcHVwXzI2YWM5NjE0NjIyNzQ2ZjliZjU3OGZmNDgzYmRmODcyKQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83YWMyNzEwMDc4MWE0NjQwYTg0NDFlMmEyNWMyNWI4ZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEzLjExMzEsIDgwLjI4ODZdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzNkODRhZCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzZDg0YWQiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgLmFkZFRvKG1hcF8wMjM3Yzk0MTc3YTM0OGZmOTM0MzFkMWE4ODZkMDFlMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xMTFmY2EwMjIwOWQ0YTkzYTgwNDhjMDZlNzkzMGJkYiA9IEwucG9wdXAoe21heFdpZHRoOiAnMTAwJScKICAgICAgICAgICAgCiAgICAgICAgICAgIH0pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80MTViM2NjNjA2ODk0NjJmOTRhMWQwYjViNDIzOWU1YSA9ICQoYDxkaXYgaWQ9Imh0bWxfNDE1YjNjYzYwNjg5NDYyZjk0YTFkMGI1YjQyMzllNWEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlZhbGxhbGFyIE5hZ2FyIENsdXN0ZXIgMS4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8xMTFmY2EwMjIwOWQ0YTkzYTgwNDhjMDZlNzkzMGJkYi5zZXRDb250ZW50KGh0bWxfNDE1YjNjYzYwNjg5NDYyZjk0YTFkMGI1YjQyMzllNWEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfN2FjMjcxMDA3ODFhNDY0MGE4NDQxZTJhMjVjMjViOGQuYmluZFBvcHVwKHBvcHVwXzExMWZjYTAyMjA5ZDRhOTNhODA0OGMwNmU3OTMwYmRiKQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84YTFmNWFhZGZjNmY0OWZlOWFjYTUzZGMzZmYwODc1OSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzEzLjA0MTQsIDgwLjIzM10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICAuYWRkVG8obWFwXzAyMzdjOTQxNzdhMzQ4ZmY5MzQzMWQxYTg4NmQwMWUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2ZlOTFmODBiN2MyZDQ3MjU5NzZiMDA2ZWExZjE2MGNkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICcxMDAlJwogICAgICAgICAgICAKICAgICAgICAgICAgfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2RkMjRiNmJlYmRhMTQzNWFiOGVhZTY0ZjMyNThmZGMxID0gJChgPGRpdiBpZD0iaHRtbF9kZDI0YjZiZWJkYTE0MzVhYjhlYWU2NGYzMjU4ZmRjMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VmFuYWdhcmFtIENsdXN0ZXIgMC4wPC9kaXY+YClbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mZTkxZjgwYjdjMmQ0NzI1OTc2YjAwNmVhMWYxNjBjZC5zZXRDb250ZW50KGh0bWxfZGQyNGI2YmViZGExNDM1YWI4ZWFlNjRmMzI1OGZkYzEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOGExZjVhYWRmYzZmNDlmZTlhY2E1M2RjM2ZmMDg3NTkuYmluZFBvcHVwKHBvcHVwX2ZlOTFmODBiN2MyZDQ3MjU5NzZiMDA2ZWExZjE2MGNkKQogICAgICAgICAgICA7CgogICAgICAgICAgICAKICAgICAgICAKPC9zY3JpcHQ+" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>




```python
chennai_merged.loc[chennai_merged['Cluster Labels'] == 0, chennai_merged.columns[[0] + list(range(4, chennai_merged.shape[1]))]]['1st Most Common Venue'].value_counts().head()
```




    Indian Restaurant    26
    Bus Station           2
    Clothing Store        2
    Café                  1
    Boutique              1
    Name: 1st Most Common Venue, dtype: int64




```python
chennai_merged.loc[chennai_merged['Cluster Labels'] == 1, chennai_merged.columns[[0] + list(range(4, chennai_merged.shape[1]))]]['1st Most Common Venue'].value_counts().head()
```




    Ice Cream Shop                   9
    Fast Food Restaurant             5
    Vegetarian / Vegan Restaurant    5
    Pizza Place                      5
    Coffee Shop                      4
    Name: 1st Most Common Venue, dtype: int64




```python
chennai_merged.loc[chennai_merged['Cluster Labels'] == 2, chennai_merged.columns[[0] + list(range(4, chennai_merged.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>46</th>
      <td>Kolathur</td>
      <td>Pharmacy</td>
      <td>Flower Shop</td>
      <td>Fishing Store</td>
      <td>Fish Market</td>
      <td>Fast Food Restaurant</td>
      <td>Farmers Market</td>
      <td>Farm</td>
      <td>Electronics Store</td>
      <td>Donut Shop</td>
      <td>Diner</td>
    </tr>
    <tr>
      <th>60</th>
      <td>Manali</td>
      <td>Pharmacy</td>
      <td>Market</td>
      <td>Motorcycle Shop</td>
      <td>Electronics Store</td>
      <td>Fishing Store</td>
      <td>Fish Market</td>
      <td>Fast Food Restaurant</td>
      <td>Farmers Market</td>
      <td>Farm</td>
      <td>Donut Shop</td>
    </tr>
    <tr>
      <th>64</th>
      <td>Mangadu</td>
      <td>Pharmacy</td>
      <td>Spa</td>
      <td>Donut Shop</td>
      <td>Fishing Store</td>
      <td>Fish Market</td>
      <td>Fast Food Restaurant</td>
      <td>Farmers Market</td>
      <td>Farm</td>
      <td>Electronics Store</td>
      <td>Diner</td>
    </tr>
    <tr>
      <th>88</th>
      <td>Otteri</td>
      <td>Pharmacy</td>
      <td>Flower Shop</td>
      <td>Fishing Store</td>
      <td>Fish Market</td>
      <td>Fast Food Restaurant</td>
      <td>Farmers Market</td>
      <td>Farm</td>
      <td>Electronics Store</td>
      <td>Donut Shop</td>
      <td>Diner</td>
    </tr>
    <tr>
      <th>93</th>
      <td>Pallikaranai</td>
      <td>Pharmacy</td>
      <td>Light Rail Station</td>
      <td>Donut Shop</td>
      <td>Fishing Store</td>
      <td>Fish Market</td>
      <td>Fast Food Restaurant</td>
      <td>Farmers Market</td>
      <td>Farm</td>
      <td>Electronics Store</td>
      <td>Diner</td>
    </tr>
    <tr>
      <th>107</th>
      <td>Poonamallee</td>
      <td>Pharmacy</td>
      <td>Flower Shop</td>
      <td>Fishing Store</td>
      <td>Fish Market</td>
      <td>Fast Food Restaurant</td>
      <td>Farmers Market</td>
      <td>Farm</td>
      <td>Electronics Store</td>
      <td>Donut Shop</td>
      <td>Diner</td>
    </tr>
  </tbody>
</table>
</div>




```python
chennai_merged.loc[chennai_merged['Cluster Labels'] == 3, chennai_merged.columns[[0] + list(range(4, chennai_merged.shape[1]))]]['1st Most Common Venue'].value_counts().head()
```




    Train Station    8
    Bus Station      5
    Church           1
    Beach            1
    Metro Station    1
    Name: 1st Most Common Venue, dtype: int64



# Results

The following bar graph shows that Cluster 1 has the most number of restaurants. While a bar graph does not make enough sense here (because the clusters are clustered based on similarity factors other than distance between each other), it is useful to demonstrate the success of clustering. 


```python
clus1 = pd.DataFrame(pd.DataFrame(list(chennai_merged[chennai_merged['Cluster Labels'] == 0].iloc[:, 4:15].values.ravel()), columns = ['venue_count'])['venue_count'].value_counts()[:6])

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
fig, ax = plt.subplots(figsize=(12, 9))
ax = sns.barplot(x = clus1.index, y = clus1['venue_count'], palette=(flatui))
ax.set_xticklabels(ax.get_xticklabels(), rotation = 30,  fontsize = 15)
ax.yaxis.label.set_size(15)
plt.title('Most frequent venues in cluster 0', fontsize = 15)
plt.show()
```


![png](output_35_0.png)


The bar graphs for the rest of the clusters show that they do not have any significant similarities between each other.


```python
clus1 = pd.DataFrame(pd.DataFrame(list(chennai_merged[chennai_merged['Cluster Labels'] == 1].iloc[:, 4:15].values.ravel()), columns = ['venue_count'])['venue_count'].value_counts()[:6])
clus2 = pd.DataFrame(pd.DataFrame(list(chennai_merged[chennai_merged['Cluster Labels'] == 2].iloc[:, 4:15].values.ravel()), columns = ['venue_count'])['venue_count'].value_counts()[:6])
clus3 = pd.DataFrame(pd.DataFrame(list(chennai_merged[chennai_merged['Cluster Labels'] == 3].iloc[:, 4:15].values.ravel()), columns = ['venue_count'])['venue_count'].value_counts()[:6])

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
fig, ax = plt.subplots(2,2, figsize = (20, 18))


plt.subplot(2, 2, 1)
ax = sns.barplot(x = clus1.index, y = clus0['venue_count'], palette=(flatui))
ax.set_xticklabels(ax.get_xticklabels(), rotation = 30)
plt.title('Most frequent venues in cluster 1')

plt.subplot(2, 2, 2)
ax = sns.barplot(x = clus2.index, y = clus2['venue_count'], palette=(flatui))
ax.set_xticklabels(ax.get_xticklabels(), rotation = 30)
plt.title('Most frequent venues in cluster 2')

plt.subplot(2, 2, 3)
ax = sns.barplot(x = clus3.index, y = clus3['venue_count'], palette=(flatui))
ax.set_xticklabels(ax.get_xticklabels(), rotation = 30)
plt.title('Most frequent venues in cluster 3', )


plt.show()
```


![png](output_37_0.png)


# Discussion



```python
chennai_venues['Venue Category'].value_counts()
```




    Indian Restaurant                121
    Fast Food Restaurant              32
    Vegetarian / Vegan Restaurant     32
    Train Station                     31
    Bus Station                       29
                                    ... 
    Soccer Stadium                     1
    Arts & Crafts Store                1
    Antique Shop                       1
    Pub                                1
    Pier                               1
    Name: Venue Category, Length: 147, dtype: int64



The sheer number of restaurants probably skew the results. There might be two reaons for this: 
* Foursquare data predominantly contains restaurant data.
* There are way too many restaurants in Chennai.

Chennai is not in any means a developed city. It contains numerous shops and stalls which simply will not be indexed or recorded by any location API like Foursquare, or even Google Maps. And multiple venues open each day, the updation of this data is also an issue. Due to these reasons, accurate analysis cannot be done. 


# Conclusion

Mr. Nolan would not have trouble finding food in the city of Chennai. For lesser travel times, he can choose any of the 58 neighborhoods in Cluster 1, however he'll find that most of them are Indian restaurants. Independent of travel distance, the cluster choice does not matter much as there are more restaurants than any other venue. There is simply not enough data to do an in-depth analysis. However, individually marking the venues which are food-related is also a possibility- something to do for the future. 

The efficacy of the project was impacted by the limited effectives of the Foursquare API for a place like Chennai. Using some other city would've probably yielded better results. The way the neighborhoods were suggested were due to Indian restaurants alone. More diversity from more food stalls could've been used, i.e. Donut Shops, Bakeries, Pizzerias, etc. 

Further analysis could've been done by using the rating of each venue however yet again ratings were not available for venues in Chennai. If ratings were available, individual restaurants in each locality could've been suggested. 
