
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



The sheer number of restaurants probably skew the results. There might be two reasons for this: 
* Foursquare data predominantly contains restaurant data.
* There are way too many restaurants in Chennai.

Chennai is not in any means a developed city. It contains numerous shops and stalls which simply will not be indexed or recorded by any location API like Foursquare, or even Google Maps. And multiple venues open each day, the updation of this data is also an issue. Due to these reasons, accurate analysis cannot be done. 


# Conclusion

Mr. Nolan would not have trouble finding food in the city of Chennai. For lesser travel times, he can choose any of the 58 neighborhoods in Cluster 1, however he'll find that most of them are Indian restaurants. Independent of travel distance, the cluster choice does not matter much as there are more restaurants than any other venue. There is simply not enough data to do an in-depth analysis. However, individually marking the venues which are food-related is also a possibility- something to do for the future. 

The efficacy of the project was impacted by the limited effectives of the Foursquare API for a place like Chennai. Using some other city would've probably yielded better results. The way the neighborhoods were suggested were due to Indian restaurants alone. More diversity from more food stalls could've been used, i.e. Donut Shops, Bakeries, Pizzerias, etc. 

Further analysis could've been done by using the rating of each venue however yet again ratings were not available for venues in Chennai. If ratings were available, individual restaurants in each locality could've been suggested. 
