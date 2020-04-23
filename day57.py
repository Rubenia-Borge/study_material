# -*- coding: utf-8 -*-
"""Day57.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zut2vBIRQjrECS2gX_jg7c1LlxpXD7dX
"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# Importing the dataset
data = pd.read_csv('covid_cases.csv')
print(data.shape)
data.head()

# Getting the values and plotting it

f1 = data['Country'].values
f2 = data['Cases'].values
X = np.array(list(zip(f1, f2)))
plt.scatter(f1, f2, c='black', s=7)

# Getting the values and plotting it

f1 = data['Country'].values
f2 = data['Cases'].values
X = np.array(list(zip(f1, f2)))
plt.scatter(f1, f2, c='black', s=7)

# Importing the dataset
data = pd.read_csv('ordered_cases_legend.csv')
print(data.shape)
data.head()

# Getting the values and plotting it

f1 = data['Order'].values
f2 = data['Cases'].values
X = np.array(list(zip(f1, f2)))
plt.scatter(f1, f2, c='black', s=7)

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

#Data to plot
labels = ['Europe','North America','Asia','Africa','Oceania']
sizes = [1113096,852247,396876,86521,8169]
colors = ['orange', 'yellow', 'green','red','blue']
explode = [0.1,0.1,0.1,0.1,0.5] #explode all slices

#plotting the pie chart
plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%',shadow=False,startangle=1000)
plt.title('Coronavirus: Total of Confirmed Cases\n',fontweight='bold')
plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()





import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

x = [1026,584,2718,717,24,23,2941,1339,6547,14795,1436,60,1907,2948,75,6264,39983,18,54,5,
     564,1309,20,40743,138,929,581,119,5,67,122,1163,37657,12,33,10507,83817,3977,160,332,
     662,847,1881,1087,772,6900,7711,712,846,16,4964,10128,3333,218,79,39,1535,24,111,18,
     3868,120,10,402,147065,1042,2245,14,289,622,50,65,57,9,477,1984,1773,18539,6760,83505,
     15652,13713,181228,223,10797,425,1852,281,10674,510,1995,568,19,739,677,99,51,81,1326,
     3558,121,17,5425,69,246,431,7,328,8261,2548,94,33,312,3046,39,9,16,31,33588,1440,10,
     648,665,1225,7156,1410,8418,4467,7,208,16325,6459,9593,20863,6015,8936,47121,147,15,15,
     12,462,4,10484,377,6630,11,43,8014,1173,1335,237,3300,4,200210,304,107,10,14777,27944,
     39,422,254,2792,22,84,114,884,90980,56,5710,7265,125856,535,784326,1627,256,268,449,6,
     1,65,25]
num_bins =10
n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.75, edgecolor='black')

plt.title('Confirmed Cases of COVID-19')
plt.xlabel('Confirmed Cases',fontsize=14)
plt.ylabel('Countries Registered',fontsize=14)

#save into a vector graphics format
plt.savefig('data_structures_histogram.pdf')

#save into a raster graphics format
plt.savefig('data_structures_histogram.png')
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime

try:
    # Try to fetch a list of Matplotlib releases and their dates
    # from https://api.github.com/repos/matplotlib/matplotlib/releases
    import urllib.request
    import json

    url = 'https://api.github.com/repos/matplotlib/matplotlib/releases'
    url += '?per_page=100'
    data = json.loads(urllib.request.urlopen(url, timeout=.4).read().decode())

    dates = []
    names = []
    for item in data:
        if 'rc' not in item['tag_name'] and 'b' not in item['tag_name']:
            dates.append(item['published_at'].split("T")[0])
            names.append(item['tag_name'])
    # Convert date strings (e.g. 2014-10-18) to datetime
    dates = [datetime.strptime(d, "%Y-%m-%d") for d in dates]

except Exception:
    # In case the above fails, e.g. because of missing internet connection
    # use the following lists as fallback.
    names = ['v2.2.4', 'v3.0.3', 'v3.0.2', 'v3.0.1', 'v3.0.0', 'v2.2.3',
             'v2.2.2', 'v2.2.1', 'v2.2.0', 'v2.1.2', 'v2.1.1', 'v2.1.0',
             'v2.0.2', 'v2.0.1', 'v2.0.0', 'v1.5.3', 'v1.5.2', 'v1.5.1',
             'v1.5.0', 'v1.4.3', 'v1.4.2', 'v1.4.1', 'v1.4.0']

    dates = ['2019-02-26', '2019-02-26', '2018-11-10', '2018-11-10',
             '2018-09-18', '2018-08-10', '2018-03-17', '2018-03-16',
             '2018-03-06', '2018-01-18', '2017-12-10', '2017-10-07',
             '2017-05-10', '2017-05-02', '2017-01-17', '2016-09-09',
             '2016-07-03', '2016-01-10', '2015-10-29', '2015-02-16',
             '2014-10-26', '2014-10-18', '2014-08-26']

    # Convert date strings (e.g. 2014-10-18) to datetime
    dates = [datetime.strptime(d, "%Y-%m-%d") for d in dates]

import numpy as np
from sklearn.linear_model import LinearRegression

# Commented out IPython magic to ensure Python compatibility.
# Load the Pandas libraries with alias 'pd' 
import pandas as pd 
# Read data from file 'filename.csv' 
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later) 
data = pd.read_csv("Confirmed_US.csv") 
# Preview the first 5 lines of the loaded data 
data.head()

test1 = data['Day']
test2 = data['Count']


# %matplotlib inline
import matplotlib.pyplot as plt
# time = [0, 1, 2, 3]
# position = [0, 100, 200, 300]

time = test1
position = test2

plt.plot(time, position)
plt.xlabel('Days')
plt.ylabel('Confirmed Cases')



# libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
# Data
df=pd.DataFrame({'x': range(1,11), 'y1': np.random.randn(10), 'y2': np.random.randn(10)+range(1,11), 'y3': np.random.randn(10)+range(11,21) })
 
# multiple line plot
plt.plot( 'x', 'y1', data=df, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
plt.plot( 'x', 'y2', data=df, marker='', color='olive', linewidth=2)
plt.plot( 'x', 'y3', data=df, marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")
plt.legend()

# libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
# Data
# df=pd.DataFrame({'x': range(1,11), 'y1': np.random.randn(10), 'y2': np.random.randn(10)+range(1,11), 'y3': np.random.randn(10)+range(11,21) })
 
# multiple line plot
plt.plot( 'x', 'y1', data=df, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
plt.plot( 'x', 'y2', data=df, marker='', color='olive', linewidth=2)
plt.plot( 'x', 'y3', data=df, marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")
plt.legend()



# Commented out IPython magic to ensure Python compatibility.
import numpy as np
from sklearn.linear_model import LinearRegression
# Load the Pandas libraries with alias 'pd' 
import pandas as pd 
# Read data from file 'filename.csv' 
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later) 
data = pd.read_csv("Confirmed_US.csv") 
# Preview the first 5 lines of the loaded data 
data.head()

# x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
# y = np.array([5, 20, 14, 32, 22, 38])


days = data['Day']
count = data['Count']

x = np.array(data['Day']).reshape((-1,1))
y = np.array(data['Count'])


# %matplotlib inline
import matplotlib.pyplot as plt

model = LinearRegression()

model.fit(x,y)

r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)

print('intercept:', model.intercept_)

print('slope:', model.coef_)

new_model = LinearRegression().fit(x, y.reshape((-1, 1)))

print('slope:', new_model.coef_)

y_pred = model.predict(x)

print('predicted response:', y_pred, sep='\n')

x_new = np.arange(5).reshape((-1, 1))

print(x_new)

import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.rand(10, 5), columns=['A', 'B', 'C', 'D', 'E'])
df.plot.box(grid='True')

import pandas as pd
import numpy as np

data = pd.read_csv("Confirmed_US.csv") 

df = pd.DataFrame(data,columns=['Count'])

# df.plot.box(grid='True')

df.plot.box(grid='False')
