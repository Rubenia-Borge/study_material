#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image

images = ['farm1.jpg', 'farm2.jpg', 'farm3.jpg', 'farm4.jpg',
 'farm5.jpg', 'farm6.jpg', 'farm7.jpg', 'farm8.jpg',
 'city1.jpg', 'city2.jpg', 'city3.jpg', 'city4.jpg',
 'city5.jpg', 'city6.jpg', 'city7.jpg', 'city8.jpg',
 'desert1.jpg', 'desert2.jpg', 'desert3.jpg', 'desert4.jpg',
 'desert5.jpg', 'desert6.jpg', 'desert7.jpg', 'desert8.jpg']

farm_green = []
farm_blue = []
city_green = []
city_blue = []
desert_green = []
desert_blue = []

for image in images[:8]:
    img = mpimg.imread(image)
    RGBtuple = np.array(img).mean(axis=(0,1))
    averageRed = RGBtuple[0]
    averageGreen = RGBtuple[1]
    averageBlue = RGBtuple[2]
    sumColor = averageRed+averageGreen+averageBlue
    percent_green = averageGreen/sumColor
    percent_blue = averageBlue/sumColor
    farm_green.append(percent_green)
    farm_blue.append(percent_blue)
for image in images[8:16]:
    img = mpimg.imread(image)
    RGBtuple = np.array(img).mean(axis=(0,1))
    averageRed = RGBtuple[0]
    averageGreen = RGBtuple[1]
    averageBlue = RGBtuple[2]
    sumColor = averageRed+averageGreen+averageBlue
    percent_green = averageGreen/sumColor
    percent_blue = averageBlue/sumColor
    city_green.append(percent_green)
    city_blue.append(percent_blue)
for image in images[16:]:
    img = mpimg.imread(image)
    RGBtuple = np.array(img).mean(axis=(0,1))
    averageRed = RGBtuple[0]
    averageGreen = RGBtuple[1]
    averageBlue = RGBtuple[2]
    sumColor = averageRed+averageGreen+averageBlue
    percent_green = averageGreen/sumColor
    percent_blue = averageBlue/sumColor
    desert_green.append(percent_green)
    desert_blue.append(percent_blue)

print(desert_blue)

plt.scatter(farm_green, farm_blue, s=100, facecolors ='red', edgecolors='black', alpha=0.5)
plt.scatter(city_green, city_blue, s=100, facecolors ='blue', edgecolors='black', alpha=0.5)
plt.scatter(desert_green, desert_blue, s=100, facecolors ='yellow', edgecolors='black', alpha=0.5)
plt.xlabel('Percentage of green')
plt.ylabel('Percentage of blue')
plt.title('Farm, city and desert training images')
plt.legend(['Farm','City','Desert'])
plt.show()


# In[ ]:


# 2. Now create an array of strings called training_target with the category of each.

training_target = ['farm', 'farm', 'farm', 'farm',
 'farm', 'farm', 'farm', 'farm',
 'city', 'city', 'city', 'city',
 'city', 'city', 'city', 'city',
 'desert', 'desert', 'desert', 'desert',
 'desert', 'desert', 'desert', 'desert']



# In[ ]:


# 3. Create an empty array of zeros called training_data that will eventually store the percent green and percent blue values.


# In[ ]:


training_data = np.zeros((24,2))


# In[ ]:


# 4. Now fill the training_data array with the proper values for each image, and observe the values 
# in array after it is finished.


# In[ ]:


for i in range(8):
    training_data[i,0] = farm_green[i]
    training_data[8+i,0] = city_green[i]
    training_data[16+i,0] = desert_green[i]
    training_data[i,1] = farm_blue[i]
    training_data[8+i,1] = city_blue[i]
    training_data[16+i,1] = desert_blue[i]
    
for i in training_data:
    print(i)


# In[ ]:


from sklearn import neighbors
k1 = neighbors.KNeighborsClassifier(1,weights='distance')


# In[ ]:


k1.fit(training_data,training_target)


# In[ ]:


test = ['test1.jpg','test2.jpg','test3.jpg']
test_green = []
test_blue = []
for image in test:
    img = mpimg.imread(image)
    RGBtuple = np.array(img).mean(axis=(0,1))
    averageRed = RGBtuple[0]
    averageGreen = RGBtuple[1]
    averageBlue = RGBtuple[2]
    sumColor = averageRed+averageGreen+averageBlue
    percent_green = averageGreen/sumColor
    percent_blue = averageBlue/sumColor
    test_green.append(percent_green)
    test_blue.append(percent_blue)
test_data = np.zeros((3,2))
for i in range(3):
    test_data[i,0] = test_green[i]
    test_data[i,1] = test_blue[i]
for i in test_data:
    print(i)   


# In[ ]:


k1_pred = k1.predict(test_data)


# In[ ]:


print(k1_pred)


# In[ ]:


from PIL import Image 
im = Image.open("test1.jpg")
#Plots the image from the array data 
imgplot = plt.imshow(im)
plt.show()

im = Image.open("test2.jpg")
#Plots the image from the array data 
imgplot = plt.imshow(im)
plt.show()

im = Image.open("test3.jpg")
#Plots the image from the array data 
imgplot = plt.imshow(im)
plt.show()

