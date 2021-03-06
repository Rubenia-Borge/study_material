#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
 
# implement liner regression with numpy and sklearn
 
# 1. Generate some random data
np.random.seed(seed=0)
size=1000
x_data = np.random.random(size)
y_data=6*x_data-32*x_data**4-x_data**2-np.sqrt(x_data) # -32x^4-x^2+6x-sqrt(x)
 
# 2. Split equally between test and train
x_train=x_data[:int(size/2)]
x_test = x_data[int(size/2):]
 
y_train=y_data[:int(size/2)]
y_test = y_data[int(size/2):]


# In[6]:


plot_config=['bs', 'b*', 'g^']
plt.plot(x_test, y_test, 'ro', label="Actual")
# 3. Set the polynomial degree to be fitted betwee 1 and 3
top_degree=3
d_degree = np.arange(1,top_degree+1)
for degree in d_degree:
     
    poly_fit = np.poly1d(np.polyfit(x_train,y_train, degree))
    # print poly_fit.coeffs
     
    # 4. Fit the numpy polynomial
    predict=poly_fit(x_test)
    error=np.mean((predict-y_test)**2)
     
    print ("Error with poly1d at degree %s is %s" %(degree, error))
     
    # 5. Create a fit a polynomial with sk-learn LinearRegression
    model = Pipeline([('poly', PolynomialFeatures(degree=degree)),('linear', linear_model.LinearRegression())])
    model=model.fit(x_train[:,np.newaxis], y_train[:,np.newaxis])
     
    predict_sk=model.predict(x_test[:,np.newaxis])
    error_sk=np.mean((predict_sk.flatten()-y_test)**2)
     
    print ("Error with sk-learn.LinearRegression at degree %s is %s" %(degree, error_sk))
     
    #plt.plot(x_test, y_test, '-', label="Actual")
    plt.plot(x_test, predict, plot_config[degree-1], label="Fit "+str(degree)+ " degree poly")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
 
plt.show()


# In[ ]:




