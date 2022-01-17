# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 21:26:29 2021

@author: Hp Folio 9480m
"""

#importing the libraries needed
import pandas as pd
import numpy as np
import matplotlib.pyplot as pp
import scipy
from scipy import stats
import seaborn as sb

from matplotlib import rcParams

import plotly.express as px
import plotly.graph_objects as go

np.set_printoptions(precision = 2)
rcParams['figure.figsize']= 8,6

#importing the dataset
#using your own dataset, we will continue using the one we used for the Kproto-type by just dropping the the age and income
student = 'StudentsPerformance (5).csv'
perform = pd.read_csv(student)
perform.columns = ['Gender', 'Race', 'parental', 'lunch', 'testpre','math', 'reading', 'writing','average']

perform['parental']= perform.parental.map({"associate's degree" :"ass degree", "bachelor's degree" : "bachelor",
                                          "high school" : "high sch", "master's degree" : "master", 
                                           "some college": "college", "some high school":"some high sch"})

perform.head()

#dropping the age and income 
#Here, i have to drop four continous variablies, but yours will be two continous variables (age and income)
newdf = perform.drop(['math', 'reading', 'writing', 'average'], axis=1) #newdf here

#import the Kmode library
from kmodes.kmodes import KModes

# Elbow curve to find optimal number of cluster
#this is faster than k-prototype i don't think we need to worry about the timing

import time
t1 = time.perf_counter()
cost = []
K = range(1,7)
for num_clusters in list(K):
    kmode = KModes(n_clusters=num_clusters, init = "Cao", n_init = 5, verbose=1)
    kmode.fit_predict(newdf) #change the perform1 to the newdf
    cost.append(kmode.cost_)
t2 = time.perf_counter()
print(f'Finished in {t2-t1} seconds')
pp.plot(K, cost, 'bx-')
pp.xlabel('No. of clusters')
pp.ylabel('Cost')
pp.title('Elbow Method For Optimal k')
pp.show()

#deppending on the number cluster, input the one you got
kmode = KModes(n_clusters=3, init = "Cao", n_init = 5, verbose=1)
clusters = kmode.fit_predict(newdf)#change the perform1 to newdf
clusters

newdf.insert(0, "Cluster", clusters, True)#insert cluster into newdf
newdf.head()

perform.columns# checking the columns names for visualizatiozation and crosstab

#creating newdf2
newdf2 = perform[['average',]]# creating another dataset from the initial one with continous variables
#In your case use the age and not the income

#joining newdf with newdf2
newdf3 =pd.concat([newdf,newdf2], axis=1) 
newdf3.head()
newdf3['Cluster_Modify'] = newdf3.Cluster.map({0:1, 1:2, 2:3})

#Visualization by changing just the hue with your categorical variable column names
#This will make us see some obvious reason behind the analysis 
sb.barplot(hue='Gender', y='average', x='Cluster_Modify', data=newdf3)

sb.barplot(hue='Race', y='average', x='Cluster_Modify', data=newdf3)

sb.barplot(hue='parental', y='average', x='Cluster_Modify', data=newdf3)

sb.barplot(hue='testpre', y='average', x='Cluster_Modify', data=newdf3)

sb.barplot(hue='lunch', y='average', x='Cluster_Modify', data=newdf3)

#Here comes the crosstab will the some time ago, inside the columns use your own categorical data names
crosstab = pd.crosstab(index=newdf3.Cluster_Modify, columns= [newdf3.Gender,
                                                                #newdf3.Race,
                                                                #newdf3.Race,
                                                                #newdf3.parental,
                                                                newdf3.lunch, 
                                                                #newdf3.testpre,
                                                               ],margins=True, margins_name='Total')
crosstab

crosstab3 = pd.crosstab([newdf3.Cluster_Modify, newdf3.Gender], [#newdf3.Gender,
                                                                #newdf3.Race,
                                                                #newdf3.Race,
                                                                newdf3.parental,
                                                                #newdf3.lunch, 
                                                                #newdf3.testpre,
                                                               ],margins=True, margins_name='Total')
crosstab3

#this is the parallezied for the number of cluster incase is taking long
import concurrent.futures
t1 = time.perf_counter()


#X = [1,2,3,4,5,6,7,8,9,10]

def non(k):    
    kmode = KModes(n_clusters=k, init= 'Cao')
    kmode.fit_predict(newdf)
    p = [kmode.cost_]
    print(p)
    
with concurrent.futures.ProcessPoolExecutor() as executor:
    L = [1,2,3,4,5,6,7,8]
    result = [executor.submit(non, l) for l in L]
    for f in concurrent.futures.as_completed(result):
        print(f.result)
    
t2 = time.perf_counter()

print(f'Finished in {t2-t1} seconds')