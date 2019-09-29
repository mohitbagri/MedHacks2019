# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 22:08:51 2019

@author: Shivani Rai
"""

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.metrics import precision_score

df = pd.read_csv('C:/Users/Shivani Rai/Desktop/UPenn Sem 1/MedHacks/DataSets/MedHacks.csv')

k = 10

epochs = 10
display_step = 10

learning_rate = 0.3

batch_size = 250

normalized_df=df.copy()
normalized_df=normalized_df.drop('Name',axis=1)
normalized_df=(normalized_df-normalized_df.min())/(normalized_df.max()-normalized_df.min())

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
normalized_df=normalized_df.fillna(0)
kmeansop = KMeans(n_clusters=4).fit(normalized_df)
kmeansop

kmeansop.cluster_centers_

kmeansop.labels_

from sklearn.metrics import silhouette_samples, silhouette_score
#range_n_clusters = [2, 3, 4, 5, 6]
range_n_clusters=[4]
for n_clusters in range_n_clusters:
    
    
    
    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(normalized_df)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(normalized_df, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(normalized_df, cluster_labels)
    
n = 4

temp=pd.DataFrame(clusterer.labels_,columns=["Cluster Number"])

final=df.join(temp)
#print(final)

#print(temp.head())

#lists = [[] for _ in range(n)]
list0=[]
list1=[]
list2=[]
list3=[]


list0=final[final['Cluster Number']==0]
list0=list0.values.tolist()

list1=final[final['Cluster Number']==1]
list1=list1.values.tolist()

list2=final[final['Cluster Number']==2]
list2=list2.values.tolist()


list3=final[final['Cluster Number']==3]
list3=list3.values.tolist()


  
#print("Enter the Name, PIN, Age, Blind, Deaf,Dumb,Cancer,MI,AIDS,Gender")
#user_details=["Z999",19009,51,0,1,1,0,1,0,1]
import json  
import pandas as pd  
from pandas.io.json import json_normalize 


#fdata=pd.read_json("C:/Users/Shivani Rai/Desktop/UPenn Sem 1/MedHacks/DataSets/pennpals-257fa-export.json")

with open('C:/Users/Shivani Rai/Desktop/UPenn Sem 1/MedHacks/DataSets/pennpals-257fa-export.json') as f: 
    d = json.load(f) 

#print(d)
user_details=[None] * 10    
    
for key,value in d.items():
    if key=="AIDS":
        user_details[8]= 1 if value==True else 0
    if key=="Age":
        user_details[2]=int(value)
    if key=="Blind":
        user_details[3]= 1 if value==True else 0
    if key=="Cancer":
        user_details[6]= 1 if value==True else 0
    if key=="Deaf":
        user_details[4]= 1 if value==True else 0
    if key=="Dumb":
        user_details[5]= 1 if value==True else 0
    if key=="MI":
        user_details[7]= 1 if value==True else 0    
    if key=="Name":
        user_details[0]=value
    if key=="PIN":
        user_details[1]=value
    if key=="Gender":
        user_details[9]= 1 if value=="Female" else 0
print("this is",user_details)
ghi=user_details.copy()[1:]
ghi[0]=(ghi[0]-19001)/98
ghi[1]=(ghi[1]-18)/57
#print(ghi)

#print(clusterer.predict([abc]))
#print(kmeansop.predict([ghi]))

chosen_cluster=clusterer.predict([ghi])
knn_list=[]
if chosen_cluster==0:
  knn_list=list0.copy()
elif chosen_cluster==1:
  knn_list=list1.copy()
elif chosen_cluster==2:
  knn_list=list2.copy()
else:
  knn_list=list3.copy()
  
#print(len(list0))  
#print(len(list1)) 
#print(len(list2)) 
#print(len(list3)) 
a = np.array(ghi)
#print(a)
result={}
for i in knn_list:
  #print(i)
  name=i[0]
  new_list=i.copy()[1:len(i)-1]
  #print(new_list)
  #print(i)
  b=np.array(new_list)
  b[0]=(b[0]-19001)/98
  b[1]=(b[1]-18)/57
  #print(b)
  dist = np.linalg.norm(a-b)
  result[name]=dist
  
  


f= sorted(result.items(), key = lambda x : x[1])[:3]
final_result=[]
for i in f:
  final_result.append(i[0]+" ")
  
print("Recommended Names",final_result)  
import os 
file1 = open("C:/Users/Shivani Rai/Desktop/myfile.txt","w") 
  
  
# \n is placed to indicate EOL (End of Line) 
#file1.write("Hello \n") 
file1.writelines(final_result) 
file1.close()
