#Import Library
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF

df = pd.read_csv('data/tourist_dataset.csv')

#Preprocessing
##Create a Tourist DF
dfTourist = df[['tourist','gender','age','education','job','homeland','with']]

##Create a Place DF
dfPlace = df[['tourist','mountian','sea','cultural','waterfall','museum']]
dfPlace = dfPlace.melt(['tourist'],var_name='place')
dfPlace = dfPlace[dfPlace['value'] != 0] #ตัด ศูนย์ ออก (รายการที่ tourist ไม่เลือก)

#Collaborative filtering
##Create dfTourist x dfPlace matrix
cmat = pd.crosstab(dfPlace['tourist'],dfPlace['place'],dfPlace['value'],aggfunc='mean')
cmat = df[['tourist','cultural','mountian','museum','sea','waterfall']]

##Decompose Matrix into two matrices using NMF
nmf = NMF(n_components=40)
nmf.fit(cmat)
H = pd.DataFrame(np.round(nmf.components_,2), columns=cmat.columns)
W = pd.DataFrame(np.round(nmf.transform(cmat),2), columns=H.index)

reconstructed = pd.DataFrame(np.round(np.dot(W,H),2), columns=cmat.columns)
reconstructed.index = cmat.index

#Matrix size evaluation
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
mean_squared_error(cmat,reconstructed)
np.sqrt(mean_squared_error(cmat,reconstructed))
mean_absolute_error(cmat,reconstructed)
r2_score(cmat,reconstructed)
model = LinearRegression()
model.fit(cmat,reconstructed)
model.score(cmat,reconstructed)

#Recommendation function
import re

def recomendation(uid,topk=1):
  res = reconstructed.T[uid].sort_values(ascending=False)[0:topk]
  res = list(res[res>0].index)
  res = dfPlace[dfPlace['place'].isin(res)]
  res = res.drop_duplicates(subset='place')
  res = res[:topk]
  res = res[['tourist','place','value']].sort_values(by='value',ascending=False)
  return res

def recomendation_cmat(uid,topk=5):
  res = cmat.T[uid].sort_values(ascending=False)[0:topk]
  res = list(res[res>0].index)
  res = dfPlace[dfPlace['place'].isin(res)]
  res = res.drop_duplicates(subset='place')
  res = res[:topk]
  res = res[['tourist','place','value']]
  return res

tourist = 60
res = recomendation_cmat(tourist,topk=10) #before Matrix Factorize
res = recomendation(tourist,topk=10) #after Matrix Factorize #ผลลัพธ์ไม่ต่างกันเพราะ มีตัวเลือกน้อย

# print(res)
#####################Similarity section
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

my_feature = pd.DataFrame(np.zeros((1,len(dfTourist))), columns=dfTourist)
# display(my_feature)
similarity = []
dfnew = dfTourist.transpose()

features = ['tourist','gender','age','education','job','homeland','with']
dfTourist['features'] = dfTourist['gender'].apply(str) + ' ' + dfTourist['education'].apply(str) + ' ' + dfTourist['job'].apply(str) + ' ' + dfTourist['homeland'].apply(str) + ' ' + dfTourist['with'].apply(str)
cv = CountVectorizer(tokenizer=lambda x: x.split(' ')) #แบ่งด้วยช่องว่าง
count_matrix = cv.fit_transform(dfTourist['features'])
# count_matrix.toarray()
##########################################

#import library
import streamlit as st
header = st.container()
nav = st.container()
body = st.container()

with header:
  st.title("Tourist Attractive place")
  st.write("Recommend")

with nav:
  st.sidebar.title("Navigator")
  tourist = st.sidebar.slider("Choose tourist no#",0,99)

tab1 = st.container()
with tab1:
  tab1.title("แหล่งท่องเที่ยวที่นักท่องเที่ยวน่าจะชอบ ")
  st.write("Tourist no#",tourist," love these ...")
  st.write(dfPlace[dfPlace['tourist'] == tourist])
  st.divider()
  col1,col2 = st.columns(2)
  col1.write("He/She seem tourist no# ...")
  res = recomendation_cmat(tourist,topk=10)
  col1.write(res)
  
  col2.write("After, Matrix Factorize ...")
  res = recomendation(tourist,topk=10)
  col2.write(res)

st.divider()
tab2 = st.container()
with tab2:
  tab2.title("หาคนที่เหมือน")
  cosine_sim = cosine_similarity(count_matrix)
  # tourist = 5
  similar_tourist = list(enumerate(cosine_sim[tourist]))
  # st.write(similar_tourist[:8])

  sorted_similar_tourist = sorted(similar_tourist,key=lambda x:x[1],reverse=True)[1:]
  # st.write(sorted_similar_tourist[:1])
  st.write("Charactoristic of Tourist no#",tourist," like these ...",sorted_similar_tourist[:1][0][1])
  col1,col2 = st.columns(2)
  col1.write(dfTourist.iloc[tourist])

  col2.write(dfTourist.iloc[sorted_similar_tourist[:1][0][0]])

st.divider()
tab3 = st.container()
with tab3:
  tab3.title("แนะนำแหล่งท่องเที่ยวที่น่าจะชอบ")
  st.write("Tourist no#",tourist, ' may like these ...')
  similar_tourist = list(enumerate(cosine_sim[tourist]))
  sorted_similar_tourist = sorted(similar_tourist,key=lambda x:x[1],reverse=True)[1:]
  tourist_sim = sorted_similar_tourist[0][0]
  res = recomendation(tourist_sim,topk=10)
  st.write(res)