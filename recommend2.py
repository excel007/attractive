#Import Library
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
import streamlit_analytics

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

#####################Similarity section
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

my_feature = pd.DataFrame(np.zeros((1,len(dfTourist))), columns=dfTourist)
# display(my_feature)
similarity = []
dfnew = dfTourist.transpose()

features = ['tourist','gender','age','education','job','homeland','with']
dfTourist['features'] = dfTourist['gender'].apply(str) + ' ' + dfTourist['age'].apply(str) + ' ' + dfTourist['education'].apply(str) + ' ' + dfTourist['job'].apply(str) + ' ' + dfTourist['homeland'].apply(str) + ' ' + dfTourist['with'].apply(str)
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
  with streamlit_analytics.track(unsafe_password=st.secrets["password_analytic"]):
    streamlit_analytics.start_tracking()

    st.sidebar.header("Your persona ...")
    gender = st.sidebar.radio("Gender",('ชาย','หญิง'))
    age = st.sidebar.slider("Age ",15,99)
    education = st.sidebar.selectbox("Education level",('ประถมศึกษา','มัธยมศึกษา','ปวช./ปวส.','ปริญญาตรี','ปริญญาโท','ปริญญาเอก'))
    job = st.sidebar.selectbox("Job",['นักเรียน/นักศึกษา','รับราชการ/รัฐวิสาหกิจ','เกษตรกรรม','พนักงาน/ลูกจ้างเอกชน','รับจ้างทั่วไป/บริการ','แม่บ้าน/พ่อบ้าน','ค้าขาย/ธุรกิจส่วนตัว','อื่นๆ'])
    prov = pd.read_csv('https://raw.githubusercontent.com/kongvut/thai-province-data/master/csv/thai_provinces.csv')
    homeland = st.sidebar.selectbox("ภูมิลำเนา",prov['name_th'])
    homecurrent = st.sidebar.selectbox("ที่อยู่ปัจจุบัน",prov['name_th'])
    tourwith = st.sidebar.selectbox('ชอบไปเที่ยวกับ',['เที่ยวคนเดียว','เที่ยวกับครอบครัว','เที่ยวกับเพื่อน/แฟน','อื่นๆ'])
    
    streamlit_analytics.stop_tracking()

st.divider()
tab4 = st.container()
with tab4:
  filter = (gender,str(age),education,job,homeland,tourwith)
  features = ' '.join(filter)

  dfTourist_find = pd.DataFrame({'tourist':[999],'gender':[gender],'age':[age],'education':[education],'job':[job],'homeland':[homeland],'with':[tourwith],'features':[features]})
  dfTourist_new = pd.concat([dfTourist_find,dfTourist.loc[:]]).reset_index(drop=True)
  
  st.write('Your persona ...')
  st.write(dfTourist_find)
  cv = CountVectorizer(tokenizer=lambda x: x.split(' ')) #แบ่งด้วยช่องว่าง
  count_matrix = cv.fit_transform(dfTourist_new['features'])
  cosine_sim = cosine_similarity(count_matrix)

  # st.write(len(dfTourist_new))
  # st.write(dfTourist_new['tourist'].loc(0))
  # st.write(dfTourist_new[dfTourist_new['tourist'] == 999])
  similar_tourist = list(enumerate(cosine_sim[0]))
  sorted_similar_tourist = sorted(similar_tourist,key=lambda x:x[1],reverse=True)[1:]
  st.write('like no# ',sorted_similar_tourist[0][0], ' score ',sorted_similar_tourist[0][1])
  tourist = sorted_similar_tourist[0][0]
  st.write(dfTourist[dfTourist['tourist']==tourist])
  st.write('may like these ...')
  # similar_tourist = list(enumerate(cosine_sim[tourist]))
  # sorted_similar_tourist = sorted(similar_tourist,key=lambda x:x[1],reverse=True)[1:]
  # tourist_sim = sorted_similar_tourist[0][0]
  st.write(dfPlace[dfPlace['tourist'] == tourist])
  # res = recomendation(tourist_sim,topk=10)
  # st.write(res)