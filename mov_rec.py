#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd


# In[6]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[7]:


movies.head(1)


# In[8]:


credits.head(1)


# In[9]:


movies=movies.merge(credits,on='title')


# In[10]:


movies.head(1)


# In[11]:


#genres
#id
#keywords
#title
#overview
#cast
#crew
movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[12]:


movies.info()


# In[13]:


movies.head()


# In[14]:


movies.isnull().sum()


# In[15]:


movies.dropna(inplace=True)


# In[16]:


movies.duplicated().sum()


# In[17]:


movies.iloc[0].genres


# In[18]:


#'[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
#['Action','Adventure','Fantasy','SciFi']


# In[19]:


import ast


# In[20]:


def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[21]:


movies['genres']=movies['genres'].apply(convert)


# In[22]:


movies.head(1)


# In[23]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[24]:


movies.head()


# In[25]:


import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[26]:


def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L 


# In[27]:


movies['cast'] = movies['cast'].apply(lambda x:x[0:3])


# In[28]:


movies['cast'] = movies['cast'].apply(convert3)
movies.head()


# In[29]:


movies.head()


# In[30]:


def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 


# In[31]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[32]:


movies.sample(5)


# In[33]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[34]:


movies.head()


# In[35]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[36]:


movies.head()


# In[37]:


movies['tags']=movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[38]:


movies.head()


# In[39]:


new_df = movies[['movie_id','title','tags']]


# In[40]:


new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))


# In[41]:


new_df.head()


# In[42]:


new_df['tags'][0]


# In[43]:


new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())


# In[44]:


new_df.head()


# In[45]:


pip install nltk


# In[46]:


import nltk


# In[47]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[48]:


def stem(text):
    y = []
    
    for i in text.split():
        y.append(ps.stem(i))
        
    return  " ".join(y)
    


# In[49]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[50]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
cv  = CountVectorizer(max_features=5000,stop_words='english')


# In[51]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[52]:


vectors


# In[53]:


vectors[0]


# In[54]:


cv.get_feature_names_out()


# In[55]:


stem('In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. Action Adventure Fantasy ScienceFiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d SamWorthington ZoeSaldana SigourneyWeaver JamesCameron')


# In[56]:


from sklearn.metrics.pairwise import cosine_similarity


# In[57]:


similarity = cosine_similarity(vectors)


# In[58]:


sorted(list(enumerate(similarity[0])), reverse = True, key = lambda x:x[1])[1:6]


# In[59]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse = True, key = lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[60]:


recommend('Batman Begins')


# In[61]:


import pickle


# In[62]:


pickle.dump(new_df,open('Movies.pkl','wb'))


# In[63]:


new_df['title'].values


# In[64]:


pickle.dump(new_df.to_dict,open('movie_dict.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:



