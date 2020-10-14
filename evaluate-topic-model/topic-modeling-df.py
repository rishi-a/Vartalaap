#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle


# In[14]:


#nltk.download('wordnet')


# In[22]:


#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')


# In[25]:

'''
#old data
df1 = pd.read_csv('../all-delhi-tweets-2016.csv')
df2 = pd.read_csv('../all-delhi-tweets-2017.csv')
df3 = pd.read_csv('../all-delhi-tweets-2018.csv')
df4 = pd.read_csv('../all-delhi-tweets-2019.csv')
df1 = df1.append([df2,df3,df4])
#df1.shape

df1 = pd.read_csv('tweetdata/delhi-2016.csv')
df2 = pd.read_csv('tweetdata/delhi-2017.csv')
df3 = pd.read_csv('tweetdata/delhi-2018.csv')
df4 = pd.read_csv('tweetdata/delhi-2019.csv')
df5 = pd.read_csv('tweetdata/delhi-2020.csv')
df1 = df1.append([df2,df3,df4,df5])
'''
df1 = pd.read_csv('tweetdata/final-delhi-2019-2020March.csv')

# In[26]:


textData = df1['tweet']
#textData = textData[0:100]

# In[27]:


# Apply a first round of text cleaning techniques
import re
import string
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

def clean_text_round1(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = str(text)
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text)
    text = re.sub('â€™', '', text)
    text = re.sub('œ', '', text)
    re.sub(r'[^\x00-\x7F]+','', text)
    return text

def clean_text_round2(text):
    '''Get rid of some additional punctuation and non-sensical text that was missed the first time around.'''
    text = str(text)
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    return text

#not doing stemming.
def lemmatize_stemming(text):
    text = str(text)
    #stemmer = SnowballStemmer("english")
    return WordNetLemmatizer().lemmatize(text, pos='v')

def remove_nonascii(text):
    text = str(text)
    return ''.join([i if ord(i) < 128 else ' ' for i in text])

def preprocess(text):
    text = str(text)
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return " ".join(str(x) for x in result)

def remove_mention_hashtag(text):
    text = str(text)
    text = re.sub('# [a-zA-Z]*', '', text)
    text = re.sub('# [a-zA-Z]*', '', text)
    return text
    
def some_words(text):
    text = str(text)
    text = re.sub('air pollution', '', text)
    text = re.sub('pollution', '', text)
    return text




# In[28]:


#textData = [clean_text_round1(item) for item in textData]
#textData = [clean_text_round2(item) for item in textData]
#textData = [preprocess(item) for item in textData]
#remove anything that is non-ascii
#textData = [remove_nonascii(item) for item in textData]
#textData = [remove_mention_hashtag(item) for item in textData]
textData = [some_words(item) for item in textData]



textData = pd.DataFrame(textData)


# In[29]:


tokenLists = []
tweetList = []
for item in textData[0]:
    try:
        tokenLists.append(item.split(' '))
        tweetList.append(item)
    except:
        continue


# In[30]:


dictionary = gensim.corpora.Dictionary(tokenLists)
dictionary.filter_extremes(no_below=15, no_above=0.5)
bow_corpus = [dictionary.doc2bow(doc) for doc in tokenLists]
from gensim import corpora, models
#tfidf = models.TfidfModel(bow_corpus)
#corpus_tfidf = tfidf[bow_corpus]


# ## Topic Modeling Attempt 1

# In[ ]:


lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=50, id2word=dictionary, passes=200)
lda_model.save('ldamodel/vis-50-200-NEWDATA.gensim')

# In[ ]:


#for idx, topic in lda_model.print_topics(-1):
#    print('Topic: {} \nWords: {}'.format(idx, topic))

import pyLDAvis.gensim
p=pyLDAvis.gensim.prepare(lda_model,bow_corpus,dictionary,mds='mmds', sort_topics=False)
pyLDAvis.save_html(p,'htmlfile/vis-50-200-NEWDATA.html')



def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        #each row is a list of tuple. Reverse sort on the basis of the second value of the tuple
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=bow_corpus, texts=tweetList)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Show
#df_dominant_topic.head(10)
df_dominant_topic.to_csv('doctopic/vis-50-200-NEWDATA.csv')

