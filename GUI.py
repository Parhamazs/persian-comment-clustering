import os 
import numpy as np
import json
import random
from hazm import word_tokenize
from hazm.stemmer import Stemmer
import pickle
import streamlit as st

stemmer = Stemmer()

current_dir = os.path.dirname(os.path.realpath(__file__))

#load le
model = os.path.join(current_dir, 'Text_Mining', 'lion_le.jsdh')
lion_le = open(model, 'rb')
le = pickle.load(lion_le)
lion_le.close()

#load vectorizer
model = os.path.join(current_dir, 'Text_Mining', 'lion_v.jsdh')
lion_v = open(model, 'rb')
vectoeizer = pickle.load(lion_v)
lion_v.close()

#load classifier
model = os.path.join(current_dir, 'Text_Mining', 'lion_s.jsdh')
lion_s = open(model, 'rb')
svm = pickle.load(lion_s)
lion_s.close()

#load stopwords
model = os.path.join(current_dir, 'Text_Mining', 'stopwords.txt')
with open(model, encoding='utf8') as stopwords_file:
    stopwords = stopwords_file.readlines()
stopwords = [str(line).replace('\n', ' ') for line in stopwords]

# prediction function
def predict_labels(news):
    tokenized_title_body = word_tokenize(news)
    filtered_tokenized_title_body = [w for w in tokenized_title_body if not w in stopwords]
    stemmed_filtered_tokenized_title_body = [stemmer.stem(w) for w in filtered_tokenized_title_body] 
    x = [' '.join(stemmed_filtered_tokenized_title_body)]
    x_vectorize = vectoeizer.fit(x)
    p = svm.predict(x_vectorize)
    label = le.inverse_transform(p)
    
    
st.title('News Category Detector')
text = st.text_area('enter your news:')
btn = st.button('start detection!')

if btn:
    tokenized_title_body = word_tokenize(text)
    filtered_tokenized_title_body = [w for w in tokenized_title_body if not w in stopwords]
    stemmed_filtered_tokenized_title_body = [stemmer.stem(w) for w in filtered_tokenized_title_body] 
    x = [' '.join(stemmed_filtered_tokenized_title_body)]
    x_vectorize = vectoeizer.transform(x)
    p = svm.predict(x_vectorize)
    label = le.inverse_transform(p)
    st.success('category is: ' + str(label[0]).replace('\n',''))