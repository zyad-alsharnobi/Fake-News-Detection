# make a streamlit app to show visualizations of the data and take user input to predict if a news article is fake or real

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import re
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
import nltk

nltk_packages = ['omw-1.4', 'stopwords', 'punkt', 'wordnet']

for package in nltk_packages:
    try:
        nltk.data.find(package)
    except LookupError:
        nltk.download(package)

import re
import string
from wordcloud import WordCloud
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter

st.set_page_config(layout='wide', initial_sidebar_state='expanded')

class Preprocessing(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.port_stem = PorterStemmer()

    def fit(self, X, y=None):
        return self
    
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)        
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.strip()
        text = word_tokenize(text)
        text = [word for word in text if word not in self.stop_words]
        text = [self.lemmatizer.lemmatize(word) for word in text]
        return ' '.join(text)
    
    def stemming(self, text):
        words = text.split()
        stemmed_words = [self.port_stem.stem(word) for word in words]
        return ' '.join(stemmed_words)

    def transform(self, X, y=None):
        return X.apply(lambda text: self.stemming(self.clean_text(text)))    

# load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# load the tfidf
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# load the pipeline
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)
        
# Load the data
data = pd.read_csv('prepared_data.csv')


# MAKE THE APP
# Add a radio botton to the sidebar for page selecti
def main():
    preprocessor = Preprocessing()
    
    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Go to', ['Visualizations', 'Processing Steps', 'Prediction' ])

    if page == "Prediction":
        # User input
        user_input = st.text_area('Enter a news article:')

        # Predict
        if st.button('Predict'):
            user_input = preprocessor.transform(pd.Series(user_input))
            user_input = tfidf.transform(user_input)
            prediction = model.predict(user_input)
            # Display the prediction
            if prediction == 0:
                st.write('The news article is reliable.')
            else:
                st.write('The news article is unreliable.')

    elif page == "Processing Steps":

        st.title("Processing Steps")

        example_text = st.text_area('Enter a news article:', """Chuck Todd: ’BuzzFeed Did Donald Trump a Political Favor’ - Breitbart,Jeff Poor,"Wednesday after   Donald Trump’s press conference at Trump Tower in New York City, NBC “Meet the Press” moderator Chuck Todd expressed his exasperation over the normalcy of what he called a “circus” surrounding Trump’s event.  “I was struck big picture wise which is of how normal a circus is now to us,” Todd said. “This was a circus. We’ve never seen a   a transition like we saw today where the press conference gets interrupted, you have a lawyer in here. The lawyer does half legal talk, half political spin. I’ve never seen that, using the lawyer to say he’s here to make America great again, and by the way I’m going to play constitutional lawyer. I don’t think this but clearly a constitutional lawyer told us we better not accept any of this money. So they made that exception. So I am struck at how normal crazy looked to us today. This was just a crazy scene, but this is the norm of Donald Trump. And in fact, this is where he’s most comfortable. And I will say this. just as a political show. if you’re Donald Trump, you want these press conferences because it made the press look disjointed, unorganized, all this stuff. And his people, you know, he just, it was a performance for his supporters and his people. ” Later in the segment, Todd decried what he saw as elements within the intelligence community being at odds with one another, then called a story put out by BuzzFeed a night earlier suggesting Trump had ties to Russia to be a “political favor. ” “Look, let’s be honest here,” Todd said. “Politically BuzzFeed did Donald Trump a political favor today by doing what they did by going ahead and making it all public because it allowed them to deny a specific without having to deal with the bigger picture. ” Follow Jeff Poor on Twitter @jeff_poor""")

        # Data Preprocessing
        preprocessed_text = preprocessor.transform(pd.Series(example_text))[0]
        # text before processing steps
        st.write("Text before processing steps:")
        st.text_area("", example_text, height=200)
        st.write("""
        1. **Data Preprocessing**: The input text is first cleaned by converting it to lower case, removing digits, special characters, punctuations and extra spaces. It is then tokenized into individual words. Stop words are removed and the remaining words are lemmatized.
        """)
        st.text_area("", preprocessed_text, height=200)

        # Vectorization
        vectorized_text = tfidf.transform(pd.Series(preprocessed_text))
        st.write("""
        2. **Vectorization**: The preprocessed text is then transformed into a numerical representation using TF-IDF vectorizer.
        """)
        st.write(vectorized_text)

        # Prediction
        prediction = model.predict(vectorized_text)
        st.write("""
        3. **Prediction**: The vectorized text is passed to the pre-trained model to predict if the news article is fake or real.
        """)
        st.text_area("", 'The news article is reliable.' if prediction == 0 else 'The news article is unreliable.', height=50)

    elif page == "Visualizations":
        st.title('Visualizations')

        text_series = data['processed_content']
        text = ' '.join(text_series)

        # Labels Count Chart
        st.write("1. **Labels Count Chart**")
        labels_count = data['label'].value_counts()
        labels_count_bar = go.Figure(data=[go.Bar(x=labels_count.index, y=labels_count.values)], layout=go.Layout(title='Labels Count', height=400, width=800))
        st.plotly_chart(labels_count_bar, use_container_width=True)

        # Top 20 Authors
        st.write("2. **Top 20 Authors**")
        top_authors = data['author'].value_counts().head(20)
        top_authors_bar = go.Figure(data=[go.Bar(x=top_authors.index, y=top_authors.values)])
        st.plotly_chart(top_authors_bar, use_container_width=True)

        # Word Frequency
        st.write("3. **Word Frequency**")

        txt = data['processed_content'].str.lower().str.cat(sep=' ')
        words = text.split()
        counter = Counter(words)
        most_common = counter.most_common(25)
        x, y = zip(*most_common)
        fig = go.Figure(data=[go.Bar(x=x, y=y)])
        st.plotly_chart(fig, use_container_width=True)


if __name__ == '__main__':
    main()