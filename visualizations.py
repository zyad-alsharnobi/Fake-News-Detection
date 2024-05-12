# visualizations.py

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import pandas as pd

def generate_wordcloud(text_series):
    # Combine all the text into one large string
    text = ' '.join(text_series)
    wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = None, 
                min_font_size = 10).generate(text)
    
    # plot the WordCloud image                        
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.show()

def plot_word_frequency(text_series, top_n=10):
    # Split the text into words
    words = ' '.join(text_series).split()
    # Count the frequency of each word
    word_freq = pd.Series(words).value_counts().head(top_n)
    # Create a bar plot of the word frequencies
    word_freq.sort_values().plot(kind='barh')
    plt.show()

def plot_label_distribution(labels):
    # Count the frequency of each label
    label_freq = labels.value_counts()
    # Create a bar plot of the label frequencies
    label_freq.plot(kind='bar')
    plt.show()