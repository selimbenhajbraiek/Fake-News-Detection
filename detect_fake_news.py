import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from spacy import displacy
from spacy import tokenizer
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import gensim
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import LsiModel, TfidfModel
from sklearn. feature_extraction.text import TfidfVectorizer
from sklearn. feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report



#set plot options
plt.rcParams['figure.figsize'] = (12,8)
default_color = "#00bfbf"


data = pd.read_csv(r"\dataset\fake_news_data.csv")

data['fake_or_factual'].value_counts().plot(kind= 'bar', color = default_color)
plt.title('Real and Fake')

# POS Tagging

nlp = spacy.load("en_core_web_sm")
fake_news = data[data['fake_or_factual'] == "Fake News"]
fact_news = data[data['fake_or_factual'] == "Factual News"] 

fake_spacydocs = list(nlp.pipe(fake_news['text']))
fact_spacydoc = list(nlp.pipe(fact_news['text']))

# define function to extract token, ner tag and pos tags
def extract_token_tags(doc: spacy. tokens.doc.Doc):
    return [(i.text, i.ent_type_, i.pos_) for i in doc]

# define column names
columns = ["token", "ner_tag", "pos_tag"] 

fake_tagsdf = []


for ix, doc in enumerate(fake_spacydocs):
    tags = extract_token_tags(doc)
    tags = pd.DataFrame(tags)
    tags.columns = columns
    fake_tagsdf.append(tags)


fake_tagsdf = pd.concat(fake_tagsdf)

# define the same for factual news
fact_tagsdf = []

for ix, doc in enumerate(fact_spacydoc):
    tags = extract_token_tags(doc)
    tags = pd. DataFrame(tags)
    tags.columns = columns
    fact_tagsdf.append(tags)
    
fact_tagsdf = pd.concat(fact_tagsdf)

# group by token and pos tag and count occurrences
pos_count_fake = fake_tagsdf.groupby(["token","pos_tag"]).size().reset_index(name = "counts").sort_values(by= "counts", ascending = False)  

pos_count_fact = fact_tagsdf.groupby(["token","pos_tag"]).size().reset_index(name = "counts").sort_values(by= "counts", ascending = False) 

# top 10 pos tags in fake news
pos_count_fake.groupby('pos_tag' )['token'].count() .sort_values(ascending=False).head(10) 

pos_count_fact.groupby('pos_tag')['token'].count().sort_values(ascending = False).head(10)

pos_count_fake[pos_count_fake['pos_tag'] == 'NOUN'][:15]

pos_count_fact[pos_count_fact['pos_tag'] == 'NOUN'][:15]


# NER Tagging
top_enti_fake = fake_tagsdf[fake_tagsdf['ner_tag'] != ""].groupby(['token','ner_tag']).size().reset_index(name = 'counts').sort_values(by = 'counts', ascending = False)

top_enti_fact = fact_tagsdf[fact_tagsdf['ner_tag'] != ""].groupby(['token','ner_tag']).size().reset_index(name = 'counts').sort_values(by = 'counts', ascending = False)

# define color palette for NER tags for visualization
ner_palette = {
    'ORG': sns.color_palette("Set2").as_hex() [0],
    'GPE': sns.color_palette("Set2").as_hex() [1],
    'NORP': sns.color_palette("Set2").as_hex( ) [2],
    'PERSON': sns. color_palette("Set2") .as_hex() [3],
    'DATE': sns.color_palette("Set2").as_hex( ) [4],
    'CARDINAL': sns. color_palette("Set2") .as_hex() [5],
    'PERCENT': sns.color_palette("Set2").as_hex( ) [6]
}

# plot top 10 named entities in fake news

sns.barplot(
    x = 'counts',
    y = 'token',
    hue = 'ner_tag',
    palette = ner_palette,
    data = top_enti_fact[:10],
    orient = 'h',
    dodge = False,
).set(title = 'Most common Named Entity in Fake News')