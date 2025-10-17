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


# Text Preprocessing

data['text_clean' ] = data.apply(lambda x : re.sub(r'^[^-] *- \s' , "", x['text']), axis = 1)

#Lowercasing
data['text_clean' ] = data['text_clean'].str.lower()

# Removing Punctuation
data['text_clean' ] = data.apply(lambda x: re.sub(r'([^\w\s])','', x['text_clean']), axis =1)

en_stopwords = stopwords.words('english')

# Removing stop words
data['text_clean'] = data['text_clean'].apply(lambda x : ' '.join([word for word in x.split() if word not in (en_stopwords)]))

# Tokenizing
data['text_clean' ] = data.apply(lambda x : word_tokenize(x['text_clean']), axis = 1 )

#Lemmatization
lemmatizer = WordNetLemmatizer()
data['text_clean' ] = data['text_clean'].apply(lambda tokens: [lemmatizer. lemmatize(token) for token in tokens])

tokens_clean = sum(data['text_clean' ], [])

#n-grams - unigrams
unigrams = (pd. Series(nltk.ngrams(tokens_clean, 1)).value_counts()).reset_index() [:10]
default_plot_colour = "#1f77b4"   # blue color
sns.barplot(
    x = 'count',
    y = 'token',
    data = unigrams,
    orient = 'h',
    palette = [default_plot_colour],
    hue = 'token',
    legend = False)\
.set(title = 'Most common Unigrams after preprocessing')
# Sentiment Analysis using VADER

vader_sentiment = SentimentIntensityAnalyzer()
data['vader_sentiment'] = data['text'].apply(lambda x: vader_sentiment.polarity_scores(x)['compound'])

# Visualize sentiment distribution
bins = [-1.0, -0.1, 0.1, 1.0]
labels = ['negative', 'neutral', 'positive']
data['sentiment_score_label'] = pd.cut(data['vader_sentiment'], bins=bins, labels=labels)
data['sentiment_score_label'].value_counts().plot.bar(color = default_color)

sns.countplot(
    x = 'fake_or_factual',
    hue = 'sentiment_score_label',
    palette = sns.color_palette('hls'),
    data = data
).set(title = 'Fake or Fact News')

# Topic Modeling 

# LDA
fake_news_text = data[data['fake_or_factual'] == "Fake News"]['text_clean'].reset_index(drop=True)
fake_news_text = data[data['fake_or_factual'] == "Fake News"] ['text_clean'].reset_index(drop=True) 
dictionary_fake = corpora.Dictionary(fake_news_text) 
doc_term_fake = [dictionary_fake.doc2bow(text) for text in fake_news_text] 
coherence_values = [] 
model_list = [] 
min_topics = 2 
max_topics = 11 
for num_topics_i in range(min_topics, max_topics+1): 
    model = gensim.models.LdaModel(doc_term_fake, num_topics = num_topics_i, id2word = dictionary_fake) 
    model_list.append(model) 
    coherence_model = CoherenceModel(model=model, texts=fake_news_text, dictionary=dictionary_fake, coherence='c_v') 
    coherence_values. append(coherence_model.get_coherence()) 

# Plot coherence scores
plt.plot(range(min_topics, max_topics+1), coherence_values) 
plt.xlabel("Number of Topics") 
plt.label("Coherence Scores") 
plt.legend(("coherence_values"), loc='best') 
plt.show()

# Build LDA model with optimal number of topics
num_topic_lda = 7 
lda_model = gensim.models.LdaModel(doc_term_fake , num_topics = num_topic_lda ,id2word = dictionary_fake) 
lda_model.print_topics(num_topics = num_topic_lda, num_words = 10)


#Lsi Model

def tfidf_corpus(doc_term_matrix):
    tfidf = TfidfModel(corpus = doc_term_matrix, normalize = True)
    corpus_tfidf = tfidf[doc_term_matrix]
    return corpus_tfidf

def get_coherence_scores(corpus, dictionary, text, min_topics, max_topics):
    coherence_values = []
    model_list = []
    for num_topics_i in range(min_topics, max_topics+1):
        model = LsiModel(corpus, num_topics = num_topics_i, id2word = dictionary)
        model_list.append(model)
        coherence_model = CoherenceModel(model = model, texts=text, dictionary=dictionary, coherence = 'c_v')
        coherence_values.append(coherence_model.get_coherence())

    plt.plot(range(min_topics, max_topics+1), coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.legend( ("coherence_values"), loc="best")
    plt.show( )
    
corpus_tdf_fake = tfidf_corpus(doc_term_fake)
get_coherence_scores(corpus_tdf_fake, dictionary_fake , fake_news_text, min_topics = 2, max_topics = 11 )

# Build LSI model with optimal number of topics
lsi_model = LsiModel(corpus_tdf_fake, id2word = dictionary_fake, num_topics = 7) 
lsi_model.print_topics()

