# 🧠 Fake News Detection using NLP and Machine Learning

### 📄 **Project Overview**
This project aims to detect **fake news articles** using **Natural Language Processing (NLP)** and **Machine Learning** techniques.  
It performs **linguistic analysis**, **sentiment detection**, **topic modeling**, and **classification** to distinguish between *fake* and *factual* news articles.

---

## ⚙️ **Key Features**
✅ **Exploratory Data Analysis:** Visualizing the distribution of fake vs factual news  
✅ **Linguistic Analysis:** POS tagging and Named Entity Recognition (NER) with **spaCy**  
✅ **Text Preprocessing:** Lowercasing, stopword removal, tokenization, and lemmatization  
✅ **Sentiment Analysis:** Polarity scoring using **VADER**  
✅ **Topic Modeling:** **Latent Dirichlet Allocation (LDA)** and **Latent Semantic Indexing (LSI)**  
✅ **Machine Learning Models:**
- Bag of Words + Logistic Regression  
- TF-IDF + Support Vector Machine (SVM)  
✅ **Visualization:** Word frequencies, sentiment distributions, and topic coherence plots  

---

## 🧰 **Tech Stack**

| Category | Tools & Libraries |
|-----------|------------------|
| **Language** | Python 3.x |
| **NLP** | [spaCy](https://spacy.io/), [NLTK](https://www.nltk.org/), [Gensim](https://radimrehurek.com/gensim/) |
| **Sentiment Analysis** | [VADER Sentiment](https://github.com/cjhutto/vaderSentiment) |
| **Machine Learning** | [Scikit-learn](https://scikit-learn.org/stable/) |
| **Visualization** | [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/) |
| **Data Handling** | [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) |

---

## 📁 **Project Structure**

📦 Fake-News-Detection
│
├── 📄 fake_news_detection.py # Main project script
├── 📄 fake_news_data.csv # Dataset (or link to dataset)
├── 📄 requirements.txt # Required Python libraries
├── 📄 README.md # Project documentation
└── 📊 outputs/ # Visualizations and topic plots




## 📁 **🧪 Model Workflow**
🔹 Data Cleaning

Regex-based text cleaning, punctuation removal, and normalization.

🔹 Text Representation

Using Bag-of-Words (BoW) and TF-IDF vectorization for feature extraction.

🔹 Exploration

Visualizing POS and NER patterns using spaCy.

🔹 Sentiment Analysis

Computing polarity scores and visualizing sentiment class distribution.

🔹 Topic Modeling

Performing LDA and LSI topic extraction with coherence evaluation to find the optimal number of topics.

🔹 Classification

Training and evaluating Logistic Regression and SVM models with accuracy and classification metrics.


## 📊 **Results**

| Model                | Features     | Accuracy |
| -------------------- | ------------ | -------- |
| Logistic Regression  | Bag of Words | ~90%     |
| SVM (SGD Classifier) | TF-IDF       | ~91–93%  |


##💡 **Future Improvements**

🔹 Integrate BERT or Transformer-based embeddings for richer semantic understanding
🔹 Deploy as a Flask or FastAPI web application
🔹 Extend dataset and include multilingual support


## 👨‍💻 **Author**

Selim Ben Haj Braiek
🎓 Master’s Student in Data Science & Artificial Intelligence
💼 Aspiring AI Engineer | Full-Stack Developer | Blockchain Enthusiast

