# imports
import pandas as pd
import numpy as np
import pickle
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm, tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

np.random.seed(500)

# Paths to data - edi.t if needed. If no data files found, rund DialogeSystem.py first
train_data = r"DialogueSystem/train_data.txt"
test_data = r"DialogueSystem/test_data.txt"

# Preprocessing function
def preprocess(datapath):
    # Read & Preprocess
    Corpus = pd.read_csv(datapath, encoding='latin-1', header=None, names=['label', 'text'])
    new = Corpus["label"].str.split(":", n = 1, expand = True)
    Corpus["label"]= new[1] 
    new = Corpus["text"].str.split(":", n = 1, expand = True)
    Corpus["text"]= new[1]

    #Step - a : Remove blank rows if any.
    Corpus['text'].dropna(inplace=True)
    # Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
    Corpus['text'] = [entry.lower() for entry in Corpus['text']]
    # Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
    Corpus['text']= [word_tokenize(entry) for entry in Corpus['text']]
    # Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
    # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV

    for index,entry in enumerate(Corpus['text']):
        # Declaring Empty List to store the words that follow the rules for this step
        Final_words = []
        # Initializing WordNetLemmatizer()
        word_Lemmatized = WordNetLemmatizer()
        # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
        for word, tag in pos_tag(entry):
            # Below condition is to check for Stop words and consider only alphabets
            if word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                Final_words.append(word_Final)
        # The final processed set of words for each iteration will be stored in 'text_final'
        Corpus.loc[index,'text_final'] = str(Final_words)
        
    return(Corpus)

# Use accuracy_score function to get the accuracy
#print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)
#print("F1 Scores ->                  ",f1_score(Test_Y, predictions_NB, average="micro")*100)


## Side-by-side dataframe comparison of true labels and predictions
#Corpus = pd.DataFrame(Test_X)
#predictions_inversed = Encoder.inverse_transform(predictions_NB)
#true_label = Encoder.inverse_transform(Test_Y)
#Corpus['label'], Corpus['predict'] = Test_Y, predictions_NB