# imports
import pandas as pd
import numpy as np
import pickle
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
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
def load_data(datapath):
    Corpus = pd.read_csv(datapath, encoding='latin-1', header=None, names=['label', 'text'])
    new = Corpus["label"].str.split(":", n = 1, expand = True)
    Corpus["label"]= new[1] 
    new = Corpus["text"].str.split(":", n = 1, expand = True)
    Corpus["text"]= new[1]
    return(Corpus)

df_train = load_data(train_data)
df_test = load_data(test_data)

# Special preprocess function for single utterances
def load_single(str_sentence):
    Corpus = pd.DataFrame(np.array([["", ""]]), columns=["label", "text"])
    Corpus["text"][0] = str_sentence
    return(Corpus)

def preprocess(Corpus):
    # Read & Preprocess
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

    
def revert_predictions(predictions):
    list_predicts = []
    for i in range(len(predictions)):
        if predictions[i].argmax() == 14: list_predicts.append('thankyou')
        elif predictions[i].argmax() == 13: list_predicts.append('restart')
        elif predictions[i].argmax() == 12: list_predicts.append('request')
        elif predictions[i].argmax() == 11: list_predicts.append('reqmore')
        elif predictions[i].argmax() == 10: list_predicts.append('reqalts')
        elif predictions[i].argmax() == 9: list_predicts.append('repeat')
        elif predictions[i].argmax() == 8: list_predicts.append('null')
        elif predictions[i].argmax() == 7: list_predicts.append('negate')
        elif predictions[i].argmax() == 6: list_predicts.append('inform')
        elif predictions[i].argmax() == 5: list_predicts.append('hello')
        elif predictions[i].argmax() == 4: list_predicts.append('deny')
        elif predictions[i].argmax() == 3: list_predicts.append('confirm')
        elif predictions[i].argmax() == 2: list_predicts.append('bye')
        elif predictions[i].argmax() == 1: list_predicts.append('affirm')
        elif predictions[i].argmax() == 0: list_predicts.append('ack')
        else: list_predicts.append('')

    return(list_predicts)  


def label_single(string):
    # preprocessing using custom function
    string_prp = preprocess(load_single(string))
    
    # Fit tokenizer to this text bit
    tokenizer.fit_on_texts(string_prp['text_final'].values)
    X_single = tokenizer.texts_to_sequences(single['text_final'].values)
    X_single = pad_sequences(X_single, maxlen=MAX_SEQUENCE_LENGTH)
    
    # Make model prediction
    X_pred = model.predict(X_single)
    
    # Put the tokenizer back to fit on the normal text again
    tokenizer.fit_on_texts(Corpus_train['text_final'].values)
    
    # Revert Keras' encoding back to label
    return(revert_predictions(X_pred)[0])