import pandas as pd
import json
from itertools import chain
import os
import time
import random
import MLClassifier as mlc
import nltk
import numpy as np
import RulebasedEstimator as rbe

from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm, tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import sklearn.metrics as metrics

from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences 
from keras.models import Sequential
from keras.layers import Dense, SpatialDropout1D, Flatten, LSTM, GRU, ConvLSTM2D
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization
import tensorflow as tf

import numpy
import nltk

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# global variables
buttonPressed = True
dataDirectory = "data/"

# nltk downloads
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')

# function for reading and parsing json to make the conversation readable
def read_and_parse_json_conversation(log_url, label_url):
    # System data (in log.json)
    data_sys = json.load(open(log_url))
    df_sys = pd.DataFrame(data_sys["turns"])

    # User data (in label.json)
    data_usr = json.load(open(label_url))
    df_usr = pd.DataFrame(data_usr["turns"])

    # Make info list from session-id and task description
    session_id = data_sys["session-id"]
    task = data_usr["task-information"]["goal"]["text"]
    info = [session_id, task, ""]

    sys_lst = []
    for i in range(len(df_sys)):
        sys_lst.append(df_sys["output"][i]['transcript'])

    # Prepend "system: " to system responses
    sys_lst = ["system: " + s for s in sys_lst]

    # Create list of all user responses
    usr_lst = []
    for i in range(len(df_usr)):
        usr_lst.append(df_usr["transcription"][i])

    # Prepend "user: " to user responses
    usr_lst = ["user: " + s for s in usr_lst]

    # Zip lists together
    text = list(chain.from_iterable(zip(sys_lst, usr_lst)))

    # Concatenate info and text
    conversation = info + text

    return conversation


def read_and_parse_json_dialog_act(label_url):
    # User data (in label.json)
    data_usr = json.load(open(label_url))
    df_usr = pd.DataFrame(data_usr["turns"])

    # Create list of all user responses
    usr_lst = []

    for i in range(len(df_usr)):
        for dialog_act_single_value in df_usr['semantics'][i]['cam'].split('|'):
            usr_lst.append({'dialog_act': dialog_act_single_value.partition("(")[0],
                            'utterance_content': df_usr["transcription"][i]})
    return usr_lst


# function writes a conversation to a file
def write_convo_to_file(directory, conversation):
    file_name = conversation[0]
    file = open("" + directory + "/" + file_name + ".txt", "x")
    file.write("\n".join(conversation))
    file.close()
    return


# write all conversations to files
def write_all_convos_to_file(conversations):
    dir_name = "conversation" + str(time.time())
    try:
        os.mkdir(dataDirectory + dir_name)
    except FileExistsError:
        print("directory already exists")
    for conversation in conversations:
        write_convo_to_file("" + dataDirectory + str(dir_name), conversation)
    return


def parse_all_json(header, option):
    conversations = []
    entries = os.listdir(header)
    print("Parsing data in " + header +' please wait a moment...')
    for parent_directories in entries:
        if not parent_directories.startswith("."):
            entries_child = os.listdir(header + parent_directories + '/')
            for child_directories in entries_child:
                if not child_directories.startswith("."):
                    json_log = header + parent_directories + '/' + child_directories + '/log.json'
                    json_label = header + parent_directories + '/' + child_directories + '/label.json'
                    # Grabs data from data folder.
                    if option == '1':
                        conversations.append(read_and_parse_json_conversation(json_log, json_label))
                    elif option == '2':
                        conversations.append(read_and_parse_json_dialog_act(json_label))
    # print(len(conversations))
    return conversations


header_test = 'data/dstc2_test/data/'
header_train = 'data/dstc2_traindev/data/'

print('Select an option:')
answer = input('1)Part 1a: domain modelling. \n2)Part 1b: Produce text files.\n3)Part 1b: Train, Test, Evaluate.\n')

if answer == '1':
    print('Parsing data...')
    test_conversations = parse_all_json(header_test, answer)
    train_conversations = parse_all_json(header_train, answer)

    all_conversations = test_conversations + train_conversations
    amount = len(all_conversations)

    currentAmount = 0
    while currentAmount < amount:
        text = input("Press ENTER to print the next item, write SAVE to save all conversations to files, write DONE "
                     "to exit the program")
        if text == "":
            print("you pressed enter")
            print("\n".join(all_conversations[currentAmount]))
            currentAmount += 1
        if text == "DONE":
            break
        if text == "SAVE":
            write_all_convos_to_file(all_conversations)
        else:
            print("Dont type anything beforehand if you want to see the next item")
elif answer == '2':
    print('Parsing data...')

    temp_test_conversations = parse_all_json(header_test, answer)
    temp_train_conversations = parse_all_json(header_train, answer)

    all_conversations = temp_test_conversations + temp_train_conversations

    random.shuffle(all_conversations)

    number_training = round(len(all_conversations) * 85 / 100)
    number_test = round(len(all_conversations) * 15 / 100)

    test_conversations = []
    train_conversations = []
    for i in range(len(all_conversations)):
        if i < number_training:
            for data in all_conversations[i]:
                train_conversations.append(data)
        else:
            for data in all_conversations[i]:
                test_conversations.append(data)

    selected_train = []
    selected_test = []

    file = open("train_data.txt", "w")
    for i in range(len(train_conversations)):
        file.write('dialog_act:' + train_conversations[i]['dialog_act'] +
                   ', utterance_content:' + train_conversations[i]['utterance_content'] + '\n')
    file.close()
    print('train_data.txt was created successfully!')

    file = open("test_data.txt", "w")
    for i in range(len(test_conversations)):
        file.write('dialog_act:' + test_conversations[i]['dialog_act'] +
                   ', utterance_content:' + test_conversations[i]['utterance_content'] + '\n')
    file.close()
    print('test_data.txt was created successfully!')

elif answer == '3':
    # open training and testing data
    train_data_path = r"train_data.txt"
    test_data_path = r"test_data.txt"

    # open hard case data
    hard_test_data_path = r"Mistakes.txt"
    negation_test_data_path = r"Negation.txt"

    # pre process all data
    print("Started processing data, this may take a while...")
    Corpus_train = mlc.preprocess(train_data_path)
    print("Processing training data complete.")
    Corpus_test = mlc.preprocess(test_data_path)
    print("Processing test data complete.")
    hard_test_data = mlc.preprocess(hard_test_data_path)
    print("Processing hard test data complete.")
    negation_test_data = mlc.preprocess(negation_test_data_path)
    print("Processing negation test data complete")
    print("All processing completed.")

    # Tokenization setup
    MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM = 50000, 1000, 100
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(Corpus_train['text_final'].values)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    # Define X and Y
    X_tr = tokenizer.texts_to_sequences(Corpus_train['text_final'].values)
    X_tr = pad_sequences(X_tr, maxlen=MAX_SEQUENCE_LENGTH)
    X_tt = tokenizer.texts_to_sequences(Corpus_test['text_final'].values)
    X_tt = pad_sequences(X_tt, maxlen=MAX_SEQUENCE_LENGTH)
    X = tf.concat([X_tr, X_tt], 0)
    Y_tr = pd.get_dummies(Corpus_train['label']).values
    Y_tt = pd.get_dummies(Corpus_test['label']).values

    # get predictions for baseline random
    random_test_y = pd.Series(test_Y)
    frequencies = random_test_y.value_counts(normalize=True)
    random_preds = np.random.choice(list(frequencies.index.values), len(test_Y), list(frequencies.values))

    # get predictions for baseline rule based
    rule_preds = rbe.Classify(train_X)
    rule_preds = Encoder.transform(rule_preds)

    # get predictions machine learning model of test set
    print('Compiling model...')
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(15, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    epochs = 5
    batch_size = 64
    print('Training model... this might take a long while (15+ minutes).')
    history = model.fit(X_tr, Y_tr, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
    print('Training complete!')

    # Accuracy score on train set (test set couldn't because not all label types occur in test set (12/15))
    accr = model.evaluate(X_tr,Y_tr)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
    
    # Plots
    # 1) Loss
    #plt.rcParams['figure.figsize'] = 4.5,3.5
    #plt.csfont = {'fontname':'Times New Roman'}
    #plt.title('LSTM Training Loss', **plt.csfont)
    #plt.plot(history.history['loss'], label='train')
    #plt.plot(history.history['val_loss'], label='test')
    #plt.legend()
    #plt.show();

    # 2) Accuracy per epoch
    #plt.title('LSTM Accuracy per epoch', **plt.csfont)
    #plt.plot(history.history['acc'], label='train')
    #plt.plot(history.history['val_acc'], label='test')
    #plt.legend()
    #plt.show();

    # 3) Plot of Test label set (behold; it does not contain all available label types...)
    #D = dict(collections.Counter(Corpus_test['label']).items()) # sorted by key, return a list of tuples
    #plt.barh(range(len(D)), list(D.values()), align='center')
    #plt.yticks(range(len(D)), list(D.keys()), **plt.csfont)
    #plt.title('Train label frequencies', **plt.csfont)
    #plt.show()



    # for each of the predictors print classification report

    #  show graph that compares weighted f1 scores, recall and precision of baseline 1, baseline 2 and machine learning model

    # loop to enter a sentence and let the ml model classify it

