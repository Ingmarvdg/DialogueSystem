from itertools import chain
import os
import time
import MLClassifier as mlc
import numpy as np
import pandas as pd
from Conversation import Conversation
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, SpatialDropout1D, LSTM
from keras.models import Sequential, load_model
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping
from keras.backend import set_session
import tensorflow as tf
import json

# import nltk
# nltk downloads
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')
# nltk.download('stopwords')

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} )
sess = tf.Session(config=config)
set_session(sess)

#   1A and 1B HELPER FUNCTIONS

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
    print("Parsing data in " + header + ' please wait a moment...')
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
    return conversations


#   START OF PROGRAM
buttonPressed = True
dataDirectory = "data/"
answer = 0
data_frequencies = None
try:
    dialog_act_model = load_model('seq_model.h5')
except:
    print('No model trained, run option 3 to train a model')
    pass

conversation_settings = {'classifier': 'rule',
                         'confirmation_all': True,
                         'info_per_utt': "any",
                         'levenshtein_dist': 2,
                         'allow_restarts': True,
                         'max_responses': np.inf,
                         'responses_uppercase': True,
                         'utt_lowercase': True
                         }
database_path = 'ontology/restaurantinfo.csv'
ontology_path = 'ontology/ontology_dstc2.json'
header_test = 'data/dstc2_test/data/'
header_train = 'data/dstc2_traindev/data/'

while True:

    print('Select an option:')
    answer = input('1)Part 1a: Domain modelling.\n'
                   '2)Part 1b: Produce text files.\n'
                   '3)Part 1b: Train dialog act classification model\n'
                   '4)Part 1c: Run dialog.\n'
                   '5)Part 1c: Change dialog settings.\n'
                   '6)EXIT\n')

    if answer == '1':
        print('Parsing data...')
        test_conversations = parse_all_json(header_test, answer)
        train_conversations = parse_all_json(header_train, answer)

        all_conversations = test_conversations + train_conversations
        amount = len(all_conversations)

        currentAmount = 0
        while currentAmount < amount:
            text = input(
                "Press ENTER to print the next item, write SAVE to save all conversations to files, write DONE "
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

        # pre process all data
        print("Started processing data, this may take a while...")
        Corpus_train = mlc.preprocess(train_data_path)
        print("Processing training data complete.")
        Corpus_test = mlc.preprocess(test_data_path)
        print("Processing test data complete.")

        # split to X and Y
        test_Y = Corpus_test['label']
        test_X = Corpus_test['text_final']
        # Tokenization setup
        MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM = 50000, 15, 100
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
        data_frequencies = random_test_y.value_counts(normalize=True)

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
        history = model.fit(X_tr, Y_tr, epochs=epochs, batch_size=batch_size, validation_split=0.1,
                            callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
        print('Training complete!')

        # Accuracy score on train set (test set couldn't because not all label types occur in test set (12/15))
        accr = model.evaluate(X_tr, Y_tr)
        print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))

        # store model to be used in the conversation
        model.save('seq_model.h5', save_format='tf')

    elif answer == '4':
        convo = Conversation(conversation_settings)
        convo.start_conversation()
        pass

    elif answer == '5':
        print('Select which parameter you want to change:')
        answer = input(f'1)Dialog act classifier, current: {conversation_settings["classifier"]}.\n'
                       f'2)Confirm all information, current: {conversation_settings["confirmation_all"]}.\n'
                       f'3)Information per utterance, current: {conversation_settings["info_per_utt"]}.\n'
                       f'4)Levenshtein distance, current: {conversation_settings["levenshtein_dist"]}.\n'
                       f'5)Allow restarts?, current: {conversation_settings["allow_restarts"]}.\n'
                       f'6)Maximum amount of responses, current: {conversation_settings["max_responses"]}.\n'
                       f'7)Responses in uppercase?, current: {conversation_settings["responses_uppercase"]}.\n'
                       f'8)Convert utterances to lowercase?, current: {conversation_settings["utt_lowercase"]}.\n'
                       f'9)Go back.')

        if answer == '1':
            print('Select which Dialog act classifier you want to use')
            answer = input('1)Rule based classifier.\n'
                           '2)Random classifier.\n'
                           '3)Machine learning classifier.\n'
                           )
            if answer == '1':
                conversation_settings["classifier"] = 'rule'
            if answer == '2':
                if not data_frequencies:
                    print("No data frequencies found, please train and test first")
                else:
                    conversation_settings['classifier'] = data_frequencies
            if answer == '3':
                if not dialog_act_model:
                    print("No classifier found, please train and test first")
                else:
                    conversation_settings["classifier"] = load_model('seq_model.h5')

        if answer == '2':
            print('Do you want all uttered information to be confirmed first?')
            answer = input('1)Yes.\n'
                           '2)No.\n'
                           )
            if answer == '1':
                conversation_settings["confirmation_all"] = True
            if answer == '2':
                conversation_settings['confirmation_all'] = False

        if answer == '3':
            print('How much information should each utterance contain?')
            answer = input('1)All fields.\n'
                           '2)One field max.\n'
                           '3)Any amount of fields'
                           )
            if answer == '1':
                conversation_settings["info_per_utt"] = "all"
            if answer == '2':
                conversation_settings['info_per_utt'] = "one"
            if answer == '3':
                conversation_settings['info_per_utt'] = 'any'

        if answer == '4':
            print('What is the maximum levenhstein distance')
            answer = input()
            conversation_settings['levenshtein_dist'] = answer

        if answer == '5':
            print('Allow restarts?')
            answer = input('1)Yes.\n'
                           '2)No.\n'
                           )
            if answer == '1':
                conversation_settings['allow_restarts'] = True
            if answer == '2':
                conversation_settings['allow_restarts'] = False

        if answer == '6':
            print('What is the maximum amount of utterances in the conversation?')
            answer = input()
            conversation_settings['levenshtein_dist'] = answer

        if answer == '7':
            print('Respond in uppercase?')
            answer = input('1)Yes.\n'
                           '2)No.\n'
                           )
            if answer == '2':
                conversation_settings['responses_uppercase'] = True
            if answer == '3':
                conversation_settings['responses_uppercase'] = False

        if answer == '8':
            print('Convert utterances to lowercase?')
            answer = input('1)Yes.\n'
                           '2)No.\n'
                           )
            if answer == '1':
                conversation_settings['utt_lowercase'] = True
            if answer == '2':
                conversation_settings['utt_lowercase'] = False

        else:
            pass

    elif answer == '6':
        break
