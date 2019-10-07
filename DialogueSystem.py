import pandas as pd
from itertools import chain
import os
import time
import MLClassifier as mlc
import numpy as np
import RulebasedEstimator as rbe

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

from OntologyHandler import *
from Conversation import Conversation

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
    # print(len(conversations))
    return conversations


header_test = 'data/dstc2_test/data/'
header_train = 'data/dstc2_traindev/data/'

# Delete after test.
# Guide on how to use extracting_preferences() function.
ontology_data = read_json_ontology('ontology/ontology_dstc2.json')
restaurants = read_csv_database('ontology/restaurantinfo.csv')
utterance_content = "I'm looking for african food."
# Initialise preferences
preferences = dict(food=[], pricerange=[], restaurantname=[], area=[])
# extracting_preferences() will be used only if we want to update the client's preference
# so when dialog_act is 'inform'.
dialog_act = 'inform'
if dialog_act == 'inform':
    preferences = get_preferences(utterance_content, ontology_data, preferences)
    print(preferences)
    print(get_info_from_restaurant(preferences, restaurants))
    # bot will suggest a restaurant based on the preferences
utterance_content = 'Sorry I also want the restaurant to be in the north or south area.'

if dialog_act == 'inform':
    # So here the preference will be updated. In our example, adding the east area.
    preferences = get_preferences(utterance_content, ontology_data, preferences)
    print(preferences)
    print(get_info_from_restaurant(preferences, restaurants))
#########################

print('Select an option:')
answer = input('1)Part 1a: Domain modelling.\n'
               '2)Part 1b: Produce text files.\n'
               '3)Part 1b: Train, Test, Evaluate.\n'
               '4)Part 1c: Run dialog.\n')

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
    train_data = mlc.preprocess(train_data_path)
    print("Processing training data complete.")
    test_data = mlc.preprocess(test_data_path)
    print("Processing test data complete.")
    hard_test_data = mlc.preprocess(hard_test_data_path)
    print("Processing hard test data complete.")
    negation_test_data = mlc.preprocess(negation_test_data_path)
    print("Processing negation test data complete")
    print("All processing completed.")

    # split to X and Y
    train_X, train_Y = train_data['text_final'], train_data['label']
    test_Y = test_data['label']
    test_X = test_data['text_final']
    hard_test_X, hard_test_Y = hard_test_data['text_final'], hard_test_data['label']
    negation_test_X, negation_test_Y = negation_test_data['text_final'], negation_test_data['label']

    # label encoding
    Encoder = LabelEncoder()
    train_Y = Encoder.fit_transform(train_Y)
    test_Y = Encoder.fit_transform(test_Y)

    # get predictions for baseline random
    random_test_y = pd.Series(test_Y)
    frequencies = random_test_y.value_counts(normalize=True)
    random_preds = np.random.choice(list(frequencies.index.values), len(test_Y), list(frequencies.values))

    # get predictions for baseline rule based
    rule_preds = rbe.Classify(test_X)
    rule_preds = Encoder.transform(rule_preds)

    # get predictions machine learning model of test set
    Tfidf_vect = TfidfVectorizer(max_features=5000)
    Tfidf_vect.fit(train_X)
    Train_X_Tfidf = Tfidf_vect.transform(train_X)
    Test_X_Tfidf = Tfidf_vect.transform(test_X)

    Model = naive_bayes.MultinomialNB(alpha=0.43)  # - 58%
    Model.fit(Train_X_Tfidf, train_Y)
    ml_preds = Model.predict(Test_X_Tfidf)

    # print classification reports for both baselines and the ml classifier
    random_report = classification_report(test_Y, random_preds)
    rule_report = classification_report(test_Y, rule_preds)
    ml_report = classification_report(test_Y, ml_preds)
    print(ml_report)

    # make predictions for hard data, not for random since we can assume that performance will remain unchanged
    hard_rule_preds = rbe.Classify(hard_test_X)
    negation_rule_preds = Encoder.transform(hard_rule_preds)
    print(f1_score(hard_test_Y, hard_rule_preds, average='weighted'))

    hard_test_tfidf = Tfidf_vect.transform(hard_test_X)
    hard_ml_preds = Model.predict(hard_test_tfidf)
    print(f1_score(hard_test_Y, hard_ml_preds, average='weighted'))

    # make predictions for negation data
    negation_rule_preds = rbe.Classify(negation_test_X)
    negation_rule_preds = Encoder.transform(negation_rule_preds)
    print(f1_score(negation_test_Y, negation_rule_preds, average='weighted'))

    negation_test_tfidf = Tfidf_vect.transform(negation_test_X)
    negation_ml_preds = Model.predict(negation_test_tfidf)
    print(f1_score(negation_test_Y, negation_ml_preds, average='weighted'))

elif answer == '4':
    while True:
        print('Select an option:')
        answer = input('1)Let\'s chat!\n'
                       '2)Change Settings\n'
                       '3)Exit\n')
        if answer == '1':
            convo = Conversation(classifier='rule')
            convo.start_conversation()
            pass
        elif answer == '2':
            # todo Menu for settings.
            # todo Function that change settings in the Conversation class.
            pass
        elif answer == '3':
            break
