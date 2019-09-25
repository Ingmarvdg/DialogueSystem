import pandas as pd
import json
from itertools import chain
import os
import time
import random
import InformedGuesser as ig
import MLClassifier as mlc
import nltk
import numpy as np

import numpy

import InformedGuesser as ig
import MLClassifier as mlc
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
answer = input('1)Part 1a: domain modelling. \n2)Part 1b: Produce text files.\n3)Part 1b: Train, Test, Evaluate')

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

    number_85_training = round(len(all_conversations) * 85 / 100)
    number_15_test = round(len(all_conversations) * 15 / 100)

    random.shuffle(temp_test_conversations)
    test_conversations = []
    for data_in_one_file in temp_test_conversations:
        for data in data_in_one_file:
            test_conversations.append(data)

    random.shuffle(temp_train_conversations)
    train_conversations = []
    for data_in_one_file in temp_train_conversations:
        for data in data_in_one_file:
            train_conversations.append(data)

    selected_train = []
    selected_test = []

    file = open("train_data.txt", "w")
    for i in range(number_85_training):
        file.write('dialog_act:' + train_conversations[i]['dialog_act'] +
                   ', utterance_content:' + train_conversations[i]['utterance_content'] + '\n')
    file.close()
    print('train_data.txt was created successfully!')

    file = open("test_data.txt", "w")
    for i in range(number_15_test):
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
    #train_data = mlc.preprocess(train_data_path)
    print("Processing training data complete.")
    test_data = mlc.preprocess(test_data_path)
    print("Processing test data complete.")
    hard_test_data = mlc.preprocess(hard_test_data_path)
    print("Processing hard test data complete.")
    negation_test_data = mlc.preprocess(negation_test_data_path)
    print("Processing negation test data complete")
    print("All processing completed.")

    # split to X and Y
    test_Y = test_data['label']
    # get predictions for baseline random
    frequencies = test_Y.value_counts(normalize=True)
    random_preds = np.random.choice(frequencies.index.values, len(test_Y), frequencies.values)

    # get predictions for baseline rulebased


# get predictions for baseline random
#print(train_data)

# get predictions for baseline rulebased

# get predictions machine learning model of testset

# for each of the predictors print classification report

#  show graph that compares weighted f1 scores, recall and precision of baseline 1, baseline 2 and machine learning model

# loop to enter a sentence and let the ml model classify it

