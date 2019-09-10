import pandas as pd
import json
from itertools import chain
import os
import time

# global variables
buttonPressed = True
dataDirectory = "data/"

# function for reading and parsing json to make the conversation readable
def readAndParseJson(log_url, label_url):

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

    return(conversation)

# function writes a conversation to a file
def writeConvoToFile(directory, conversation):
    fileName = conversation[0]
    file = open(""+directory+"/"+fileName+".txt", "x")
    file.write("\n".join(conversation))
    file.close()
    return

# write all conversations to files
def writeAllConvosToFile(conversations):
    dirName = "conversation" + str(time.time())
    try:
        os.mkdir(dataDirectory + dirName)
    except FileExistsError:
        print("directory already exists")
    for conversation in conversations:
        writeConvoToFile("" + dataDirectory + str(dirName), conversation)
    return


def parseAllJson(header):
    conversations = []
    entries = os.listdir(header)
    for parent_directories in entries:
        if parent_directories.startswith(".") == False:
            entries_child = os.listdir(header + parent_directories + '/')
            for child_directories in entries_child:
                if child_directories.startswith(".") == False:
                    json_log = header + parent_directories + '/' + child_directories + '/log.json'
                    json_label = header + parent_directories + '/' + child_directories + '/label.json'
                    ## Grabs data from data folder. Needs a loop
                    conversations.append(readAndParseJson(json_log, json_label))
    print(len(conversations))
    return conversations

header_test = 'data/dstc2_test/data/'
header_train = 'data/dstc2_traindev/data/'

test_conversations = parseAllJson(header_test)
train_conversations = parseAllJson(header_train)

all_conversations = test_conversations + train_conversations

amount = len(all_conversations)
print(amount)
currentAmount = 0

while(currentAmount < amount):
    text = input("Press ENTER to print the next item, write SAVE to save all conversations to files, write DONE to exit the program")
    if text == "":
        print ("you pressed enter")
        print("\n".join(all_conversations[currentAmount]))
        currentAmount+=1
    if text == "DONE":
        break
    if text == "SAVE":
        writeAllConvosToFile(all_conversations)
    else:
        print("Dont type anything beforehand if you want to see the next item")