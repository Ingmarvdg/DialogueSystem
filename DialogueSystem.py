import pandas as pd
import json
from itertools import chain
import os
import time

# global variables
buttonPressed = True
amount = 5
currentAmount = 0
dataDirectory = "data/"

json_log = "data/dstc2_test/data/Mar13_S2A0/voip-00d76b791d-20130327_005342/log.json"
json_label = "data/dstc2_test/data/Mar13_S2A0/voip-00d76b791d-20130327_005342/label.json"

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
    file.write("\n".join(readAndParseJson(json_log, json_label)))
    file.close()
    return


# write all conversations to files
def writeAllConvosToFile():
    file = 0

    return file

while(currentAmount < amount):
    text = input("Press ENTER to print the next item, write SAVE to save all conversations to files, write DONE to exit the program")
    if text == "":
        print ("you pressed enter")
        print("\n".join(readAndParseJson(json_log, json_label)))
        currentAmount+=1
    if text == "DONE":
        break
    if text == "SAVE":

        dirName = "conversations"
        try:
            # Create target Directory
            os.mkdir(dataDirectory + dirName + time.time())
            print("Directory ", dirName, " Created ")
        except FileExistsError:
            print("Directory ", dirName, " already exists")

        writeConvoToFile("" + dataDirectory + str(dirName), readAndParseJson(json_log, json_label))
        print("created directory in " + dataDirectory + str(dirName))
    else:
        print("Dont type anything beforehand if you want to see the next item")