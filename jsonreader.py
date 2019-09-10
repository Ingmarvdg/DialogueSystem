import json
import pandas as pd
from itertools import chain

json_log = "data/Mar13_S2A0/voip-00d76b791d-20130327_005342/log.json"
json_label = "data/Mar13_S2A0/voip-00d76b791d-20130327_005342/label.json"


## Grabs data from data folder. Needs a loop

# System data (in log.json)
data_sys = json.load(open(json_log))
df_sys = pd.DataFrame(data_sys["turns"])

# User data (in label.json)
data_usr = json.load(open(json_label))
df_usr = pd.DataFrame(data_usr["turns"])

# Make info list from session-id and task description
sessionid = data_sys["session-id"]
task = data_usr["task-information"]["goal"]["text"]
info = [sessionid, task, ""]



## Making the conversation

# Create list of all system responses
sys_lst = []
for i in range(len(df_sys)):
    sys_lst.append(df_sys["output"][i]['transcript'])
    
# Prepend "system: " to system responses
sys_lst = ["system: "+ s for s in sys_lst]


# Create list of all user responses
usr_lst = []
for i in range(len(df_usr)):
    usr_lst.append(df_usr["transcription"][i])
    
# Prepend "user: " to user responses
usr_lst = ["user: "+ s for s in usr_lst]
    
    
# Zip lists together
text = list(chain.from_iterable(zip(sys_lst, usr_lst)))

# Concatenate info and text
conversation = info + text