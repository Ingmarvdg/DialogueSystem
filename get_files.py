import os

json_log = "data/Mar13_S2A0/voip-00d76b791d-20130327_005342/log.json"
json_label = "data/Mar13_S2A0/voip-00d76b791d-20130327_005342/label.json"

entries = os.listdir('data/')
print(entries)
for parent_directories in entries:
	entries_child = os.listdir('data/' + parent_directories + '/')
	for child_directories in entries_child:
		log_file = 'data/' + parent_directories + '/' + child_directories + '/log.json'
		label_file = 'data/' + parent_directories + '/' + child_directories + '/label.json'
		## Grabs data from data folder. Needs a loop