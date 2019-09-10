import os

entries = os.listdir('data/')
for parent_directories in entries:
	entries_child = os.listdir('data/' + parent_directories + '/')
	for child_directories in entries_child:
		log_file = 'data/' + parent_directories + '/' + child_directories + '/log.json'
		label_file = 'data/' + parent_directories + '/' + child_directories + '/label.json'
		## Grabs data from data folder. Needs a loop
