import sklearn.metrics as skm

# dummy labels
dummy_labels_ml = ['ack', 'ack', 'affirm', 'affirm', 'repeat', 'affirm', 'affirm', 'ack', 'ack', 'repeat', 'repeat']
dummy_labels_rand = ['ack', 'ack', 'ack', 'ack', 'ack', 'affirm', 'repeat', 'affirm', 'ack', 'affirm', 'repeat']
dummy_labels_rules = ['inform', 'ack', 'ack', 'inform', 'affirm', 'ack', 'repeat', 'ack', 'ack', 'affirm', 'repeat']
dummy_labels_real = ['inform', 'ack', 'ack', 'affirm', 'affirm', 'affirm', 'repeat', 'ack', 'ack', 'affirm', 'repeat']

index_matrix = list(set(dummy_labels_real))

#print(skm.confusion_matrix(dummy_labels_real, dummy_labels_ml, labels=index_matrix))
#print(skm.classification_report(dummy_labels_real, dummy_labels_ml, labels=index_matrix))


