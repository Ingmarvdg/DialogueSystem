# imports
import MLClassifier as mlc
import pandas as pd
from PrunedDialogSystem.Conversation import Conversation
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, SpatialDropout1D, LSTM
from keras.models import Sequential, load_model
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping
from keras.backend import set_session
import tensorflow as tf

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} )
sess = tf.Session(config=config)
set_session(sess)

while True:
    print('select an option')
    answer = input('1)Retrain classifier\n'
                   '2)Run dialog\n'
                   '3)EXIT\n')

    if answer == '1':
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

    if answer == '2':
        classifier = None
        case = True

        try:
            classifier = load_model('seq_model.h5')
        except OSError:
            print('no classifier found, train it first by choosing option 1')
            raise

        print('select a case')
        answer = input('1) Case 1\n'
                       '2) Case 2\n'
                       '3)EXIT\n')

        if answer == '1':
            case = True
        elif answer == '2':
            case = False
        else:
            break

        convo = Conversation(classifier, 2, case)
        convo.start_conversation()
        pass

    if answer == '3':
        break
