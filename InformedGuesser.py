import pandas as pd
import numpy as np

# takes a 1 x N array with all data_labels
def DialogActFreqs(data_labels):
    freqs = pd.DataFrame(columns=['acts', 'frequency'])
    length = len(data_labels)
    dialog_acts = set(data_labels)

    for act in dialog_acts:
        count = data_labels.count(act)
        freqs = freqs.append({'acts': act, 'frequency': count / length}, ignore_index=True)
        print(freqs)

    # return an array with row 0 having the unique acts and row 1 having their probability of occurring
    return freqs

# takes a 2 x N array where row 0 is all possible dialog acts and row 1 is the probability of them occurring, n is the amount of guesses
def InformedGuess(frequencies, n_guess):
    guess = np.random.choice(frequencies[0], n_guess, frequencies[1])

    # return n amount of guesses
    return guess