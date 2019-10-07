#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 16:27:50 2019

@author: elfiakibv
"""

import numpy as np


def predict(sentence):
    sentence = sentence.split(" ",)

    if 'yes' in sentence or 'yeah' in sentence or 'yeap' in sentence or 'right' in sentence or 'good' in sentence and len(sentence) < 2:
        act = "affirm"

    elif 'no' in sentence:
        act = "negate"

    elif 'okay' in sentence or 'k' in sentence or 'ok' in sentence or 'um' in sentence and len(sentence) < 2:
        act = "ack"

    elif 'hi' in sentence or 'hello' in sentence:
        act = "hello"

    elif 'is' in sentence[0] or 'do' in sentence[0] or 'does' in sentence[0]:
        act = "confirm"

    elif 'not' in sentence or 'dont' in sentence:
        act = "deny"

    elif 'thanks'  in sentence or 'thank' in sentence:
        act = "thankyou"

    elif 'repeat' in sentence or 'again' in sentence:
        act = "repeat"

    elif 'different' in sentence or 'else' in sentence:
        act = "reqalts"

    elif 'more' in sentence:
        act = "reqmore"

    elif 'what' in sentence[0] or 'where' in sentence [0] or 'give'  in sentence or 'information' in sentence or 'number' in sentence or 'address' in sentence or 'post' in sentence:
        act = "request"

    elif 'start' in sentence or 'restart' in sentence:
        act = "restart"

    elif 'looking' in sentence or 'food' in sentence or 'want' in sentence:
        act = "inform"

    elif 'bye' in sentence or 'goodbye' in sentence or 'bb' in sentence or 'see you' in sentence:
        act = "bye"

    else:
        act = 'null'

    return act


def classify_multi(list_of_sentences):
    list_of_acts = []
    
    for sentence in list_of_sentences:
        act = predict(sentence)
        list_of_acts.append(act) 

    list_of_acts = np.asarray(list_of_acts)
    return list_of_acts

