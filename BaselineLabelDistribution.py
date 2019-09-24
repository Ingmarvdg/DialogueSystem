#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 17:37:23 2019

@author: elfiakibv
"""
import random, pandas

def Frequency(sentences):

    frequency = []  
    sentence_act=[]
    
    for sentence in sentences:
        sentence = sentence.split(" ", )
        sentence_act.append(sentence[0])

    f_1 = sentence_act.count('(ack)')/len(sentence_act) 
    f_2 = sentence_act.count('(affirm)')/len(sentence_act)
    f_3 = sentence_act.count('(bye)')/len(sentence_act)
    f_4 = sentence_act.count('(confirm)')/len(sentence_act)
    f_5 = sentence_act.count('(deny)')/len(sentence_act)
    f_6 = sentence_act.count('(hello)')/len(sentence_act)
    f_7 = sentence_act.count('(inform)')/len(sentence_act)
    f_8 = sentence_act.count('(negate)')/len(sentence_act)
    f_9 = sentence_act.count('(null)')/len(sentence_act)
    f_10 = sentence_act.count('(repeat)')/len(sentence_act)
    f_11 = sentence_act.count('(reqalts)')/len(sentence_act)
    f_12 = sentence_act.count('(reqmore)')/len(sentence_act)
    f_13 = sentence_act.count('(request)')/len(sentence_act)
    f_14 = sentence_act.count('(restart)')/len(sentence_act)
    f_15 = sentence_act.count('(thankyou)')/len(sentence_act)
   
    frequency = [['ack', f_1], ['affirm', f_2], ['bye', f_3], ['confirm', f_4], ['deny', f_5], ['hello', f_6], ['inform', f_7], ['negate', f_8], ['null', f_9], ['repeat', f_10], ['reqalts', f_11], ['reqmore', f_12], ['request', f_13], ['restart', f_14], [' thankyou', f_15]]
    frequency = pandas.DataFrame(frequency, columns = ['Act', 'Frequency'])
    frequency = frequency.sort_values('Frequency')
    return frequency 


def CumulativeDistribution(frequency):
    
    ascending_frequency =[]
    distribution = []

    ascending_frequency = list(frequency['Frequency'])
    distribution.append(ascending_frequency[0]) 
    for i in range(1, len(ascending_frequency)):
        distribution.append(ascending_frequency[i] + distribution[i-1])


    return distribution

def LabelsAssigning(distribution):
    
    r = random.random()
    
    for i in range (len(distribution)):
        if r <= distribution[i]:
            labelled = distribution[i]
            break
            
    return labelled

def getAct(frequency, distribution, labelled):
    
    frequency.Frequency = distribution
    index_label = list(frequency['Frequency']).index(labelled)
    act = list(frequency['Act'])[index_label]
    
    return act
