import numpy as np
import math

CHANNELS = 12
SPEAKERS = 9
TRAIN_TEST = [270, 370]

def readfile(path, type = 0):
    
    data = ''
    with open(path, 'r') as f:
        for line in f:
            if not line.isspace():
                data += line 
    
    df = data.split("\n")
    df = df[:-1]
    new = np.zeros((len(df), CHANNELS))

    for i in range(len(df)):
        new[i] = df[i].split()
    
    readIndex = 0
    res = []
    for i in range(TRAIN_TEST[type]):
        l = 0
        while new[readIndex][0] != 1.0 and new[readIndex][-1] != 1.0:
            l += 1
            readIndex += 1

        res.append(new[readIndex -l : readIndex])
        readIndex += 1
    
    res = np.array(res, dtype = object)
    return res

def calc_length(test, train):
    length = 0
    for i in range(test.shape[0]):
        if test[i].shape[0] >= length:
            length = test[i].shape[0]
    
    for i in range(train.shape[0]):
        if train[i].shape[0] >= length:
            length = train[i].shape[0]
    
    return length

def padding_data(data, maxlength, type = 0, flat = False, one_hot = False):

    input_data = np.zeros((data.shape[0], maxlength, CHANNELS))
    if one_hot:
        output_data = np.zeros((data.shape[0], SPEAKERS))
    else:
        output_data = np.zeros((data.shape[0], 1))

    speaker = 0
    blockCounter = 0
    blockLengths = [31, 35, 88, 44, 29, 24, 40, 50, 29]

    for i in range(data.shape[0]):
        if type:
            if blockCounter == blockLengths[speaker]:
                speaker += 1
                blockCounter = 1
            else:
                blockCounter += 1
        
        else:
            speaker = max(1, math.ceil(i / 30)) -1
        
        col_out = np.zeros((maxlength))
        col_length = data[i].shape[0]
        col_out[:col_length] = np.ones((col_length))
        input_data[i][:col_length] = data[i]
        if one_hot:
            output_data[i] = one_hot_encoding(speaker)
        else:
            output_data[i] = speaker
    
    if flat:
        input_data = flatten(input_data)
    
    return input_data, output_data

def one_hot_encoding(speaker):
    output = np.zeros((SPEAKERS))
    output[speaker] = 1

    return output

def flatten(data):
    a, b, c = data.shape
    return data.reshape(a, b*c)