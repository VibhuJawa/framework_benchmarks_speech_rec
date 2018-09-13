import sys
import os
import numpy
import librosa
import itertools
import pickle

def generate(dir, disp=False):
    dir = os.fsencode(curr_dir)
    data = []
    numFiles = len(os.listdir(curr_dir))
    for num, file in enumerate(os.listdir(dir)):
        fname = os.fsencode(file)
        
        y, sr = librosa.load(dir+fname)
        data.append(librosa.feature.mfcc(y=y, sr=sr))
        print('processing wavs '+str(round(100*(num+1)/numFiles, 2))+'%', end='\r')
    print('')
    return data




if '__main__' == __name__:
    if len(sys.argv)<3:
        print('provide target dir and source dir')
        exit()
    input_dir = sys.argv[1]
    target_dir = sys.argv[2]
    os.makedirs(target_dir,exist_ok=True)
    for label in os.listdir(input_dir):
        if label == '.DS_Store':
            continue
        curr_dir = input_dir + label +'/'
        print(curr_dir)
        with open(target_dir + '{}_features.pickle'.format(label), 'wb') as f: pickle.dump(generate(curr_dir, disp=True), f)
        print('wrote {}_features.pickle'.format(label))

