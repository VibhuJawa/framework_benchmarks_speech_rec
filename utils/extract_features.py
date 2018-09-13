import sys
import os
import numpy
import librosa
import itertools
import pickle

def generate(dir, disp=False):
    dir = os.fsencode(sys.argv[1])
    data = []
    numFiles = len(os.listdir(sys.argv[1]))
    for num, file in enumerate(os.listdir(dir)):
        fname = os.fsencode(file)

        y, sr = librosa.load(dir+fname)
        data.append(librosa.feature.mfcc(y=y, sr=sr))
        if disp: print('processing wavs '+str(round(100*(num+1)/numFiles, 2))+'%', end='\r')

    if disp: print('padding... ', end='')
    maxRows = max([elem.shape[0] for elem in data])
    maxCols = max([elem.shape[1] for elem in data])
    padded = [np.zeros((maxRows, maxCols)) for _ in data]
    for ins, zer in zip(data, padded): zer[:ins.shape[0], :ins.shape[1]] = ins
    if disp: print('done')
   
    return padded




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

