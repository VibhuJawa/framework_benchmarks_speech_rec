import sys
import os
import numpy
import librosa
import itertools
import pickle

def generate(dir_name, disp=False):
    dir = os.fsencode(dir_name)
    data = []
    numFiles = len(os.listdir(dir_name))
    for num, file in enumerate(os.listdir(dir)):
        fname = os.fsencode(file)

        y, sr = librosa.load(dir+fname)
        data.append(list(itertools.chain.from_iterable(librosa.feature.mfcc(y=y, sr=sr))))
        print('processing wavs '+str(round(100*(num+1)/numFiles, 2))+'%', end='\r')
    print('')
    length = min([len(elem) for elem in data])
    return numpy.array([numpy.array(l[:length]) for l in data])

if '__main__' == __name__:
    if len(sys.argv)<3:
        print('provide target dir and source dir')
        exit()
    input_dir = sys.argv[1]
    target_dir = sys.argv[2]
    os.makedirs(target_dir,exist_ok=True)
    print(input_dir,target_dir)
    for label in os.listdir(input_dir):
        if label == '.DS_Store':
            continue
        curr_dir = input_dir + label +'/'
        print(curr_dir)
        with open(target_dir + '{}_features.pickle'.format(label), 'wb') as f: pickle.dump(generate(curr_dir, disp=True), f)
        print('wrote {}_features.pickle'.format(label))

