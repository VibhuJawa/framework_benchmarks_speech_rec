import os
import pickle

data_dir = "../data/pickles"
scaleup = 10 
files = os.listdir(data_dir)
for file in files:
    pickled_file = pickle.load(open("{}/{}".format(data_dir,file),"rb"))
    pickled_file = pickled_file * scaleup
    print("Len now is {}".format(len(pickled_file)))
    pickle.dump( pickled_file, open( "{}/{}".format(data_dir,file), "wb" ) )
