import pickle
import os
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, CuDNNLSTM
from keras.utils import print_summary
from keras.losses import categorical_crossentropy
from keras import optimizers
from keras import backend as K


# Constants
input_dir = 'data/pickles/'

n_categories = 30
batch_size = 32
input_freq = 20
n_time = 44

hidden_size=256
n_epochs = 1000
n_iters = 50

linear_output_size = 128



# Pickling 
label_2_num_mapping = {}
num_2_label_ar= []
data = {"x":[],"y":[]}

label_num = 0

for label in os.listdir(input_dir):
    if label == '.DS_Store' or label == '.ipynb_checkpoints':
        continue
    p_file = input_dir + label
    label_2_num_mapping[label]=label_num
    num_2_label_ar.append(label)

    with open(p_file, 'rb') as f:
        label_examples  = pickle.load(f)
        # data dim = num_examples * time * frequency
        label_ar = [np.swapaxes(np.array(example),0,1) for example in label_examples]
        data['x'].extend(label_ar)
        y_labels = [label_num for i in range(0,len(label_examples))]
        data['y'].extend(y_labels)
   
    label_num+=1


# Sampling
def sample_batch(n, X, Y):
    """
    takes input and returns padded sample
    n= num_samples
    X = input_featurs_list
    Y = label_list
    """
    label_ids = np.random.randint(low = 0,high = len(X), size=n)
    frequency = len(X[0][0])
    sampled_X = [X[label_id] for label_id in label_ids]
    sampled_y = [Y[label_id] for label_id in label_ids]
    padded_X = []
    
    max_batch_len = max([len(x) for x in sampled_X])
    for x in sampled_X:
        padding_time_count = max_batch_len-len(x)
        if padding_time_count!=0:
            x_padded = np.zeros(shape = (max_batch_len,frequency))
            x_padded[:x.shape[0],:x.shape[1]] = x
            padded_X.append(x_padded)
        else:
            padded_X.append(x)
        
    return np.asarray(padded_X),np.asarray(sampled_y)



# Generate Target Dataset
print("Generate Data")

# X_data = np.random.randn((20, 242, 10))
# Y_data = np.random.rand((20, 242, 2)) * 10

_inputs,_labels = sample_batch(batch_size, data['x'], data['y'])
# print(_inputs.shape)
# print(len(_inputs[1]))
# print(_labels.shape)
# print(_labels[:])


# Generate Targets
_targets = list()

# print(len(_labels))

for label in _labels:
    categories = [0 for x in range(n_categories)]
    categories[label] = 1
    
    sample = list()
    for time in range(n_time):
        sample.append(categories)
    
    _targets.append(sample)
    
    
_targets = np.asarray(_targets)
# print(_targets.shape)
# print("These are the targets", _targets)


print("Finish Generating Data")


# Model
model = Sequential()

model.add(CuDNNLSTM(128, return_sequences=True, input_shape=(44, 20)))
model.add(Dense(32, activation='relu'))
model.add(Dense(30, activation='linear'))

print("Finished defining model")


sgd = optimizers.SGD(lr=0.01)

model.compile(loss=categorical_crossentropy,
              optimizer=sgd,
              metrics=[categorical_crossentropy])

model.fit(_inputs, _targets, batch_size=batch_size, epochs=n_epochs, shuffle=True)


