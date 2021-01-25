import os
import getpass


from utils import *
from keras.models import Sequential, Model, load_model
from keras.layers import Flatten, Dense, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


#user defined variables
version='imgsize64_batchsize64_4layers_2048_512_128'
IMG_SIZE   = 32 #deafult 32
BATCH_SIZE  = 64 #default 16
DATASET_DIR = '../../Databases/MIT_split'#'/home/mcv/datasets/MIT_split'
MODEL_FNAME = f'output_results/my_first_mlp_{version}.h5'

if not os.path.exists(DATASET_DIR):
  print(Color.RED, 'ERROR: dataset directory '+DATASET_DIR+' does not exist!\n')
  quit()


print('Building MLP model...\n')

#Build the Multi Layer Perceptron model

####
# Split into tiles
####

model = Sequential()
model.add(Reshape((IMG_SIZE*IMG_SIZE*3,),input_shape=(IMG_SIZE, IMG_SIZE, 3),name='first'))
model.add(Dense(units=2048, activation='relu',name='second'))
model.add(Dense(units=512, activation='relu', name='3'))
model.add(Dense(units=128, activation='relu', name='4'))
model.add(Dense(units=8, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

print(model.summary())
#plot_model(model, to_file=f'output_results/feature_extraction_{version}.png', show_shapes=True, show_layer_names=True)

model.load_weights(MODEL_FNAME)
#plot_model(model, to_file=f'output_results/feature_extraction_{version}.png', show_shapes=True, show_layer_names=True)

print('Done!\n')

#to get the output of a given layer
 #crop the model up to a certain layer
model_layer = Model(inputs=model.input, outputs=model.get_layer('4').output)

#get the features from images
directory = DATASET_DIR+'/test/coast'
x = np.asarray(Image.open(os.path.join(directory, os.listdir(directory)[0])))
x = np.expand_dims(np.resize(x, (IMG_SIZE, IMG_SIZE, 3)), axis=0)
print('prediction for image ' + os.path.join(directory, os.listdir(directory)[0]))
features = model_layer.predict(x/255.0)
print(features.shape)
print(features)
print('Done!')




