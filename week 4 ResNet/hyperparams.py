from kerastuner.applications import HyperResNet
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from keras import optimizers
from keras import backend
import tensorflow as tf
import kerastuner as kt
import matplotlib.pyplot as plt
import os
import IPython
import time
import pickle as cPickle

end='activation_37'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
gpus = tf.config.experimental.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(gpus[0], True)

#end='activation_'+str(idx)
REMOTE = False
BOARD_PATH = 'boards/'
HYPERBAND_EPOCHS = 25
HYPERBAND_FACTOR = 5
EXPERIMENT_NAME = f'hb_E{HYPERBAND_EPOCHS}_F{HYPERBAND_FACTOR}'
MODEL_FNAME = f'models/modelRN50_{EXPERIMENT_NAME}.h5'
EPOCH_ARR=[50]

train_data_dir='../datasets/MIT_split/train_400'
test_data_dir='../datasets/MIT_split/train_400'
val_data_dir='../datasets/MIT_split/test'

train_labels = cPickle.load(open('../datasets/MIT_split/labels/train_labels.dat','rb'))
test_labels = cPickle.load(open('../datasets/MIT_split/labels/test_labels.dat','rb'))

if REMOTE:

    BOARD_PATH = '/home/group02/working/week4/outs/'
    EXPERIMENT_NAME = f'hyperparams'

    train_data_dir='/home/group02/datasets/MIT_split/train'
    test_data_dir='/home/mcv/datasets/MIT_split/train'
    val_data_dir='/home/mcv/datasets/MIT_split/test'


img_width = 224
img_height=224
batch_size=16
validation_samples=807

class ClearTrainingOutput(tf.keras.callbacks.Callback):
  def on_train_end(*args, **kwargs):
    IPython.display.clear_output(wait = True)


def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_data_format()
    assert dim_ordering in {'channels_first', 'channels_last'}

    if dim_ordering == 'channels_first':
        # 'RGB'->'BGR'
        x = x[ ::-1, :, :]
        # Zero-center by mean pixel
        x[ 0, :, :] -= 103.939
        x[ 1, :, :] -= 116.779
        x[ 2, :, :] -= 123.68
    else:
        # 'RGB'->'BGR'
        x = x[:, :, ::-1]
        # Zero-center by mean pixel
        x[:, :, 0] -= 103.939
        x[:, :, 1] -= 116.779
        x[:, :, 2] -= 123.68
    return x

LR_list = [0.01]
# LR_list = [0.1]
LR_results_dict = {}
d = {}
EPOCHS = 50
LR = 0.01
results_dir=f'hyperparams/_E{HYPERBAND_EPOCHS}_F{HYPERBAND_FACTOR}'
results_txt_file = f"{results_dir}/results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

with open(f"{results_txt_file}.txt", "a") as fi:
    fi.write("Epochs\tLearning_Rate\tAccuracy\tValidation_accuracy\tLoss\tValidation_loss\n")

# create the base pre-trained model
base_model = HyperResNet(input_shape=(224, 224, 3), classes=8)
# plot_model(base_model, to_file=f'{results_dir}/hyperRes_base.png', show_shapes=True, show_layer_names=True)
# base_model.summary()

#hypers
tuner = kt.Hyperband(base_model,
    objective='val_accuracy',
    max_epochs=HYPERBAND_EPOCHS,
    factor=HYPERBAND_FACTOR,
    directory=f'hyperbandData/E{HYPERBAND_EPOCHS}_F{HYPERBAND_FACTOR}',
    project_name=F'hyperparam_opt_E{HYPERBAND_EPOCHS}_F{HYPERBAND_FACTOR}'
)

# tuner = kt.RandomSearch(base_model,
#     objective="val_accuracy",
#     max_trials=15
# )

# x = base_model.output
# intermediate = 'inter'
# x = Dense(8, activation='softmax',name=intermediate)(x)


model = base_model
print("\n###########################################\n")
tuner.search_space_summary()
print("\n###########################################\n")

#Freezing layers
#for layer in base_model.layers:
#    layer.trainable = False

#Unfreezeing layers
#for idx in range(-2,end,-1):
#  base_model.layers[idx].trainable=True

# new_opt = optimizers.Adadelta(learning_rate= LR)
# model.compile(loss='categorical_crossentropy',optimizer=new_opt, metrics=['accuracy'])


#preprocessing_function=preprocess_input,
datagen = ImageDataGenerator(featurewise_center=False,
samplewise_center=False,
featurewise_std_normalization=False,
samplewise_std_normalization=False,
preprocessing_function=preprocess_input,
rotation_range=0.,
width_shift_range=0.,
height_shift_range=0.,
shear_range=0.,
zoom_range=0.,
channel_shift_range=0.,
fill_mode='nearest',
cval=0.,
horizontal_flip=False,
vertical_flip=False,
rescale=None)

train_generator = datagen.flow_from_directory(train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
    class_mode='categorical')

test_generator = datagen.flow_from_directory(test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
    class_mode='categorical')

validation_generator = datagen.flow_from_directory(val_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
    class_mode='categorical')

tuner.search(train_generator, validation_data = (test_generator),  epochs=10, callbacks=[ClearTrainingOutput()] )

best_hyperparams = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete""")

print("\n###########################################")
print("\n###########################################\n")
tuner.results_summary()
print("\n###########################################")
print("\n###########################################\n")

model = tuner.hypermodel.build(best_hyperparams)

history = model.fit(train_generator,
          validation_data=validation_generator, epochs=100)
result = model.evaluate(validation_generator)
print(result)

#saving model
model.save(f'{results_dir}/hyper_model_{EXPERIMENT_NAME}.h5')
plot_model(model, to_file=f'{results_dir}/hyper_{EXPERIMENT_NAME}.png', show_shapes=True, show_layer_names=True)




accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']
loss = history.history['loss']
validation_loss = history.history['val_loss']

plt.plot(accuracy)
plt.plot(validation_accuracy)
plt.title(f'Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(f'{results_dir}/acc.jpg')
plt.close()

# summarize history for loss
plt.plot(loss)
plt.plot(validation_loss)
plt.title(f'Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(f'{results_dir}/loss.jpg')
plt.close()

with open(f"{results_txt_file}_summarized.txt", "a") as fi:
    fi.write(f'{EPOCHS}\t{LR}\t{accuracy[-1]}\t{validation_accuracy[-1]}\t{loss[-1]}\t{validation_loss[-1]}\n')

with open(f"{results_txt_file}_raw.txt", "a") as fi:
    fi.write(f'accuracy\tvalidation_accuracy\tloss\tvalidation_loss\n')
    for a, va, l, vl in zip(accuracy, validation_accuracy, loss, validation_loss):
        fi.write(f'{a}\t{va}\t{l}\t{vl}\n')





























# list all data in history

'''
# summarize history for accuracy
print(history.history.keys())

accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']
loss = history.history['loss']
validation_loss = history.history['val_loss']
LR_results_structured = [accuracy,  validation_accuracy, loss, validation_loss]
LR_results_dict[f'{LR}'] = LR_results_structured
print(LR_results_dict)        


with open(f"{results_txt_file}.txt", "a") as fi:
    fi.write(f'{EPOCHS}\t{LR}\t{accuracy[-1]}\t{validation_accuracy[-1]}\t{loss[-1]}\t{validation_loss[-1]}\n')

with open(f"{results_txt_file}_raw.txt", "a") as fi:
    fi.write(f'accuracy\tvalidation_accuracy\tloss\tvalidation_loss\n')
    for a, va, l, vl in zip(accuracy, validation_accuracy, loss, validation_loss):
        fi.write(f'{a}\t{va}\t{l}\t{vl}\n')
    
plt.plot(accuracy)
plt.plot(validation_accuracy)
plt.title(f'Learning_rate = {LR} accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(f'{results_dir}/acc_{LR}.jpg')
plt.close()

# summarize history for loss
plt.plot(loss)
plt.plot(validation_loss)
plt.title(f'Learning_rate = {LR} model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(f'{results_dir}/loss_{LR}.jpg')
plt.close()
    

for tmpLR in LR_list:
    plt.plot(LR_results_dict[f'{tmpLR}'][0])
    plt.plot(LR_results_dict[f'{tmpLR}'][1])

plt.title(f'{EPOCHS} Epochs Accuracy Aggregate')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_0.1', 'validation_0.1', 'train_0.01', 'validation_0.01',
            'train_0.001',  'validation_0.001'], loc='upper left')
plt.savefig(f'hyperparams/graph_{EPOCHS}.jpg')
plt.close()

'''
