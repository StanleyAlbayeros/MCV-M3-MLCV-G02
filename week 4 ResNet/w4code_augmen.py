
from keras.applications.resnet50 import ResNet50 as RN50
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

end='activation_22'

#end='activation_'+str(idx)
BOARD_PATH = '/home/group02/working/week4/outs/'
EXPERIMENT_NAME = f'base_with_400_cut_{end}_training_30epoch_01_flip_norot'

train_data_dir='/home/group02/datasets/MIT_split/train_400'
val_data_dir='/home/mcv/datasets/MIT_split/test'
test_data_dir='/home/mcv/datasets/MIT_split/test'
img_width = 224
img_height=224
batch_size=32
number_of_epoch=30
validation_samples=807


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
    
# create the base pre-trained model
base_model = RN50(weights='imagenet')
plot_model(base_model, to_file='../week4/models/RN50_base.png', show_shapes=True, show_layer_names=True)

#cropping the model

x = base_model.get_layer(end).output
x = GlobalAveragePooling2D()(x)
x = Dense(8, activation='softmax',name='predictions')(x)

model = Model(input=base_model.input, output=x)

plot_model(model, to_file=f'../week4/models/modelRN50_{EXPERIMENT_NAME}.png', show_shapes=True, show_layer_names=True)

#Freezing layers
#for layer in base_model.layers:
#    layer.trainable = False

#Unfreezeing layers
#for idx in range(-2,end,-1):
#  base_model.layers[idx].trainable=True
    
    
model.compile(loss='categorical_crossentropy',optimizer='adadelta', metrics=['accuracy'])
for layer in model.layers:
    print(layer.name, layer.trainable)

#preprocessing_function=preprocess_input,
datagen = ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
	  preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=.1,
    height_shift_range=.1,
    shear_range=.1,
    zoom_range=.1,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=True,
    rescale=None)

train_generator = datagen.flow_from_directory(train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

test_generator = datagen.flow_from_directory(test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = datagen.flow_from_directory(val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

tbCallBack = TensorBoard(log_dir=BOARD_PATH+EXPERIMENT_NAME+"_augmen", histogram_freq=0, write_graph=True)
history=model.fit_generator(train_generator,
        steps_per_epoch=(int(400//batch_size)+1),
        nb_epoch=number_of_epoch,
        validation_data=validation_generator,
        validation_steps= (int(validation_samples//batch_size)+1), callbacks=[tbCallBack])


result = model.evaluate_generator(test_generator, val_samples=validation_samples)
print( result)

#saving model
model.save(f'../week4/models/modelRN50_{EXPERIMENT_NAME}_augmen.h5')

# list all data in history

if False:
  # summarize history for accuracy
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.savefig('accuracy.jpg')
  plt.close()
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.savefig('loss.jpg')
