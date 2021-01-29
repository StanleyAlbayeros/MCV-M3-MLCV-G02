
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

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims

end='activation_37'

#end='activation_'+str(idx)
BOARD_PATH = '/home/group02/working/week4/outs/'
EXPERIMENT_NAME = f'base_with_400_cut_{end}_training_50epoch'

train_data_dir='/home/group02/datasets/MIT_split/train_400'
val_data_dir='/home/mcv/datasets/MIT_split/test'
test_data_dir='/home/mcv/datasets/MIT_split/test'
img_width = 224
img_height=224
batch_size=32
number_of_epoch=50
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

model = Model(input=base_model.input, output=base_model.layers[2].output)
model.summary()

dir_to_image=train_data_dir+"/"+"coast"+"/"+"arnat59.jpg"
img = load_img(dir_to_image,target_size=(224,224))
img=img_to_array(img)
img = expand_dims(img, axis=0)
# prepare the image (e.g. scale pixel values for the vgg)
img = preprocess_input(img)
# get feature map for first hidden layer
feature_maps = model.predict(img)
# plot all 64 maps in an 8x8 squares
square = 8
ix = 1

for _ in range(square):
	for _ in range(square):
		# specify subplot and turn of axis
		ax = pyplot.subplot(square, square, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
		ix += 1
# show the figure
pyplot.savefig(f'/home/group02/working/week4/layers_viz/'+img_name)

