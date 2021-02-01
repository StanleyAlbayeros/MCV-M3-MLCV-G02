from keras.applications.resnet50 import ResNet50 as RN50
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
import matplotlib.pyplot as plt
import os

end='activation_37'

#end='activation_'+str(idx)
BOARD_PATH = 'boards/'
EXPERIMENT_NAME = f'training_50epoch_LRFull'
MODEL_FNAME = f'models/modelRN50_{EXPERIMENT_NAME}.h5'
EPOCH_ARR=[50, 100, 200]

train_data_dir='../datasets/MIT_split/train'
val_data_dir='../datasets/MIT_split/test'
test_data_dir='../datasets/MIT_split/test'

img_width = 224
img_height=224
batch_size=32
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

LR_list = [0.1, 0.01, 0.0001]
# LR_list = [0.1]
LR_results_dict = {}
d = {}
for EPOCHS in EPOCH_ARR:
    for LR in LR_list:     
        results_dir=f'learningRateDiffs/epochs_{EPOCHS}_LR_{LR}'
        results_txt_file = f"{results_dir}/results_{EPOCHS}_LR_{LR}"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        with open(f"{results_txt_file}.txt", "a") as fi:
            fi.write("Epochs\tLearning_Rate\tAccuracy\tValidation_accuracy\tLoss\tValidation_loss\n")

        # create the base pre-trained model
        base_model = RN50(weights='imagenet')
        plot_model(base_model, to_file=f'{results_dir}/RN50_base.png', show_shapes=True, show_layer_names=True)
        # base_model.summary()

        #cropping the model

        x = base_model.layers[-2].output
        intermediate = 'inter'
        x = Dense(8, activation='softmax',name=intermediate)(x)

        model = Model(base_model.input, x)

        plot_model(model, to_file=f'{results_dir}/modelRN50_{EXPERIMENT_NAME}.png', show_shapes=True, show_layer_names=True)

        #Freezing layers
        #for layer in base_model.layers:
        #    layer.trainable = False

        #Unfreezeing layers
        #for idx in range(-2,end,-1):
        #  base_model.layers[idx].trainable=True
            
        new_opt = optimizers.Adadelta(learning_rate= LR)
        model.compile(loss='categorical_crossentropy',optimizer=new_opt, metrics=['accuracy'])
        for layer in model.layers:
            print(layer.name, layer.trainable)

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
                class_mode='categorical')

        test_generator = datagen.flow_from_directory(test_data_dir,
                target_size=(img_width, img_height),
                batch_size=batch_size,
                class_mode='categorical')

        validation_generator = datagen.flow_from_directory(val_data_dir,
                target_size=(img_width, img_height),
                batch_size=batch_size,
                class_mode='categorical')

        tbCallBack = TensorBoard(log_dir=BOARD_PATH+EXPERIMENT_NAME, histogram_freq=0, write_graph=True)
        history=model.fit_generator(train_generator,
                steps_per_epoch=(int(1881//batch_size)+1),
                epochs=EPOCHS,
                validation_data=validation_generator,
                validation_steps= (int(validation_samples//batch_size)+1), callbacks=[tbCallBack])


        result = model.evaluate_generator(test_generator, validation_samples)
        print( result)

        #saving model
        model.save(f'{results_dir}/modelRN50_{EXPERIMENT_NAME}.h5')

        # list all data in history

        if True:
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
        
        backend.clear_session()
    
    for tmpLR in LR_list:
        plt.plot(LR_results_dict[f'{tmpLR}'][0])
        plt.plot(LR_results_dict[f'{tmpLR}'][1])

    plt.title(f'{EPOCHS} Epochs Accuracy Aggregate')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_0.1', 'validation_0.1', 'train_0.01', 'validation_0.01',
                'train_0.001',  'validation_0.001'], loc='upper left')
    plt.savefig(f'learningRateDiffs/graph_{EPOCHS}.jpg')
    plt.close()

