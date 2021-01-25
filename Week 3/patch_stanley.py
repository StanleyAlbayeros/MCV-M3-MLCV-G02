from __future__ import print_function
from utils import *
from keras import Model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras import backend
import matplotlib.pyplot as plt
import warnings
import _pickle as Pickle
warnings.filterwarnings('ignore')

#user defined variables
TRAIN = True
VERBOSE = 1 # 0 NOTHING, 1 PROGRESS BAR
# PATCH_SIZE  = 32
# BATCH_SIZE  = 128
# EPOCHS = 50

LOCAL = 1
REMOTE = 0
ENV = LOCAL ## LOCAL, REMOTE (UAB CLUSTER) 

with open("results.txt", "a") as fi:
    fi.write("Patches\tBatches\tEpochs\tTest Accuracy\n")
    
with open("results_per_class.txt", "a") as fi:
    fi.write("Patches\tBatches\tEpochs\t{class_name} accuracy\n")
    

for EPOCHS in [50,100]:
  for PATCH_SIZE in [32,64,128]:
    for BATCH_SIZE in [32,64,128]:

      MAX_PATCHES = (256//PATCH_SIZE)**2
      STEPS_PER_EPOCH_HELPER = 1881 * MAX_PATCHES
      VERSION = f'P{PATCH_SIZE}_B{BATCH_SIZE}_E{EPOCHS}'

      DATASET_DIR = '../mcv/datasets/MIT_split'
      PATCHES_DIR = f'../mcv/datasets/Patches_{VERSION}'
      MODEL_FNAME = f'../mcv/datasets/models/{VERSION}.h5'

      if ENV == REMOTE:
        DATASET_DIR = '../../mcv/datasets/MIT_split'
        PATCHES_DIR = f'../Patches_{VERSION}'
        MODEL_FNAME = f'../output_results/{VERSION}.h5'



      def build_mlp(input_size=PATCH_SIZE,phase='TRAIN'):
        model = Sequential()
        model.add(Reshape((input_size*input_size*3,),input_shape=(input_size, input_size, 3)))
        model.add(Dense(units=512, activation='relu'))
        model.add(Dense(units=256, activation='relu'))
        model.add(Dense(units=128, activation='relu'))
        model.add(Dense(units=64, activation='relu'))
        if phase=='TEST':
          model.add(Dense(units=8, activation='linear'))
        else:
          model.add(Dense(units=8, activation='softmax'))
        return model

      if not os.path.exists(DATASET_DIR):
        colorprint(Color.RED, 'ERROR: dataset directory '+DATASET_DIR+' do not exists!\n')
        quit()
      if not os.path.exists(PATCHES_DIR):
        colorprint(Color.YELLOW, 'WARNING: patches dataset directory '+PATCHES_DIR+' does not exist!\n')
        colorprint(Color.BLUE, 'Creating image patches dataset into '+PATCHES_DIR+'\n')
        generate_image_patches_db(DATASET_DIR,PATCHES_DIR, patch_size=PATCH_SIZE)
        colorprint(Color.BLUE, 'Done!\n')


      colorprint(Color.BLUE, 'Building MLP model...\n')

      model = build_mlp(input_size=PATCH_SIZE)

      model.compile(loss='categorical_crossentropy',
                    optimizer='sgd',
                    metrics=['accuracy'])

      print(model.summary())

      colorprint(Color.BLUE, 'Done!\n')

      if not os.path.exists(MODEL_FNAME):
        colorprint(Color.YELLOW, 'WARNING: model file '+MODEL_FNAME+' do not exists!\n')
        colorprint(Color.BLUE, 'Start training...\n')
        # this is the dataset configuration we will use for training
        # only rescaling
        train_datagen = ImageDataGenerator(
                rescale=1./255,
                horizontal_flip=True)
        
        # this is the dataset configuration we will use for testing:
        # only rescaling
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # this is a generator that will read pictures found in
        # subfolers of 'data/train', and indefinitely generate
        # batches of augmented image data
        train_generator = train_datagen.flow_from_directory(
                PATCHES_DIR+'/train',  # this is the target directory
                target_size=(PATCH_SIZE, PATCH_SIZE),  # all images will be resized to PATCH_SIZExPATCH_SIZE
                batch_size=BATCH_SIZE,
                classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
                class_mode='categorical')  # since we use binary_crossentropy loss, we need categorical labels
        
        # this is a similar generator, for validation data
        validation_generator = test_datagen.flow_from_directory(
                PATCHES_DIR+'/train',
                target_size=(PATCH_SIZE, PATCH_SIZE),
                batch_size=BATCH_SIZE,
                classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
                class_mode='categorical')
        
        model.fit_generator(
                train_generator,
                steps_per_epoch=STEPS_PER_EPOCH_HELPER // BATCH_SIZE, # use correct number of images 1881 * patches_per_image
                epochs=EPOCHS,
                validation_data=validation_generator,
                validation_steps= (807 * MAX_PATCHES) // BATCH_SIZE,
                verbose=VERBOSE) # use correct number of images 807 * patches_per_image
        
        colorprint(Color.BLUE, 'Done!\n')
        colorprint(Color.BLUE, 'Saving the model into '+MODEL_FNAME+' \n')
        model.save_weights(MODEL_FNAME)  # always save your weights after training or during training
        colorprint(Color.BLUE, 'Done!\n')



      colorprint(Color.BLUE, 'Building MLP model for testing...\n')

      model = build_mlp(input_size=PATCH_SIZE, phase='TEST')
      print(model.summary())

      colorprint(Color.BLUE, 'Done!\n')

      colorprint(Color.BLUE, 'Loading weights from '+MODEL_FNAME+' ...\n')
      print ('\n')

      model.load_weights(MODEL_FNAME)

      colorprint(Color.BLUE, 'Done!\n')

      colorprint(Color.BLUE, 'Start evaluation ...\n')
      colorprint(Color.MAGENTA, f'Patches: {PATCH_SIZE}\tBatches: {BATCH_SIZE}\tEpochs: {EPOCHS}...\n')

      directory = DATASET_DIR+'/test'
      classes = {'coast':0,'forest':1,'highway':2,'inside_city':3,'mountain':4,'Opencountry':5,'street':6,'tallbuilding':7}
      correct = 0.
      total   = 807
      count   = 0
      class_correct = 0
      class_count = 0
      class_accu = 0

      for class_dir in os.listdir(directory):
          cls = classes[class_dir]
          for imname in os.listdir(os.path.join(directory,class_dir)):
            im = Image.open(os.path.join(directory,class_dir,imname))
            patches = image.extract_patches_2d(np.array(im), (PATCH_SIZE, PATCH_SIZE), max_patches=MAX_PATCHES)
            out = model.predict(patches/255.)
            predicted_cls = np.argmax( softmax(np.mean(out,axis=0)) )
            if predicted_cls == cls:
              correct += 1
              class_correct += 1
            count += 1
            class_count += 1
            print('Evaluated images: '+str(count)+' / '+str(total), end='\r')    
              
          class_accu = class_correct/class_count
          class_correct = 0
          class_count = 0
          with open("results_per_class.txt", "a") as fi:
              fi.write(f"Patches: {PATCH_SIZE}\tBatches: {BATCH_SIZE}\tEpochs: {EPOCHS}\t{class_dir} accuracy: {class_accu:.4f}\n")
      
      with open("results_per_class.txt", "a") as fi:
          fi.write(f"\n\n")

      colorprint(Color.BLUE, 'Done!\n')
      accu = correct/total
      colorprint(Color.GREEN, 'Test Acc. = '+str(accu)+'\n')
      with open("results.txt", "a") as fi:
          fi.write(f"{PATCH_SIZE}\t{BATCH_SIZE}\t{EPOCHS}\t{accu:.4f}\n")

      backend.clear_session()

