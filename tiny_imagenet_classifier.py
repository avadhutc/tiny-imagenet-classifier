'''Trains a convnet on the tiny imagenet dataset

'''

# System
import numpy as np
from PIL import Image
np.random.seed(1337)  # for reproducibility

#Keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.datasets import cifar10
from keras.regularizers import WeightRegularizer, ActivityRegularizer 
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization 
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from plotter import Plotter
from keras.utils.visualize_util import plot

#Custom
from load_images import load_images

#Params
# loss_functions = ['hinge', 'squared_hinge','categorical_crossentropy']
loss_functions = ['categorical_crossentropy']
num_classes = 2
batch_size = 32
nb_epoch = 100
data_augmentation = True


#Load images
path='./tiny-imagenet-200'
X_train,y_train,X_test,y_test=load_images(path,num_classes)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

num_samples=round(len(X_train)*0.8)

# input image dimensions
num_channels , img_rows, img_cols = X_train.shape[1], X_train.shape[2], X_train.shape[3]

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test = np_utils.to_categorical(y_test, num_classes)

#Define model
model = Sequential()

#conv-spatial batch norm - relu #1 
model.add(ZeroPadding2D((2,2),input_shape=(3,64,64)))
model.add(Convolution2D(64,5,5,subsample=(2,2),W_regularizer=WeightRegularizer(l1=1e-7,l2=1e-7)))
model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
model.add(Activation('relu'))
print "added conv1"

#conv-spatial batch norm - relu #2
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64,3,3,subsample=(1,1)))
model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
model.add(Activation('relu')) 
print "added conv2"

#conv-spatial batch norm - relu #3
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128,3,3,subsample=(2,2)))
model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
model.add(Activation('relu')) 
model.add(Dropout(0.25)) 
print "added conv3" 

#conv-spatial batch norm - relu #4
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128,3,3,subsample=(1,1)))
model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
model.add(Activation('relu')) 
print "added conv4" 

#conv-spatial batch norm - relu #5
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256,3,3,subsample=(2,2)))
model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
model.add(Activation('relu')) 
print "added conv5" 

#conv-spatial batch norm - relu #6
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256,3,3,subsample=(1,1)))
model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
model.add(Activation('relu')) 
model.add(Dropout(0.25))
print "added conv6" 

#conv-spatial batch norm - relu #7
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512,3,3,subsample=(2,2)))
model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
model.add(Activation('relu')) 
print "added conv7" 

#conv-spatial batch norm - relu #8
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512,3,3,subsample=(1,1)))
model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
model.add(Activation('relu')) 
print "added conv8" 


#conv-spatial batch norm - relu #9
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(1024,3,3,subsample=(2,2)))
model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
model.add(Activation('relu'))
print "added conv9" 
model.add(Dropout(0.25)) 

#Affine-spatial batch norm -relu #10 
model.add(Flatten())
model.add(Dense(512,W_regularizer=WeightRegularizer(l1=1e-5,l2=1e-5)))
model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
model.add(Activation('relu')) 
print "added affine!" 
model.add(Dropout(0.5))

#SGD params
sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True)

for loss_function in loss_functions:
    # for num_classes in num_classes_arr: # num classes loop
    

    print()
    print()
    print('===========================')
    print('Testing: ' + loss_function + ' with ' + str(num_classes) + ' classes')
    print('===========================')
    print()

    if loss_function=='categorical_crossentropy':
        #affine layer w/ softmax activation added 
        model.add(Dense(200,activation='softmax',W_regularizer=WeightRegularizer(l1=1e-5,l2=1e-5)))#pretrained weights assume only 100 outputs, we need to train this layer from scratch
        print "added final affine"

    model.summary()
    plot(model, to_file='model_'+loss_function+'_.png')
    
    model.compile(loss=loss_function,
                  optimizer=sgd,
                  metrics=['accuracy'])


    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=True,  # apply ZCA whitening
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images


    fpath = 'loss-' + loss_function + '-' + str(num_classes)
    datagen.fit(X_train)

    df=datagen.flow(X_train, Y_train, batch_size=batch_size, shuffle=True)
    
    model.fit_generator(df,
                samples_per_epoch=num_samples, nb_epoch=nb_epoch,
                verbose=1, validation_data=df,nb_val_samples=len(X_train)-num_samples,
                callbacks=[Plotter(show_regressions=False, save_to_filepath=fpath, show_plot_window=False)])

    # model.fit(X_train, Y_train, batch_size=64, nb_epoch=nb_epoch,
    #           verbose=1, validation_data=(X_test, Y_test),
    #           callbacks=[Plotter(show_regressions=False, save_to_filepath=fpath, show_plot_window=False)])



    score = model.evaluate(X_test, Y_test, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

