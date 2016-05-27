'''Trains a convnet on the tiny imagenet dataset

0.4 val acc after 20 epochs with 100 classes
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
# from keras.utils.visualize_util import plot
import h5py

#Custom
from load_images import load_images

#Params
loss_functions = ['categorical_crossentropy','squared_hinge','hinge']
num_classes = 200
batch_size = 32
nb_epoch = 30

#Load images
path='./tiny-imagenet-200'
X_train,y_train,X_test,y_test=load_images(path,num_classes)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

num_samples=len(X_train)

# input image dimensions
num_channels , img_rows, img_cols = X_train.shape[1], X_train.shape[2], X_train.shape[3]

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test = np_utils.to_categorical(y_test, num_classes)

for loss_function in loss_functions:
    # for num_classes in num_classes_arr: # num classes loop
    
    model = Sequential()
    #conv-spatial batch norm - relu #1 
    model.add(ZeroPadding2D((2,2),input_shape=(3,64,64)))
    model.add(Convolution2D(64,5,5,subsample=(2,2),W_regularizer=WeightRegularizer(l1=1e-7,l2=1e-7)))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    model.add(Activation('relu')) 

    #conv-spatial batch norm - relu #2
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64,3,3,subsample=(1,1)))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    model.add(Activation('relu')) 

    #conv-spatial batch norm - relu #3
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128,3,3,subsample=(2,2)))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    model.add(Activation('relu')) 
    model.add(Dropout(0.25)) 

    #conv-spatial batch norm - relu #4
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128,3,3,subsample=(1,1)))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    model.add(Activation('relu')) 

    #conv-spatial batch norm - relu #5
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256,3,3,subsample=(2,2)))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    model.add(Activation('relu')) 

    #conv-spatial batch norm - relu #6
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256,3,3,subsample=(1,1)))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    model.add(Activation('relu')) 
    model.add(Dropout(0.25))

    #conv-spatial batch norm - relu #7
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512,3,3,subsample=(2,2)))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    model.add(Activation('relu')) 

    #conv-spatial batch norm - relu #8
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512,3,3,subsample=(1,1)))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    model.add(Activation('relu')) 
    

    #conv-spatial batch norm - relu #9
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(1024,3,3,subsample=(2,2)))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    model.add(Activation('relu'))
    model.add(Dropout(0.25)) 

    #Affine-spatial batch norm -relu #10 
    model.add(Flatten())
    model.add(Dense(512,W_regularizer=WeightRegularizer(l1=1e-5,l2=1e-5)))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    model.add(Activation('relu')) 
    model.add(Dropout(0.5)) 

    print()
    print()
    print('===========================')
    print('Testing: ' + loss_function + ' with ' + str(num_classes) + ' classes')
    print('===========================')
    print()

    if loss_function=='categorical_crossentropy':
        #affine layer w/ softmax activation added 
        model.add(Dense(num_classes,activation='softmax',W_regularizer=WeightRegularizer(l1=1e-5,l2=1e-5)))
        sgd = SGD(lr=0.05, decay=1e-5, momentum=0.9, nesterov=True)
    else:
        if loss_function=='hinge':
            sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        else:
            sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.add(Dense(num_classes,W_regularizer=WeightRegularizer(l1=1e-5,l2=1e-5)))


    model.compile(loss=loss_function,
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.summary()
    # plot(model, to_file='model_'+loss_function+'_.png')


    # datagen = ImageDataGenerator(
    #     featurewise_center=True,  # set input mean to 0 over the dataset
    #     samplewise_center=False,  # set each sample mean to 0
    #     featurewise_std_normalization=True,  # divide inputs by std of the dataset
    #     samplewise_std_normalization=False,  # divide each input by its std
    #     zca_whitening=False,  # apply ZCA whitening
    #     rotation_range=10.,  # randomly rotate images in the range (degrees, 0 to 180)
    #     width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    #     height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    #     horizontal_flip=True,  # randomly flip images
    #     vertical_flip=False)  # randomly flip images


    fpath = 'loss-' + loss_function + '-' + str(num_classes)
    # datagen.fit(X_train)
    
    # model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size, shuffle=True, save_to_dir='./datagen/', save_prefix='datagen-',save_format='png'), # To save the images created by the generator
                # samples_per_epoch=num_samples, nb_epoch=nb_epoch,
                # verbose=1, validation_data=(X_test,Y_test),
                # callbacks=[Plotter(show_regressions=False, save_to_filepath=fpath, show_plot_window=False)])

    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_data=(X_test, Y_test),
              callbacks=[Plotter(show_regressions=False, save_to_filepath=fpath, show_plot_window=False)])



    score = model.evaluate(X_test, Y_test, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    pathWeights='model'+loss_function+'.h5'
    model.save_weights(pathWeights)
