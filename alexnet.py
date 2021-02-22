from __future__ import print_function
import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D, Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
import numpy as np
from tfdeterminism import patch
from tensorflow.python.keras.backend import set_session
from matplotlib import pyplot as plt

DETERMINISTIC = False

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if DETERMINISTIC:
	os.environ['TF_DETERMINISTIC_OPS'] = '1'
	os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
	SEED = 123
	np.random.seed(SEED)
	tf.random.set_seed(SEED)
	os.environ['PYTHONHASHSEED'] = str(SEED)


class alexnet:
    def __init__(self, n=0):
        self.num_classes = 10
        self.weight_decay = 0.0005
        self.x_shape = [32,32,3]
        self.n = n
        if DETERMINISTIC:
        	self.model = self.build_deterministic_model()
        else:
        	self.model = self.build_basic_model()
        self.model = self.train(self.model)
        


    def build_deterministic_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

        model = Sequential()

        #Layer 1 
        model.add( Conv2D(48, kernel_size=(3,3),strides=(1,1), activation='relu', padding='same', input_shape=x_train.shape[1:], kernel_initializer=glorot_normal(seed=SEED)) )
        model.add( MaxPool2D(pool_size=(2,2),strides=(2,2)) )

        #Layer 2
        model.add( Conv2D(96, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer=glorot_normal(seed=SEED)) )
        model.add( MaxPool2D(pool_size=(2,2),strides=(2,2)) )

        #Layer 3
        model.add( Conv2D(192, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer=glorot_normal(seed=SEED)) )


        #Layer 4
        model.add( Conv2D(192, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer=glorot_normal(seed=SEED)) )
        model.add( MaxPool2D(pool_size=(2,2),strides=(2,2)) )

        #Layer 5
        model.add( Conv2D(256, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer=glorot_normal(seed=SEED)) )
        model.add( MaxPool2D(pool_size=(2,2),strides=(2,2)) )

        model.add(Flatten())

        #Layer 6
        model.add(Dense(512, activation='tanh', kernel_initializer=glorot_normal(seed=SEED)))

        #Layer 7 
        model.add(Dense(256, activation='tanh', kernel_initializer=glorot_normal(seed=SEED)))

        #Prediction
        model.add(Dense(10, activation='softmax', kernel_initializer=glorot_normal(seed=SEED)))
        return model

    def build_basic_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

        model = Sequential()

        #Layer 1 
        model.add( Conv2D(48, kernel_size=(3,3),strides=(1,1), activation='relu', padding='same', input_shape=x_train.shape[1:]) )
        model.add( MaxPool2D(pool_size=(2,2),strides=(2,2)) )

        #Layer 2
        model.add( Conv2D(96, kernel_size=(3,3), activation='relu', padding='same') )
        model.add( MaxPool2D(pool_size=(2,2),strides=(2,2)) )

        #Layer 3
        model.add( Conv2D(192, kernel_size=(3,3), activation='relu', padding='same') )


        #Layer 4
        model.add( Conv2D(192, kernel_size=(3,3), activation='relu', padding='same') )
        model.add( MaxPool2D(pool_size=(2,2),strides=(2,2)) )

        #Layer 5
        model.add( Conv2D(256, kernel_size=(3,3), activation='relu', padding='same') )
        model.add( MaxPool2D(pool_size=(2,2),strides=(2,2)) )

        model.add(Flatten())

        #Layer 6
        model.add(Dense(512, activation='tanh'))

        #Layer 7 
        model.add(Dense(256, activation='tanh'))

        #Prediction
        model.add(Dense(10, activation='softmax'))
        return model


    def normalize(self,X_train,X_test):
        #this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(X_train,axis=(0,1,2,3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)
        return X_train, X_test

    def normalize_production(self,x):
        #this function is used to normalize instances in production according to saved training set statistics
        # Input: X - a training set
        # Output X - a normalized training set according to normalization constants.

        #these values produced during first training and are general for the standard cifar10 training set normalization
        mean = 120.707
        std = 64.15
        return (x-mean)/(std+1e-7)

    def predict(self,x,normalize=True,batch_size=50):
        if normalize:
            x = self.normalize_production(x)
        return self.model.predict(x,batch_size)

    def train(self,model):

        #training parameters
        batch_size = 128
        maxepoches = 300
        learning_rate = 0.1
        lr_decay = 1e-6
        lr_drop = 20
        # The data, shuffled and split between train and test sets:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = self.normalize(x_train, x_test)

        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))
        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

        #data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)



        #optimization details
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


        # training process in a for loop with learning rate drop every 25 epoches.
        if DETERMINISTIC:
        	historytemp = model.fit(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=maxepoches,
                            validation_data=(x_test, y_test),callbacks=[reduce_lr],verbose=2, workers=1, shuffle=False)
        else:
        	historytemp = model.fit(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=maxepoches,
                            validation_data=(x_test, y_test),callbacks=[reduce_lr],verbose=2)
        
        if self.n:
        	n = self.n
	        plt.plot(historytemp.history['accuracy'])
	        plt.plot(historytemp.history['val_accuracy'])
	        plt.ylabel('accuracy')
	        plt.xlabel('epoch')
	        plt.legend(['train', 'val'], loc='upper left')
	        plt.title("non-deterministic model acc")
	        plt.savefig('n_'+str(n)+'_acc')
	        plt.clf()
	        plt.plot(historytemp.history['loss'])
	        plt.plot(historytemp.history['val_loss'])
	        plt.ylabel('loss')
	        plt.xlabel('epoch')
	        plt.title("non-deterministic model loss")
	        plt.legend(['train', 'val'], loc='upper left')
	        plt.savefig('n_'+str(n)+'_loss')
	        plt.clf()
	        model.save_weights('alexnet_n_'+str(n)+'.h5')
        return model

if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    for i in range(1, 6):
	    model = alexnet(n=i)

	    predicted_x = model.predict(x_test)
	    residuals = np.argmax(predicted_x,1)!=np.argmax(y_test,1)

	    loss = sum(residuals)/len(residuals)
	    print("the validation 0/1 loss is: ",loss)

