import os
import zipfile
import logging

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Train:
    def __init__(self):
        #self.force_download = force_download
        #self.data_path = os.path.join(os.getcwd(), '/')
        #self.model_path = os.path.join(os.getcwd(), 'classifier/models/')
        self.dataset_name = 'paultimothymooney/chest-xray-pneumonia'
        self.model: Sequential
        self.val: ImageDataGenerator
        self.train: ImageDataGenerator
        self.test: ImageDataGenerator
        self.epochs = 5

    def load_data(self):
        train_generator = ImageDataGenerator(rescale=1/255.0)
        test_generator = ImageDataGenerator(rescale=1/255.0)
        val_generator = ImageDataGenerator(rescale=1/255.0)
        self.train = train_generator.flow_from_directory('chest_xray/train',
                                                         target_size=(64, 64),
                                                         batch_size=32,
                                                         color_mode='grayscale',
                                                         class_mode='binary')
        self.test = test_generator.flow_from_directory('chest_xray/test',
                                                         target_size=(64, 64),
                                                         batch_size=32,
                                                         color_mode='grayscale',
                                                         class_mode='binary')

        self.val = val_generator.flow_from_directory('chest_xray/val',
                                                     target_size=(64, 64),
                                                     batch_size=32,
                                                     color_mode='grayscale',
                                                     class_mode='binary')
        logger.info('{} images in training set; {} are PNEUMONIA'.format(self.train.samples, sum(self.train.labels)))

    def define_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
        model.add(MaxPooling2D(2, 2))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self):
        self.load_data()
        steps_per_epoch = self.train.samples // self.train.batch_size
        validation_steps = self.val.samples // self.val.batch_size
        self.model = self.define_model()
        self.model.fit_generator(self.train,
                                 epochs=self.epochs,
                                 validation_data=self.val,
                                 steps_per_epoch=steps_per_epoch,
                                 validation_steps=validation_steps)

    def deploy_model(self):
        self.train_model()
        loss, acc = self.model.evaluate_generator(self.test,
                                                  steps=self.test.samples // self.test.batch_size)
        logger.info('Model has been trained with loss, accuracy of {}, {}'.format(loss, acc))
        self.model.save_weights('weights.h5')
        logger.info('Model weights have been saved')


if __name__ == '__main__':
    train_model = Train()
    train_model.deploy_model()
