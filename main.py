from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import argparse

img_width, img_height = 150, 150

train_dir = 'data/train'
val_dir = 'data/test'
epochs = 25
batch_size = 2
model_save_dir = 'model'

train_sample_num = 10
val_sample_num = 90

def main(args):
    global img_width
    if args.img_width is not None:
        img_width = args.img_width
    
    global img_height
    if args.img_height is not None:
        img_height = args.img_height
    
    global train_dir
    if args.train_dir is not None:
        train_dir = args.train_dir
    
    global val_dir
    if args.val_dir is not None:
        val_dir = args.val_dir
    
    global epochs
    if args.epochs is not None:
        epochs = args.epochs
    
    global batch_size
    if args.batch_size is not None:
        batch_size = args.batch_size
    
    global model_save_dir
    if args.model_save_dir is not None:
        model_save_dir = args.model_save_dir
    
    global train_sample_num
    if args.train_sample_num is not None:
        train_sample_num = args.train_sample_num
        
    global val_sample_num
    if args.val_sample_num is not None:
        val_sample_num = args.val_sample_num
    
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    
    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    
    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=1,
        save_to_dir='/Users/yiming/Downloads/trainsave',
        class_mode='binary')
    
    validation_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(img_width, img_height),
        batch_size=1,
        save_to_dir='/Users/yiming/Downloads/valsave',
        class_mode='binary')
    
    model.fit_generator(
        train_generator,
        steps_per_epoch=10,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=10)
    
    model.save_weights(model_save_dir)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_width")
    parser.add_argument("--img_height")
    parser.add_argument("--train_dir")
    parser.add_argument("--val_dir")
    parser.add_argument("--model_save_dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--train_sample_num", type=int)
    parser.add_argument("--val_sample_num", type=int)
    args = parser.parse_args()
    
    main(args)
    # test
    