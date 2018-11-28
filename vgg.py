from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.applications.vgg16 import VGG16
import argparse

img_width, img_height = 224, 224

train_dir = 'data_expanded/train'
val_dir = 'data_expanded/validation'
epochs = 10
batch_size = 20
model_save_dir = 'jjjjj'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_width")
    parser.add_argument("--img_height")
    parser.add_argument("--train_dir")
    parser.add_argument("--val_dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--model_save_dir")
    
    args = parser.parse_args()
    
    if args.img_width is not None:
        img_width = args.img_width
        
    if args.img_height is not None:
        img_height = args.img_height
        
    if args.train_dir is not None:
        train_dir = args.train_dir
        
    if args.val_dir is not None:
        val_dir = args.val_dir
        
    if args.epochs is not None:
        epochs = args.epochs
        
    if args.batch_size is not None:
        batch_size = args.batch_size
        
    if args.model_save_dir is not None:
        model_save_dir = args.model_save_dir
    
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)
    
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')
    
    validation_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')
    
    
    model_vgg = VGG16(input_shape=input_shape, include_top=False)
    
    model = Sequential()
    
    model.add(model_vgg)
    top_model = Sequential()
    top_model.add(Flatten(input_shape=model_vgg.output_shape[1:]))
    top_model.add(Dense(256))
    top_model.add(Activation('relu'))
    top_model.add(Dropout(0.8))
    top_model.add(Dense(2))
    top_model.add(Activation('softmax'))
    
    model.add(top_model)
    
    model.layers[0].trainable = False
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    
    model.fit_generator(
            train_generator,
            steps_per_epoch=500 // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=200 // batch_size)
    
    top_model.save_weights(model_save_dir)
    
    
    print("The result of evaluation on test/validation dataset")
    model.evaluate_generator(validation_generator)