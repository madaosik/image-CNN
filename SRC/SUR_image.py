from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation
from keras.models import model_from_json
from keras import regularizers

from keras_preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
import numpy as np

import os, shutil, argparse

TRAIN_DIR = os.path.join(os.path.curdir, 'SRC', 'train')
DEV_DIR = os.path.join(os.path.curdir, 'SRC', 'dev')
EVAL_DIR = os.path.join(os.path.curdir, 'SRC', 'eval')
MODELDATA_PATH = os.path.join(os.path.curdir, 'SRC', 'modeldata')
INPUT_SHAPE = (80, 80, 3)
BATCH_SIZE = 8
EPOCHS = 10

def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=INPUT_SHAPE))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64, kernel_regularizer=regularizers.l2(0.03)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model

def training_set_init():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    training_set = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(80, 80),
        batch_size=BATCH_SIZE,
        class_mode='binary',
    )
    return training_set

def dev_set_init():
    dev_datagen = ImageDataGenerator(rescale=1. / 255)

    dev_set = dev_datagen.flow_from_directory(
        DEV_DIR,
        target_size=(80, 80),
        batch_size=BATCH_SIZE,
        class_mode='binary',
    )
    return dev_set

def eval_set_init():
    eval_datagen = ImageDataGenerator(rescale=1. / 255)

    eval_set = eval_datagen.flow_from_directory(
        EVAL_DIR,
        target_size=(80, 80),
        shuffle=False
    )
    return eval_set


def train_model(untrained_model):
    training_set = training_set_init()
    dev_set = dev_set_init()

    STEP_SIZE_TRAIN = training_set.n // training_set.batch_size
    STEP_SIZE_DEV = dev_set.n // dev_set.batch_size


    class_weights = class_weight.compute_class_weight('balanced', np.unique(training_set.classes), training_set.classes)

    untrained_model.fit_generator(generator=training_set,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=dev_set,
                        validation_steps=STEP_SIZE_DEV,
                        class_weight=class_weights,
                        epochs=EPOCHS)

    history = untrained_model.evaluate_generator(generator=dev_set, steps=STEP_SIZE_TRAIN, verbose=1)
    return untrained_model


def save_trained_network(model):
    if os.path.exists(MODELDATA_PATH):
        shutil.rmtree(MODELDATA_PATH)
    model_json = model.to_json()

    os.mkdir(MODELDATA_PATH)
    with open(os.path.join(MODELDATA_PATH, "model.json"), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(os.path.join(MODELDATA_PATH, "model.h5"))

def predict_output(trained_model):
    eval_set = eval_set_init()

    pred = trained_model.predict_generator(eval_set, verbose=1, steps=len(eval_set))
    predicted_class_indices = (pred >= 0.5).astype(np.int)

    predictions = []
    for class_ind in predicted_class_indices.tolist():
        predictions.append(class_ind[0])

    filenames = eval_set.filenames

    with open(os.path.join(os.path.curdir, 'image_CNN.txt'), 'w') as f:
        for i in range(0,len(filenames)):
            basename = os.path.basename(filenames[i])
            rowstring = os.path.splitext(basename)[0] + " " + str('%.5f' % pred[i])
            # Add predictions
            rowstring +=  " " + str(predictions[i]) + "\n"
            f.write(rowstring)


def load_trained_network():
    # load json and create model
    json_file = open(os.path.join(MODELDATA_PATH, 'model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(os.path.join(MODELDATA_PATH, 'model.h5'))
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return loaded_model

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--notrain", help="Use the locally stored and trained CNN for image recognition", action="store_true")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()

    if not args.notrain:
        print("Setting up and training the CNN.....")
        model = create_model()
        trained_model = train_model(model)
        save_trained_network(trained_model)

    print("Loading the trained CNN.....\n")
    trained_model = load_trained_network()
    predict_output(trained_model)