import sys
import pandas as pd
import numpy as np

# keras 2.0.4
# tensorflow 1.1.0
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint

# path
train_file = sys.argv[1] if len(sys.argv) > 1 else './data/mnist_train.csv'
test_file = sys.argv[2] if len(sys.argv) > 2 else './data/mnist_test.csv'
prediction_file = sys.argv[3] if len(sys.argv) > 3 else 'prediction.csv'
best_model_weight_path = './model/mnist_cnn_best.h5'

# input / output
input_shape = [28, 28, 1]
output_dim = 10

def load_data(filename):
    def to_one_hot(digit, cat_num):
        one_hot = [0.0] * cat_num
        one_hot[digit] = 1.0
        return one_hot

    df_train = pd.read_csv(filename, header=None)
    X = df_train.values[:,1:].reshape([-1,28,28,1]) / 255.
    Y = np.array([to_one_hot(digit, output_dim) for digit in df_train.values[:,0]])
    return X, Y

def build_model():
    model = Sequential()

    model.add(Conv2D(64, (5, 5), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=(2,2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(AveragePooling2D(pool_size=(3, 3), padding='same', strides=(2,2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(AveragePooling2D(pool_size=(3, 3), padding='same', strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(output_dim, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

    return model

'''
# load data
train_X, train_Y = load_data(train_file)
valid_X, valid_Y = X[-1000:], Y[-1000:]
train_X, train_Y = X[:-1000], Y[:-1000]

# train
model = build_model()
earlystopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='min')
checkpoint = ModelCheckpoint(
                filepath=best_model_weight_path,
                verbose=1,
                save_best_only=True,
                save_weights_only=True,
                monitor='val_acc',
                mode='max'
            )
datagen = ImageDataGenerator(
        rotation_range=40,
        shear_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,        
        horizontal_flip=True,
        fill_mode='nearest'
    )
model.fit_generator(
        datagen.flow(train_X, train_Y, batch_size=64), 
        #batch_size=128, 
        steps_per_epoch=len(train_X)/64,
        epochs=1000, 
        validation_data=(valid_X, valid_Y), 
        callbacks=[earlystopping, checkpoint]
    )
'''



# predict
test_X, _ = load_data(test_file)
model = build_model()
model.load_weights(best_model_weight_path)
y_preds = model.predict(test_X)
y_preds = np.array([np.argmax(y) for y in y_preds])
df_pred = pd.DataFrame({'id': range(len(y_preds)), 'label': y_preds})
df_pred.to_csv(prediction_file, columns=['id','label'], index=False)



