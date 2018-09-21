import os
import glob
import numpy as np
from skimage.io import imread
from tensorflow import keras


def get_data():
    """
    Loads a training and validation set. This is just a toy dataset for illustration.
    The output values are heatmaps with the same size as the input images and a
    single channel. Input and output values are both in [0,1] and float32.
    """
    # Load the data
    images_dir = 'data/images'
    heatmaps_dir = 'data/heatmaps'
    X = []
    y = []
    img_paths = sorted(glob.glob(os.path.join(images_dir, '*.jpg')))
    for path in img_paths:
        img = imread(path)
        img = img.astype('float32') / 255
        X.append(img)
        heatmap_path = os.path.join(heatmaps_dir, os.path.basename(path))
        heatmap = imread(heatmap_path)
        heatmap = heatmap.astype('float32') / 255
        y.append(heatmap[:,:,None])
    X = np.stack(X)
    y = np.stack(y)
    print('y stats (min, max, mean)')
    print(y.min())
    print(y.max())
    print(y.mean())

    # Split into train and validation
    p_train = 0.8
    n_train = int(X.shape[0]*p_train)
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train:]
    y_val = y[n_train:]

    return X_train, y_train, X_val, y_val


def get_model():
    """
    Create a very simple fully convolutional model.
    """
    input_layer = keras.Input(shape=(128,128,3), name='input')
    x = input_layer
    x = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    heatmap = keras.layers.Conv2D(1, 3, activation='sigmoid', padding='same', name='Output')(x)
    model = keras.Model(inputs=input_layer, outputs=heatmap)
    model.compile(optimizer=keras.optimizers.RMSprop(), loss='mse')
    model.summary()
    return model


def get_callbacks():
    """
    A single callback to save the best model during training.
    """
    model_dir = 'models/h5'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, 'best_model.h5')

    callbacks = [keras.callbacks.ModelCheckpoint(model_path,
                                                 save_best_only=True,
                                                 save_weights_only=False,
                                                 verbose=1)]
    return callbacks, model_path



# Get the data and model and run training for a few epochs
X_train, y_train, X_val, y_val = get_data()
model = get_model()
callbacks, model_path = get_callbacks()
model.fit(X_train,
          y_train,
          batch_size=2,
          epochs=10,
          validation_data=(X_val,y_val),
          callbacks=callbacks)


