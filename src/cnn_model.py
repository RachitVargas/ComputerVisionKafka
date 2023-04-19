import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from PIL import Image, ImageFilter

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import re
import cv2
import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import LeakyReLU
from urllib.request import Request, urlopen
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

os.chdir(path="/Users/antony.vargasulead.ac.cr/BigData/proyecto_final")


labels = []
imagenes = []

for carpeta in next(os.walk('data/piedrapapeltijera'))[1]:
  for nombrearchivo in next(os.walk('data/piedrapapeltijera' + '/' + carpeta))[2]:
    if re.search("\\.(jpg|jpeg|png|bmp|tiff)$", nombrearchivo):
      try:
        img = cv2.imread('data/piedrapapeltijera' + '/' + carpeta + '/' + nombrearchivo)[...,::-1]
        img = cv2.resize(img, (64, 64))
        imagenes.append(img)
        labels.append(carpeta)
      except:
        print("No se pudo cargar la imagen: " + nombrearchivo + " en la carpeta: " + carpeta)

X = np.array(imagenes, dtype = np.float32)
y = np.array(labels)

print('Total de individuos: ', len(X),
  '\nNúmero total de salidas: ', len(np.unique(y)), 
  '\nClases de salida: ', np.unique(y))


# Dividir en train y test
train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size = 0.10)

# Normalizar
train_X = train_X / 255.
test_X  = test_X / 255.

# Cambiamos las etiquetas de categoricas a one-hot encoding
train_Y_one_hot = pd.get_dummies(train_Y)
train_Y_one_hot = train_Y_one_hot.to_numpy()

test_Y_one_hot = pd.get_dummies(test_Y)
test_Y_one_hot = test_Y_one_hot.to_numpy()


epochs = 50
batch_size = 64
 
modelo_ppt = Sequential()

modelo_ppt.add(Conv2D(64, kernel_size=(3, 3),
                       activation='linear',padding='same',input_shape=(64,64,3)))
modelo_ppt.add(LeakyReLU(alpha=0.1))
modelo_ppt.add(MaxPooling2D((2, 2),padding='same'))
modelo_ppt.add(Dropout(0.5))
modelo_ppt.add(Conv2D(64, kernel_size=(2, 2),
                      activation='linear',padding='same'))
modelo_ppt.add(MaxPooling2D((2, 2),padding='same'))
modelo_ppt.add(Flatten())
modelo_ppt.add(Dense(32, activation='linear'))
modelo_ppt.add(LeakyReLU(alpha=0.1))
modelo_ppt.add(Dropout(0.5))

modelo_ppt.add(Dense(len(np.unique(y)), activation='softmax'))
 
modelo_ppt.summary()


INIT_LR = 1e-3
modelo_ppt.compile(loss=keras.losses.categorical_crossentropy, 
                    optimizer=keras.optimizers.Adam(lr=INIT_LR),
                    metrics=['accuracy'])


#Entrenamos el modelo
modelo_ppt.fit(train_X, train_Y_one_hot, batch_size = batch_size,
                epochs = epochs, verbose = 0)
    
# Guardamos la red, para reutilizarla en el futuro, sin tener que volver a entrenar
modelo_ppt.save("algorithms/modelCNN.h5")

from tensorflow.keras.models import load_model
modelo_ppt = load_model('algorithms/modelCNN.h5')

#Evaluamos con respecto a la tabla de testing
test_eval = modelo_ppt.evaluate(test_X, test_Y_one_hot, verbose=0)
 
print('\nTest loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

