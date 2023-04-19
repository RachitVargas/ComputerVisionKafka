import streamlit as st
from kafka import KafkaConsumer
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import tensorflow as tf

consumer = KafkaConsumer('my-topic', bootstrap_servers='localhost:9092')
model = load_model('/Users/antony.vargasulead.ac.cr/BigData/proyecto_final/algorithms/modelCNN.h5')


def kafkastream():
    for message in consumer:
        yield np.frombuffer(message.value, np.uint8)

def predict_image(image):
    im = cv2.resize(image, (64, 64))
    im = im.astype('float32')
    im = im / 255.
    im_input = tf.reshape(im, shape = [1, 64, 64, 3])

    predict_proba = sorted(model.predict(im_input)[0])[-1]
    predict_class = np.argmax(model.predict(im_input))

    if predict_class == 0:
        predict_label = 'Papel'
    elif predict_class == 1:
        predict_label = 'Piedra'
    else:
        predict_label = 'Tijera'

    return predict_label, round(predict_proba*100,2)

def main():
    st.title('Computer Vision with Kafka')

    for frame in kafkastream():
        image = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        st.image(image, channels='BGR')
        st.write(predict_image(image))
        print(predict_image(image))
        
if __name__ == '__main__':
    main()