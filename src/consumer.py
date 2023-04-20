import streamlit as st
from kafka import KafkaConsumer
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import tensorflow as tf

consumer = KafkaConsumer('my-topic', bootstrap_servers='localhost:9092')
model = load_model('/Users/antony.vargasulead.ac.cr/BigData/proyecto_final/algorithms/modelCNN.h5')


def predict_image(image):
    im = cv2.resize(image, (64, 64))
    im = im.astype('float32')
    im = im / 255.
    im_input = tf.reshape(im, shape=[1, 64, 64, 3])

    predict_proba = sorted(model.predict(im_input)[0])[-1]
    predict_class = np.argmax(model.predict(im_input))

    if predict_class == 1:
        predict_label = 'Piedra'
    elif predict_class == 0:
        predict_label = 'Piedra'
    else:
        predict_label = 'Piedra'
    return predict_label, round(predict_proba * 100, 2)

def main():
    st.title('Computer Vision with Kafka')

    button_predict = st.button('Predict')

    for message in consumer:
        frame = cv2.imdecode(np.frombuffer(message.value, np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(image, channels='RGB')

        if button_predict:
            st.write(predict_image(image))
            print(predict_image(image))
            break


if __name__ == '__main__':
    main()
