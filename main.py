import os
import cv2
import time
import numpy as np
from utils import *
from keras.models import load_model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

capture = cv2.VideoCapture("assets/digits-video.mp4")
recognizer = load_model("assets/cnn-digits-recognition.sav")

start_time = time.time()
x, counter = 1, 0

while True:
    _, frame = capture.read()

    letter_frames, letter_positions = find_letters_on_frame(frame)
    letter_frames = prepare_data(letter_frames)

    predicted = recognizer.predict(letter_frames)

    predicted = predicted_to_numbers(predicted)

    append_results(frame, letter_positions, predicted)

    cv2.imshow("frame", cv2.resize(frame, (400, 700)))

    if cv2.waitKey(1) & 0xFF == 27:
        break

    if capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT):
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    counter += 1
    delta_time = time.time() - start_time
    if delta_time > x:
        print(f"fps: {counter/delta_time}")
        start_time = time.time()
        counter = 0
