import os
import cv2
import time
import numpy as np
from utils import *
from keras.models import load_model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

capture = cv2.VideoCapture("assets/digits-video.mp4")
model = load_model("assets/cnn-digits-recognition.sav")

out = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (800, 700))
start_time = time.time()
x, counter = 1, 0

while True:
    _, frame = capture.read()

    letter_frames, letter_positions = find_letters_on_frame(frame)
    prepared_letter_frames = prepare_data(letter_frames)

    predicted = model.predict(prepared_letter_frames)

    predicted = predicted_to_numbers(predicted)

    append_results(frame, letter_positions, predicted)

    debug_window = create_letters_frame(frame, prepared_letter_frames, letter_positions)
    debug_window = cv2.cvtColor(debug_window, cv2.COLOR_GRAY2RGB)
    resized_debug_window = cv2.resize(debug_window, (400, 700))
    resized_frame = cv2.resize(frame, (400, 700))
    result_frame = np.hstack([resized_debug_window, resized_frame])

    print(resized_debug_window.shape, resized_frame.shape)

    cv2.imshow("Window", result_frame)
    out.write(result_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    # if capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT):
    #     capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    counter += 1
    delta_time = time.time() - start_time
    if delta_time > x:
        print(f"fps: {counter/delta_time}")
        start_time = time.time()
        counter = 0

out.release()
capture.release()
