import cv2
import numpy as np


def find_letters_on_frame(frame):
    frame = frame.copy()
    ks, sigma = 1, 1
    frame = cv2.GaussianBlur(frame, (ks, ks), sigma)
    blue_mask = cv2.inRange(frame, np.array([0, 0, 0]), np.array([255, 135, 82]))
    frame = cv2.bitwise_and(frame, frame, mask=blue_mask)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    points = []

    for contour in contours:
        if 200 < cv2.contourArea(contour) < 1000:
            points.append(cv2.boundingRect(contour))

    letter_frames = []
    valid_points = []

    for point in points:
        x, y = point[0] - 30, point[1] - 30
        w, h = point[2] + 60, point[3] + 60

        if x > 30 and y > 30:
            letter_frames.append(frame[y:y+h, x:x+w])
            valid_points.append(point)

    return letter_frames, valid_points


def prepare_data(letters):
    for i, letter in enumerate(letters):
        letters[i] = cv2.resize(letter, (28, 28))

    letters = np.array(letters)
    # letters[letters > 120] = 255
    letters = letters / 255.0

    letters = letters.reshape(-1, 28, 28, 1)

    return letters


def predicted_to_numbers(predicted):
    return [(np.argmax(variants), max(variants)) for variants in predicted]


def append_results(frame, letter_positions, predicted):
    for idx, pos in enumerate(letter_positions):
        x, y, w, h = pos[0], pos[1], pos[2], pos[3]

        color = (255, 0, 255)
        if predicted[idx][1] > 0.9:
            color = (0, 0, 255)

        cv2.rectangle(frame, (x - 30, y - 30), (x + w + 30, y + h + 30), color, 3)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{predicted[idx][0]} ({predicted[idx][1]})",
                    (x-30, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
