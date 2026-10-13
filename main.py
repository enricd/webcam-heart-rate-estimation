"""
Live heart rate from your webcam (rPPG with the green channel).

Left: webcam with the detected face. Right: the green signal of the last
~20 s, its DFT and the estimated heart rate.

Run:   uv run main.py      (quit with q)
"""
import time
from collections import deque

import cv2
import numpy as np

BUFFER = 600                 # frames kept: ~20 s at 30 fps
BOX_SMOOTH = 10              # face boxes averaged: ~0.33 s at 30 fps
LOW_HZ, HIGH_HZ = 0.6, 3.0   # heart-rate band: 36 to 180 bpm
GREEN, RED, WHITE = (0, 255, 0), (0, 0, 255), (255, 255, 255)   # colors are BGR
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_plot(img, values, title, top, color, marker=None):
    """Line plot inside a 600x180 box. The y axis fits the data's min and max."""
    left, w, h = 20, 600, 180
    cv2.rectangle(img, (left, top), (left + w, top + h), (80, 80, 80), 1)
    cv2.putText(img, title, (left + 8, top + 20), FONT, 0.5, WHITE, 1)
    y = np.asarray(values, dtype=float)
    if len(y) < 2:
        return
    px = left + np.linspace(0, w, len(y))
    py = top + h - 5 - (y - y.min()) / (y.max() - y.min() + 1e-9) * (h - 35)
    cv2.polylines(img, [np.column_stack([px, py]).astype(np.int32)], False, color, 1)
    if marker is not None:   # red vertical line at index `marker`
        mx = int(left + marker / (len(y) - 1) * w)
        cv2.line(img, (mx, top + 25), (mx, top + h), RED, 1)


# Step 1: open the webcam (640x480 is plenty and keeps face detection fast)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Step 2: face detector (the Haar cascade ships with OpenCV 4.x)
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

greens = deque(maxlen=BUFFER)    # one green value per frame
times = deque(maxlen=BUFFER)     # when each frame arrived, to measure the real fps
boxes = deque(maxlen=BOX_SMOOTH) # the last face boxes, averaged to remove jitter

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Step 2: find the largest face (keep the last boxes if this frame misses it)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    if len(faces):
        boxes.append(max(faces, key=lambda f: f[2] * f[3]))

    # Step 3: one number per frame, the mean GREEN value (index 1 in BGR) of the face.
    # The box is the average of the last boxes, so it does not jitter from frame to frame.
    if boxes:
        x, y, w, h = np.mean(boxes, axis=0).astype(int)
        greens.append(frame[y:y+h, x:x+w, 1].mean())
        times.append(time.time())
        cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 2)

    dash = np.zeros((frame.shape[0], 640, 3), np.uint8)   # same height as the camera image
    draw_plot(dash, greens, "green mean of the face (last 20 s)", 80, GREEN)

    if len(greens) > 150:        # wait for ~5 s of signal
        fps = (len(times) - 1) / (times[-1] - times[0])

        # Step 4: remove slow drift (light, exposure, small moves) by subtracting
        # a 1 s moving average. What is left are the faster changes, like the pulse.
        g = np.array(greens) - np.mean(greens)
        k = int(fps)
        g = g - np.convolve(g, np.ones(k) / k, mode="same")

        # Step 5: DFT. The Hann window fades the edges of the signal in and out,
        # so whatever drift is left does not leak into the heart-rate band.
        spectrum = np.abs(np.fft.rfft(g * np.hanning(len(g))))
        freqs = np.fft.rfftfreq(len(g), d=1 / fps)
        band = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)

        # Step 6: strongest frequency in the band -> beats per minute
        peak = np.argmax(spectrum[band])
        bpm = freqs[band][peak] * 60

        cv2.putText(dash, f"HR {bpm:.0f} bpm", (20, 50), FONT, 1.2, RED, 2)
        cv2.putText(dash, f"{fps:.0f} fps", (520, 50), FONT, 0.6, WHITE, 1)
        draw_plot(dash, spectrum[band], "DFT from 36 to 180 bpm", 285, WHITE, marker=peak)
    else:
        cv2.putText(dash, "collecting signal...", (20, 50), FONT, 1, WHITE, 2)

    cv2.imshow("rPPG heart rate", np.hstack([frame, dash]))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
