# import tensorflow as tf
# import numpy as np
# import cv2
# import argparse
# import os
# from tensorflow.keras.preprocessing.image import img_to_array
# from focal_loss import BinaryFocalLoss
# import tensorflow_addons as tfa

# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-v", "--video_path", type=str, required=True, help="path to input video; enter 0 for camera")
# ap.add_argument("-o", "--output_folder", type=str, required=True, help="path to output frames; enter 0 for display")
# ap.add_argument("-s", "--skip", type=int, required=False, default=10, help="number of consecutive frames to skip")
# ap.add_argument("-d", "--delay", type=int, required=False, default=200, help="delay in ms")
# args = vars(ap.parse_args())

# # custom objects
# fl = BinaryFocalLoss(gamma=2)
# F1Score = tfa.metrics.F1Score(num_classes=2, average='macro')

# MODEL = r'best_model_16_11pm.h5'
# CONFIDENCE_REAL = 0.7
# CONFIDENCE_FAKE = 0.8
# LABELS = ['fake', 'real']
# N_SKIP = args["skip"]
# VID_PATH = args["video_path"]
# DEST = args["output_folder"]
# DELAY = args['delay']

# # load the liveness detector model and label encoder from disk
# print("[INFO] loading liveness detector...")
# model = tf.keras.models.load_model(MODEL, custom_objects={'BinaryFocalLoss': fl, 'F1Score': F1Score})

# # initialize the video stream
# print("[INFO] starting video stream...")
# if VID_PATH == '0':
#     print('Using camera.....')
#     cap = cv2.VideoCapture(0)
# else:
#     print('Using the input video')
#     cap = cv2.VideoCapture(VID_PATH)

# # create output directory
# if DEST != '0':
#     if not os.path.exists(DEST):
#         os.makedirs(DEST)

# FRAME_COUNT = 0
# PROC_COUNT = 0

# # load our serialized face detector from disk
# print("[INFO] loading face detector...")
# FACE_DETECTOR = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

# # loop over the frames from the video stream
# while True:
#     (grabbed, frame) = cap.read()
#     if not grabbed:
#         break

#     FRAME_COUNT += 1
#     if FRAME_COUNT % N_SKIP != 0:
#         continue

#     (h, w) = frame.shape[:2]
#     min_size = int(min((h, w)) * 0.2)

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     detections = FACE_DETECTOR.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(min_size, min_size),
#                                                 flags=cv2.CASCADE_SCALE_IMAGE)

#     for (x, y, w, h) in detections:
#         face = frame[y:y + h, x:x + w]
#         face = cv2.resize(face, (64, 64))
#         face = face.astype("float") / 255.0
#         face = img_to_array(face)
#         face = np.expand_dims(face, axis=0)

#         preds = model.predict(face)[0]
#         fake_prob = preds[0]
#         if fake_prob >= CONFIDENCE_FAKE:
#             label = 'FAKE'
#             rect_color = (0, 0, 255)
#             conf = fake_prob
#         else:
#             label = 'REAL'
#             rect_color = (0, 255, 0)
#             conf = fake_prob + 0.2

#         txt_label = "{}: {} %".format(label.upper(), round(100 * conf, 1))
#         cv2.putText(frame, txt_label, (x, y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, rect_color, 2)
#         cv2.rectangle(frame, (x, y), (x + w, y + h),
#                       rect_color, 2)

#     PROC_COUNT += 1

#     if DEST == '0':
#         print('Displaying the detections..')
#         cv2.imshow("Frame", frame)
#         key = cv2.waitKey(DELAY) & 0xFF
#         if key == ord("q"):
#             break
#     else:
#         print('Saving the processed frame to ', DEST)
#         p = os.path.join(DEST, "{}.png".format(PROC_COUNT))
#         cv2.imwrite(p, frame)

# cv2.destroyAllWindows()
# cap.release()
import tensorflow as tf
import numpy as np
import cv2
import argparse
import os
from tensorflow.keras.preprocessing.image import img_to_array
from focal_loss import BinaryFocalLoss
import tensorflow_addons as tfa

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video_path", type=str, required=True, help="path to input video; enter 0 for camera")
ap.add_argument("-o", "--output_folder", type=str, required=True, help="path to output frames; enter 0 for display")
ap.add_argument("-s", "--skip", type=int, required=False, default=10, help="number of consecutive frames to skip")
ap.add_argument("-d", "--delay", type=int, required=False, default=200, help="delay in ms")
args = vars(ap.parse_args())

# custom objects
fl = BinaryFocalLoss(gamma=2)
F1Score = tfa.metrics.F1Score(num_classes=2, average='macro')

MODEL = r'best_model_16_11pm.h5'
CONFIDENCE_REAL = 0.7
CONFIDENCE_FAKE = 0.8
LABELS = ['fake', 'real']
N_SKIP = args["skip"]
VID_PATH = args["video_path"]
DEST = args["output_folder"]
DELAY = args['delay']

# load the liveness detector model and label encoder from disk
print("[INFO] loading liveness detector...")
model = tf.keras.models.load_model(MODEL, custom_objects={'BinaryFocalLoss': fl, 'F1Score': F1Score})

# initialize the video stream
print("[INFO] starting video stream...")
if VID_PATH == '0':
    print('Using camera.....')
    cap = cv2.VideoCapture(0)
else:
    print('Using the input video')
    cap = cv2.VideoCapture(VID_PATH)

# create output directory
if DEST != '0':
    if not os.path.exists(DEST):
        os.makedirs(DEST)

FRAME_COUNT = 0
PROC_COUNT = 0

# load our serialized face detector from disk
print("[INFO] loading face detector...")
FACE_DETECTOR = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

# loop over the frames from the video stream
while True:
    (grabbed, frame) = cap.read()
    if not grabbed:
        break

    FRAME_COUNT += 1
    if FRAME_COUNT % N_SKIP != 0:
        continue

    (h, w) = frame.shape[:2]
    min_size = int(min((h, w)) * 0.2)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = FACE_DETECTOR.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(min_size, min_size),
                                                flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in detections:
        face = frame[y:y + h, x:x + w]
        face = cv2.resize(face, (64, 64))
        face = face.astype("float") / 255.0
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)

        preds = model.predict(face)[0]
        fake_prob = preds[0]
        if fake_prob >= CONFIDENCE_FAKE:
            label = 'FAKE'
            rect_color = (0, 0, 255)
            conf = fake_prob
        else:
            label = 'REAL'
            rect_color = (0, 255, 0)
            conf = fake_prob + 0.2

        txt_label = "{}: {} %".format(label.upper(), round(100 * conf, 1))
        cv2.putText(frame, txt_label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, rect_color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      rect_color, 2)

    PROC_COUNT += 1

    if DEST == '0':
        print('Displaying the detections..')
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(DELAY) & 0xFF
        if key == ord("q"):
            break
    else:
        print('Saving the processed frame to ', DEST)
        p = os.path.join(DEST, "{}.png".format(PROC_COUNT))
        cv2.imwrite(p, frame)

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

