import numpy as np
import argparse
import cv2
import time
from queue import Queue

import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input


# Tensorflow Error
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to input image")
ap.add_argument("-w", "--webcam", help="get image from webcam, please input camera number")
ap.add_argument("-v", "--video", help="path to input video")
args = vars(ap.parse_args())

# load face_detection_model
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('face_detection_model/deploy.prototxt.txt',
                               'face_detection_model/res10_300x300_ssd_iter_140000.caffemodel')

# load mask recognition model
trained_model = tf.keras.models.load_model('3class_less_image_without_augmentation.h5')

# setup a VideoCapture
cap = None
if args["webcam"]:
    cap = cv2.VideoCapture(int(args["webcam"]))
elif args["video"]:
    cap = cv2.VideoCapture(args["webcam"])

# video writer setting
fps = cap.get(cv2.CAP_PROP_FPS)
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
video_writer = cv2.VideoWriter('./result_video/mask_test.avi', fcc, fps, (video_width, video_height))

# mask warning
mask_warning_queue = Queue(100)

# time setting
prevTime = 0

# main Code
while cap.isOpened():

    # time setting
    curTime = time.time()
    sec = curTime - prevTime
    prevTime = curTime

    # Get Image from our source
    if cap:
        _, image = cap.read()
    else:
        image = cv2.imread(args["image"])

    # getting height and width of image
    (height, width) = image.shape[:2]

    # Creating a blob from image
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # [,frame,no of detections,[classid,class score,conf,x,y,h,w]
    # loop over the detections

    # find box as many as a number of detected face
    for i in range(0, detections.shape[2]):

        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # Accuracy(Confidence) Threshold (default = 50%)
        if confidence > 0.5:
            # get face box from detected face
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")

            # box offset config
            startX -= 15 if startX - 10 > 0 else startX
            startY -= 20 if startY - 20 > 0 else startY
            endX = endX + 15 if endX + 10 < image.shape[1] else image.shape[1]
            endY = endY + 15 if endY + 15 < image.shape[0] else image.shape[0]

            # Mask Prediction
            resized_image = cv2.resize(image[startY:endY, startX:endX],
                                       dsize=(256, 256),
                                       interpolation=cv2.INTER_LINEAR)
            x = img_to_array(resized_image)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            predictions = trained_model.predict(x)
            predictions = tf.nn.softmax(predictions)

            labels = np.argmax(predictions, axis=1)

            # if labels is 0 -> with mask
            if labels == 0:
                label = "with_mask"
                color = (0, 255, 0)
                if not mask_warning_queue.empty() is True:
                    mask_warning_queue.get()

            # if labels is 1 -> without mask
            elif labels == 1:
                label = "without_mask"
                color = (0, 0, 255)

            # if labels is 2 -> nose mask
            else:
                # check the mask_warning_queue
                if mask_warning_queue.full() is True:
                    label = "put on your mask correctly!!"
                    color = (0, 0, 255)


                else:
                    label = "nose_mask"
                    color = (0, 255, 255)
                    mask_warning_queue.put(1)

            # draw the bounding box of the face along with the associated probability
            accuracy_position = startY - 10 if startY - 10 > 10 else startY + 10

            # information print
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
            cv2.putText(image, label, (startX, accuracy_position),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # FPS
    fps = 1 / (sec)
    fps_str = "FPS : %0.1f" % fps
    cv2.putText(image, fps_str, (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # show the output image
    cv2.imshow("Output", image)
    video_writer.write(image)
    key = cv2.waitKey(1)

    if key == ord("s"):
        cv2.waitKey(0)

    if key == 27:
        break

video_writer.release()
cap.release()
cv2.destroyAllWindows()
