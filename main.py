import cv2
from nms import nms  # Importing the Non-Maximum Suppression (NMS) module

# Configuration paths for the pre-trained model and labels
configPath = ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt
weightsPath = frozen_inference_graph.pb
classLabels = []  # List to store class labels

# File containing class labels is read and stored in classLabels list
filename = r'C:\Users\PRAKHAR TYAGI\Desktop\geu\c\mini_project\labels.txt.crdownload'
with open(filename, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

# Loading the pre-trained SSD MobileNet V3 Large model using OpenCV's dnn module
net = cv2.dnn_DetectionModel(weightsPath, configPath)
cap = cv2.VideoCapture(r'C:\Users\PRAKHAR TYAGI\Downloads\pexels_videos_2034115 (1080p).mp4')  # Opening video file
font_scale = 3  # Font scale for text in the video frames
font = cv2.FONT_HERSHEY_PLAIN  # Font type

# Function to resize frames for computational efficiency
def resize(frame, size=0.65):
    width = int(frame.shape[1] * size)
    height = int(frame.shape[0] * size)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

while True:
    isTrue, frame = cap.read()  # Reading frames from the video
    if not isTrue:
        break
    isTrue, frame2 = cap.read()
    if not isTrue:
        break
    frame = resize(frame)
    frame2 = resize(frame2)
    diff = cv2.absdiff(frame, frame2)  # Calculating the absolute difference between consecutive frames
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)  # Converting the difference image to grayscale
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Applying Gaussian blur to reduce noise
    _, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)  # Thresholding to isolate potential objects
    dilated = cv2.dilate(thresh, None, iterations=3)  # Dilation to expand object regions for robust detection
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # Finding contours in the dilated image

    bboxes = []  # List to store bounding boxes of potential objects

    for c in contours:
        if cv2.contourArea(c) < 5500:  # Ignoring contours with small area (noise)
            continue

        net.setInputSize(300, 300)  # Setting input size for the model
        net.setInputScale(1.0 / 127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)

        ClassIndex, confidence, bbox = net.detect(frame, confThreshold=0.55)  # Object detection using the pre-trained model

        for i, box in enumerate(bbox):
            x, y, w, h = box
            if 0 <= ClassIndex[i] - 1 < len(classLabels):
                class_name = classLabels[ClassIndex[i] - 1].upper()
                confidence_score = confidence[i]
                bboxes.append([class_name, confidence_score, x, y, x + w, y + h])  # Storing bounding box information

    # Applying Non-Maximum Suppression (NMS)
    bboxes_after_nms = nms(bboxes, iou_threshold=0.5, threshold=0.1, box_format="corners")

    for box in bboxes_after_nms:
        class_name, confidence_score, x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)  # Drawing rectangles around detected objects
        cv2.putText(frame, f"{class_name}: {round(confidence_score * 100, 2)}%", (x1 + 10, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)  # Displaying class name and confidence score

    cv2.imshow('frame', frame)  # Displaying the frame with detected objects
    if cv2.waitKey(10) == ord('q'):  # Exiting the loop if 'q' is pressed
        break

cap.release()  # Releasing the video capture object
cv2.destroyAllWindows()  # Closing all OpenCV windows
