import cv2
import numpy as np

"""Main part"""
whT = 320
confThreshold = 0.8
nmsThreshold = 0.1
Number_of_frame = []
cap = cv2.VideoCapture("DRONE-SURVEILLANCE-CONTEST-VIDEO.mp4")

#### LOAD MODEL
## Coco Names
classesFile = r"Car.names"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('n').split('\n')
print(classNames)
## Model Files
modelConfiguration = r"yolov4-custom.cfg"
modelWeights = "yolov4-custom-last.weights"
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    Number_of_frame.append(indices)
    print(len(Number_of_frame), len(indices))
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img, f'{classNames[0].upper()} {int(confs[i] * 100)}%',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


while True:
    success, img = cap.read()
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    findObjects(outputs, img)

    cv2.putText(img, "Marco Nguyen", (850, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
    cv2.putText(img, str(10), (1700, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key & 0xFF == 27:
        break


"""Extract all the frames"""
# import cv2
#
# # Opens the Video file
# cap = cv2.VideoCapture('DRONE-SURVEILLANCE-CONTEST-VIDEO.mp4')
# i = 0
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#     cv2.imwrite('junk-car-park' + str(i)+'.jpg', frame)
#     i += 1
#
# cap.release()
# cv2.destroyAllWindows()
