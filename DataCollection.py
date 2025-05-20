from time import time
import cv2
import cvzone
from cvzone.FaceDetectionModule import FaceDetector

####################################
classID = 0  # 0 is fake and 1 is real
outputFolderPath = 'Dataset/DataCollect/Fake'
confidence = 0.8
save = True
blurThreshold = 35  # Larger is more focus

debug = False
offsetPercentageW = 10
offsetPercentageH = 20
camWidth, camHeight = 640, 480
floatingPoint = 6
####################################

cap = cv2.VideoCapture(0)
cap.set(3, camWidth)
cap.set(4, camHeight)

detector = FaceDetector()

while True:
    success, img = cap.read()
    imgOut = img.copy()

    img, bboxs = detector.findFaces(img, draw=False)

    listBlur = []  # True/False indicating if faces are blur or not
    listInfo = []  # Normalized values and class for label txt file

    if bboxs:
        for bbox in bboxs:
            x, y, w, h = bbox["bbox"]
            score = bbox["score"][0]

            if score > confidence:
                # Add offset to face bounding box
                offsetW = (offsetPercentageW / 100) * w
                x = int(x - offsetW)
                w = int(w + 2 * offsetW)
                offsetH = (offsetPercentageH / 100) * h
                y = int(y - offsetH * 3)
                h = int(h + offsetH * 3.5)

                # Avoid negative values
                x = max(0, x)
                y = max(0, y)
                w = max(0, w)
                h = max(0, h)

                # Check blurriness
                imgFace = img[y:y + h, x:x + w]

                blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())
                listBlur.append(blurValue > blurThreshold)

                # Normalize bounding box values
                ih, iw, _ = img.shape
                xc, yc = x + w / 2, y + h / 2

                xcn = round(xc / iw, floatingPoint)
                ycn = round(yc / ih, floatingPoint)
                wn = round(w / iw, floatingPoint)
                hn = round(h / ih, floatingPoint)

                # Clamp values to max 1
                xcn = min(1, xcn)
                ycn = min(1, ycn)
                wn = min(1, wn)
                hn = min(1, hn)

                listInfo.append(f"{classID} {xcn} {ycn} {wn} {hn}\n")

                # Draw rectangle and text
                cv2.rectangle(imgOut, (x, y), (x + w, y + h), (255, 0, 0), 3)
                cvzone.putTextRect(imgOut, f'Score: {int(score * 100)}% Blur: {blurValue}', (x, y),
                                   scale=2, thickness=3)

                if debug:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
                    cvzone.putTextRect(img, f'Score: {int(score * 100)}% Blur: {blurValue}', (x, y),
                                       scale=2, thickness=3)

        # Save image and label if all faces are clear
        if save and listBlur and all(listBlur):
            timeNow = str(time()).replace('.', '')
            cv2.imwrite(f"{outputFolderPath}/{timeNow}.jpg", img)
            with open(f"{outputFolderPath}/{timeNow}.txt", 'a') as f:
                f.writelines(listInfo)

    cv2.imshow("Image", imgOut)
    cv2.waitKey(1)
