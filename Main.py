import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math, random, time


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

counter = 0
countDown = 0
AIChoice = ''
YourChoice = ''
aScore = 0
uScore = 0
result = ''

labels = ["Rock", "Paper", "Scissors"]

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    cv2.putText(imgOutput, "FLL 2023.24".format(), (360, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.88, (255, 255, 0), 2)
    cv2.putText(imgOutput, "MASTERPIECE".format(), (360, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.88, (0, 255, 0), 2)
    cv2.putText(imgOutput, "{}".format(result), (400, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 255), 2)
    cv2.putText(imgOutput, "AI: {}".format(AIChoice), (20, 60), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(imgOutput, "You: {}".format(YourChoice), (20, 90), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(imgOutput, "AI Score: {}".format(aScore), (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
    cv2.putText(imgOutput, "Your Score: {}".format(uScore), (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            ## print(prediction, index)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                      (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 0.55, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset),
                      (x + w+offset , y + h+offset ), (255, 0, 255), 3)

        if labels[index] and countDown < 3:
            countDown = countDown + 1
            if countDown > 2 :
                AIChoice = random.choice(labels)
                YourChoice = labels[index]

                if YourChoice == AIChoice:
                    result = 'Tie'
                elif YourChoice == "Rock" and AIChoice == "Scissors":
                    uScore = uScore + 1
                    result = 'You Win'
                elif YourChoice == "Paper" and AIChoice == "Rock":
                    uScore = uScore + 1
                    result = 'You Win'
                elif YourChoice == "Scissors" and AIChoice == "Paper":
                    uScore = uScore + 1
                    result = 'You Win'
                else:
                    result = 'AI Win'
                    aScore = aScore + 1

        cv2.putText(imgOutput, "AI: {}".format(AIChoice), (20, 60), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(imgOutput, "You: {}".format(YourChoice), (20, 90), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(imgOutput, "AI Score: {}".format(aScore), (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        cv2.putText(imgOutput, "Your Score: {}".format(uScore), (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

        ## print(labels[index])

    else:
        countDown = 0
        result = ''



    ## cv2.imshow("ImageCrop", imgCrop)
    ## cv2.imshow("ImageWhite", imgWhite)
    imgOutput = cv2.resize(imgOutput, (1024, 768))
    cv2.imshow("Image", imgOutput)

    key = cv2.waitKey(1)
    if key == ord('r'):  # press r to reset the image
        aScore = 0
        uScore = 0
    if key == ord('q'):  # press q to quit
        break
