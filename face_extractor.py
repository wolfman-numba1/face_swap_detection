import cv2
import sys
import os

imagePath = "/Volumes/My1TBDrive/FF++/extracted_frames/original/"
pathOut = "/Volumes/My1TBDrive/FF++/extracted_frames/original_extracted_faces/"

for i in os.listdir(imagePath):
    if i == ".DS_Store":
        continue
    image = cv2.imread(imagePath + i)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )

    print("[INFO] Found {0} Faces.".format(len(faces)))

    for (x, y, w, h) in faces:
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_color = image[y-20:y + h+20, x-20:x + w+20]
        print("[INFO] Object found. Saving locally.")
        try:
            cv2.imwrite(pathOut + i, roi_color)
        except Exception:
            print("Error saving image.")
            continue
        
