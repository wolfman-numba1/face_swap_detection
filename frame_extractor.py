import cv2
import os

path = "/Volumes/My1TBDrive/FF++/original_sequences/youtube/c23/videos/"
pathOut = "/Volumes/My1TBDrive/FF++/extracted_frames/original/"

for i in os.listdir(path):
    count = 0
    name = i.replace(".mp4", "")
    reader = cv2.VideoCapture(path+i)
    success,image = reader.read()
    success = True
    while success:
        reader.set(cv2.CAP_PROP_POS_MSEC,(count*1000))    # added this line 
        success,image = reader.read()
        # print ('Read a new frame: ', success)
        if success is False:
            break
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
        )
        print("[INFO] Found {0} Faces.".format(len(faces)))
        count2 = 0
        for (x, y, w, h) in faces:
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_color = image[y-50:y + h+50, x-50:x + w+50]
            print("[INFO] Object found. Saving locally.")
            try:
                # cv2.imwrite(pathOut + i, roi_color)
                cv2.imwrite(pathOut + name + "_frame%d_%d.jpg" % (count,count2), roi_color)
            except Exception:
                print("Error saving image.")
                continue
            count2 = count2 + 1
        # cv2.imwrite( pathOut + name + "_frame%d.jpg" % count, image)     # save frame as JPEG file
        count = count + 1