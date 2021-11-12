import os
imagePath = "/Volumes/My1TBDrive/FF++/extracted_frames/original_processed/"

j = 0
for i in os.listdir(imagePath):
    if "test" in i or "validation" in i or "train" in i:
        continue
    if j % 5 == 0:
        os.rename(imagePath + i, imagePath + "train/" + i)
        # print(imagePath + "train/" + i)
    if j % 5 == 1:
        os.rename(imagePath + i, imagePath + "validation/" + i)
        # print(imagePath + "validation/" + i)
    if j % 5 == 2:
        os.rename(imagePath + i, imagePath + "train/" + i)
        # print(imagePath + "train/" + i)
    if j % 5 == 3:
        os.rename(imagePath + i, imagePath + "test/" + i)
        # print(imagePath + "test/" + i)
    if j % 5 == 4:
        os.rename(imagePath + i, imagePath + "train/" + i)
        # print(imagePath + "train/" + i)
    j = j + 1