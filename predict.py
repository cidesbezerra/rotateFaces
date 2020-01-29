# import the necessary packages
from keras.models import load_model
from imutils import paths
from PIL import Image
import numpy as np
import easydict
import argparse
import pickle
import cv2
import csv
import os

def loadTestImages(testImages, width, height, flatten):
    listTestImages = list()
    listTestNames = list()
    for imagePath in paths.list_images(testImages):
        # load the input test image
        image = cv2.imread(imagePath)
        output = image.copy()
        image = cv2.resize(image, (width, height))

        # scale the pixel values to [0, 1]
        image = image.astype("float") / 255.0

        # check to see if we should flatten the image and add a batch
        # dimension
        if flatten > 0:
            image = image.flatten()
            image = image.reshape((1, image.shape[0]))

        # otherwise, we must be working with a CNN -- don't flatten the
        # image, simply add the batch dimension
        else:
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        
        listTestImages.append(image)
        listTestNames.append(imagePath[imagePath.rfind("/") + 1:])# image name
    
    return listTestImages, listTestNames

def loadModelLabelBinarizer(model, label_bin):
    # load the model and label binarizer
    model = load_model(model)
    lb = pickle.loads(open(label_bin, "rb").read())
    
    return model, lb
    
def makePrediction(listTestImages, listTestNames, model, lb):
    correctedOrientation = []
    listTestPredicts = []
    for image, name in zip(listTestImages, listTestNames):
        # make a prediction on the image
        preds = model.predict(image)

        # find the class label index with the largest corresponding
        # probability
        i = preds.argmax(axis=1)[0]
        label = lb.classes_[i]

        print ("{} -- {}: {:.2f}%".format(name, label, preds[0][i] * 100))
        listTestPredicts.append((name, label))
        rotateTestImage(image, name, label, correctedOrientation)
    correctedOrientation = np.array(correctedOrientation)
    print ("Size Correct Orientation: ", correctedOrientation.shape)
    np.save('correctedOrientationFaces.npy', correctedOrientation)

    return listTestPredicts

def rotateTestImage(image, name, label, correctedOrientation):
    orientation = ["upright", "rotated_left", "rotated_right", "upside_down"]
    # scale the pixel values to [0, 1]
    image = np.reshape(image, (64, 64, 3)) * 255.0
    image = image.astype(np.uint8)
    # get image height, width
    (h, w) = image.shape[:2]
    # calculate the center of the image
    center = (w / 2, h / 2)
    angle90 = 90
    angle180 = 180
    angle270 = 270
    scale = 1.0
    
    name = name.split(".")[0] + ".png"
    
    if label == orientation[1]:
        # Perform the counter clockwise rotation holding at the center 270 degrees
        rotationMatrix = cv2.getRotationMatrix2D(center, angle270, scale)
        rotated270 = cv2.warpAffine(image, rotationMatrix, (h, w))
        correctedOrientation.append(rotated270)
        cv2.imwrite("./output/rotatedImages/" + name, rotated270)
        # print (name, label)
    elif label == orientation[2]:
        # 90 degrees
        rotationMatrix = cv2.getRotationMatrix2D(center, angle90, scale)
        rotated90 = cv2.warpAffine(image, rotationMatrix, (h, w))
        correctedOrientation.append(rotated90)
        cv2.imwrite("./output/rotatedImages/" + name, rotated90)
        # print (name, label)
    elif label == orientation[3]:
        # 180 degrees
        rotationMatrix = cv2.getRotationMatrix2D(center, angle180, scale)
        rotated180 = cv2.warpAffine(image, rotationMatrix, (w, h))
        correctedOrientation.append(rotated180)
        cv2.imwrite("./output/rotatedImages/" + name, rotated180)
        # print (name, label)
    else:
        cv2.imwrite("./output/rotatedImages/" + name, image)
        correctedOrientation.append(image)
        # print (name, label)
    return

def writeCsvFilePredictions(listTestPredicts):
    with open('test.preds.csv', 'w', newline='') as testPreds:
        writer = csv.writer(testPreds)
        writer.writerow(['fn', 'label']) # write first line
        for line in listTestPredicts:
            writer.writerow(line)
    testPreds.close()

def main():
    #/content/drive/My Drive/keras-tutorial
    args = easydict.EasyDict({
        "testImages": "./input/test",
        "model": "./output/simple_nn.model",
        "label_bin": "./output/simple_nn_lb.pickle",
        "width": 64,
        "height": 64,
        "flatten": 1,
        "outputPredImages" : "rotatedImages"
        # "plot": "./output/simple_nn_plot.png"
    })
    
    testImages = args["testImages"]
    model = args["model"]
    label_bin = args["label_bin"]
    width = args["width"]
    height = args["height"]
    flatten = args["flatten"]
    outputPredImages = args["outputPredImages"]
    
    if outputPredImages and not os.path.exists("./output/" + outputPredImages):
        os.makedirs("./output/" + outputPredImages)

    listTestImages, listTestNames = loadTestImages(testImages, width, height, flatten)
    
    print("[INFO] loading network and label binarizer...")
    model, lb = loadModelLabelBinarizer(model, label_bin)
    
    listTestPredicts = makePrediction(listTestImages, listTestNames, model, lb)
    
    writeCsvFilePredictions(listTestPredicts)



    # # draw the class label + probability on the output image
    # text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
    # cv2.putText(output, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            # (0, 0, 255), 2)

    # # show the output image
    # cv2.imshow("Image", output)
    # cv2.waitKey(0)

if __name__ == '__main__':
    main()
