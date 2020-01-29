# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from itertools import islice
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import csv
import cv2

# load csv file from path
def loadCsvFile(csvPath):
    with open(csvPath) as csvFile:
        next(csvFile) # skip first line ['fn', 'label']
        csvData = list(csv.reader(csvFile))
    return csvData

# randomly shuffle the data 
def shuffleCsvData(csvData):
    csvData = sorted(csvData)
    random.seed(42)
    random.shuffle(csvData)

    return csvData
# load images from path
def loadImages(csvData, imagePath):
    listImages = list()
    listLabels = list()
    limitQtdImages = len(csvData)
    for (i, row) in enumerate(islice(csvData, limitQtdImages)): #48896
        image = cv2.imread(imagePath + row[0]).flatten()
        listImages.append(image)
        listLabels.append(row[1])
        # show an update every 1,000 images
        if i > 0 and i % 1000 == 0:
            print("[INFO] processed {}/{} images".format(i, len(csvData)))
    # scale the raw pixel intensities to the range [0, 1]
    listImages = np.array(listImages, dtype="float32") / 255.0
    listLabels = np.array(listLabels)

    return listImages, listLabels
    
# prepare train and test dataset
def prepare_data(listImages, listLabels):
    #75% of the data for training and the remaining 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(listImages,
            listLabels, test_size=0.25, random_state=42)

    # convert the labels from integers to vectors
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.transform(testY)
    return trainX, testX, trainY, testY, lb

# fit a Neural Network model
def fit_model(trainX, testX, trainY, testY, lb):
    # define the architecture model
    model = Sequential()
    input_shape = trainX[0].shape
    model.add(Dense(1024, input_shape=input_shape, activation="sigmoid"))
    model.add(Dense(512, activation="sigmoid"))
    model.add(Dense(len(lb.classes_), activation="softmax"))
    model.summary()

    # initialize learning rate, epochs to train for and bath_size
    lrate = 0.1
    epochs = 75
    bathSize = 64

    # compile the model using SGD as optimizer and categorical
    # cross-entropy loss
    print("[INFO] training network...")
    opt = SGD(lr=lrate)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
            metrics=["accuracy"])

    # train the neural network
    H = model.fit(trainX, trainY, validation_data=(testX, testY),
            epochs=epochs, batch_size=bathSize)

    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(testX, batch_size=bathSize)
    print(classification_report(testY.argmax(axis=1),
            predictions.argmax(axis=1), target_names=lb.classes_))
    
    # save the model and label binarizer to disk
    print("[INFO] serializing network and label binarizer...")
    model.save("./output/" + "simple_nn.model")
    f = open("./output/" + "simple_nn_lb.pickle", "wb")
    f.write(pickle.dumps(lb))
    f.close()
    
    return H, epochs, lrate

# save the training loss and accuracy graphic
def plotTrainingLossAccuracy(H, epochs, lrate):
    print("[INFO] Save the training loss and accuracy graphic...")
    # plot the training loss and accuracy
    N = np.arange(0, epochs)
    plt.style.use("ggplot")
    plt.figure(figsize=(10, 5))
    plt.plot(N, H.history["loss"], label="train_loss")
    plt.plot(N, H.history["val_loss"], label="val_loss")
    plt.plot(N, H.history["acc"], label="train_acc")
    plt.plot(N, H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy (Simple NN) bathSize = " + str(bathSize))
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig("./output/" + "simple_nn_plot_bs" + str(bathSize) + ".png")
    
    return

def main():
    # Path of input files (images and csv)
    csvPath =  "./input/train.truth.csv"
    imagePath = "./input/train/"

    csvData = loadCsvFile(csvPath)
    
    csvData = shuffleCsvData(csvData)
    
    listImages, listLabels = loadImages(csvData, imagePath)
    
    # prepare dataset
    trainX, testX, trainY, testY, lb = prepare_data(listImages, listLabels)

    H, epochs, lrate = fit_model(trainX, testX, trainY, testY, lb)
    
    # plotTrainingLossAccuracy(H, epochs, bathSize)
    
    print("[INFO] OK! finished.")

if __name__ == '__main__':
    main()
