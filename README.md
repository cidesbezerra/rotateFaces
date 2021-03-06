# rotateFaces

This project consists of recognizing the rotation of people's face images (`upright`,`rotated_left`, `rotated_right`, or `upside_down`) and correcting that rotation so that the faces are upright.


For this, the model of Neural Network with the Keras library was adapted to the following specifications:
* Read images converted to 1d with resolution `64 * 64 * 3 = 12288`.
* Scale the raw pixel intensities to the range `[0, 1]`.
* Convert the labels from integers to binary vectors (a binary vector corresponding to each label).
* The Neural Network model was compiled using using SGD as optimizer and categorical cross-entropy loss.
* Learning Rate: `lrate = 0.1`.
* Epochs: `epochs = 75`.
* Bath Size: `bathSize = 64`.

The source code consists of:
* `trainSimpleNeuralNetwork.py` - the entire configuration of the Neural Network.
* `predict.py` - loads the model and weights from the neural network (generated by `trainSimpleNeuralNetwork.py`) to make predictions in the test dataset.

# Run training model
To start training the Neural Network use:
    
    python trainSimpleNeuralNetwork.py

The input data must be:
`csvPath =  "./input/train.truth.csv"`- path and name of the csv file with the names and labels of the images for training.
`imagePath = "./input/train/"`- path of the images for training.

## Output after training
The output after training step is:
* Directory called `output` containing the weights of the trained model and the training loss and accuracy graphic.

#  Run predict 
To make prediction use:

    python predict.py

The input data must be:

`testImages = "./input/test"` - path of the images for test predict. 

`model = "./output/simple_nn.model"` - configuration file and weights of Neural Network model.

`label_bin = "./output/simple_nn_lb.pickle"` - binary file of labels.

`outputPredImages = "rotatedImages"` - folder called `rotatedImages` created inside the `./output./` directory to save the images with the corrected rotation.

## Output after predict
The output after predict step is:
* Binary file called `correctedOrientationFaces.npy` containing the images with the corrected rotation.
* Csv file called `test.preds.csv` containing the names and the predicted labels of the test images.

# Download
The binary file:

[**Download link**](https://www.dropbox.com/s/35f5y5q52t5giz6/rotatedImages.zip?dl=0)
