# Shashank_Portfolio
Data Science Portfolio

## [Project 1: Eye for Blind Project](https://github.com/ShashankRaghu/Data_Science_Projects/tree/main/eye_for_blind)
### The premise of this project is to create a model, which take a image as input and converts it into a sound byte describing the image. The ML part involves, converting image to caption. Conversion of the captions generated into an audio output with the help of text-to-speech library.

* #### The dataset used in this project can be downloaded from Kaggle: https://www.kaggle.com/adityajn105/flickr8k
* #### The dataset is an image as an input and a caption as the output. Each image is repeat 5 times with slightly different captions.
![Eye_for_blind1](https://user-images.githubusercontent.com/77088516/125979586-13e516e7-40a5-4030-9121-b8482f750be3.PNG)
* #### We have preprocessed the image and the caption, we have also converted the caption into vectors.
* #### The model used is a encoder-decoder model. The encoder used to encode the image into feature map is a Inception net V3 model. The decoder used is RNN model.
* #### We have also used an attention model in between the encoder and decoder. We can see the attention model at work below:
![Eye_for_blind2](https://user-images.githubusercontent.com/77088516/125981732-35874d14-871e-4da5-88cb-ba3ea14bc21f.PNG)
* #### We have used Beam Search to select the output caption, and BLUE score to evaluate the model. The BLUE achieved wa 32.38

## [Project 2:Gesture Recognition Project](https://github.com/ShashankRaghu/Data_Science_Projects/tree/main/GestureRecognition)
### This project deals with developing a model which takes a video(30 frames) of a hand gesture as input and classify it into one of 5 gestures.

* #### The dataset can be downloaded from this [drive](https://drive.google.com/uc?id=1ehyrYBQ5rbQQe6yL4XbLWe3FMvuVUGiL).
* #### We have build a generator to set up the data ingestion pipeline, we have perpormed image pre-processing techniques.
* #### We have build several 3D CNN models, and tuned the hyper parameters; learning rate, optimizer used etc.
* #### We have tried Batch Normalization, Dropout and some other techniques to arrive at the best model.
* #### The best model is a 5 layer CNN model with relu activation function, Batch Normalization, Dropout, optimizer: RMSProp and learning rate: 0.005.
