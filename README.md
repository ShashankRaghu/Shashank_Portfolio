# Shashank_Portfolio
### Data Science Portfolio

# [Project 1: Eye for Blind Project](https://github.com/ShashankRaghu/Data_Science_Projects/tree/main/eye_for_blind)
### The premise of this project is to create a model, which take a image as input and converts it into a sound byte describing the image. The ML part involves, converting image to caption. Conversion of the captions generated into an audio output with the help of text-to-speech library.

* #### The dataset used in this project can be downloaded from Kaggle: https://www.kaggle.com/adityajn105/flickr8k
* #### The dataset is an image as an input and a caption as the output. Each image is repeat 5 times with slightly different captions.
![Eye_for_blind1](https://user-images.githubusercontent.com/77088516/125979586-13e516e7-40a5-4030-9121-b8482f750be3.PNG)
* #### We have preprocessed the image and the caption, we have also converted the caption into vectors.
* #### The model used is a encoder-decoder model. The encoder used to encode the image into feature map is a Inception net V3 model. The decoder used is RNN model.
* #### We have also used an attention model in between the encoder and decoder. We can see the attention model at work below:
![Eye_for_blind2](https://user-images.githubusercontent.com/77088516/125981732-35874d14-871e-4da5-88cb-ba3ea14bc21f.PNG)
* #### We have used Beam Search to select the output caption, and BLUE score to evaluate the model. The BLUE achieved wa 32.38

# [Project 2: Gesture Recognition Project](https://github.com/ShashankRaghu/Data_Science_Projects/tree/main/GestureRecognition)
### This project deals with developing a model which takes a video(30 frames) of a hand gesture as input and classify it into one of 5 gestures.

* #### The dataset can be downloaded from this [drive](https://drive.google.com/uc?id=1ehyrYBQ5rbQQe6yL4XbLWe3FMvuVUGiL).
* #### We have build a generator to set up the data ingestion pipeline, we have perpormed image pre-processing techniques.
* #### We have build several 3D CNN models, and tuned the hyper parameters; learning rate, optimizer used etc.
* #### We have tried Batch Normalization, Dropout and some other techniques to arrive at the best model.
* #### The best model is a 5 layer CNN model with relu activation function, Batch Normalization, Dropout, optimizer: RMSProp and learning rate: 0.005.

# [Project 3: Telecom Churn Prediction](https://github.com/ShashankRaghu/Data_Science_Projects/tree/main/TelecomChurn)
### The aim of this project is to build a model which can predict wheather a custmoer will churn(leave the netwrok). The dataset conatins information about a customers cellular transactions for the months June(6) to September(9). We need to predict of the customer will churn in September using the data from month 6 to 8.

* #### The dataset can be download from [this link](https://drive.google.com/file/d/1SWnADIda31mVFevFcfkGtcgBHTKKI94J/view).
* #### We have performed the following preprocessing steps:
  * #### Removing columns which have only one unique value.
  * #### For handling the missing values we found out that the missing values were MNAR(Missing not at random). We saw that the columns were missing when customers haven't used the service, hence we replaced these columns with zero.
  * #### Since there is a large number of customers, we want to focus on only the high value customer. So we sepearted the dataset with the top 30 percentile spenders in month 6 and 7.
* #### We calculated dependent variable "churn" by looking at total expenditure of customers in month 9, and those customers who spent 0 amount were tagged as churned.
* #### For EDA we have looked at the histograms, boxplots, barplots and heatmap. Below is the heatmap showing the correlation co-efficients.
![TelecomChurnGit](https://user-images.githubusercontent.com/77088516/125994623-84991dfb-f8c2-4ce2-bb85-f560e0deeb32.PNG)
* #### We performed some manual feature engineering and PCA to reduce the data to 60 columns.
* #### The skewness in the data is handles by PowerTransformation, and the data imbalance is handled by class weights. 
* #### For the model building we used StrifiedKFold for validation testing, and GridSearchCV for finding the best parameter. XGBoost, RandomForestClassifier and Logistic Regression models were tried.
* #### The Logistic Regression was the best model, with an accuracy of ~83.5%.
* #### We have also built Logistic Regression and Decision Tree models with non PCA data, in order to find the top predictors in the dataset.

# [Project 4: Parts of Speech Tagger](https://github.com/ShashankRaghu/Data_Science_Projects/tree/main/PartsOfSpeechTagger)
### To build a model which can predict the parts of speech of a sentence

- #### The dataset used in from the NLTK corpora, which contains sentences, and POS of each word.
- #### The sentences are converted to intergers using Tokenizer() function of Keras.
![POSTagger](https://user-images.githubusercontent.com/77088516/126310867-b375f2e7-fc8c-4618-a350-e2bd3e29dc8e.PNG)
- #### We have further padded the sentences to 100 words, by using pre padding and post truncating.
- #### We have also, used word embeddings from word2vec to transform the training sentences, and we have used one hot encoding for the 13 unique POS output.
- #### The word embedding created is of size 300, and we have build models with and without embedding to check their effectiveness.
- #### We have build 3 vanilla RNN models(10 epochs), with 'adam' optimizer, 'categorical_crossentropy' loss, and 'accuracy' metric:
  - #### Without fixed embeddings: validation accuracy of 0.9587, validation loss of 0.1229.
  - #### With uninitialized embeddings: validation accuracy of 0.9893, validation loss of 0.0359.
  - #### With pre-trained word2vec embeddings: validation accuracy of 0.9907, validation loss of 0.0301.
- #### We further build 3 more models with word2vec embedding(10 epochs): LSTM, GRU and Bi directional LSTM. The accuracy of all for models on test data can be se below:
![POSTagger2](https://user-images.githubusercontent.com/77088516/126312413-1683f5bc-132a-408a-98c2-58c2834c5514.PNG)[
