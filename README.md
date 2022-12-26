# ApplicationTracker

## To do:
1. Gather data
2. Create dataset with pandas
3. Train LSTM/RNN/Transformer model using PyTorch
4. Create backend using django
5. Create frontend using react native
6. Use RESTAPI for endpoints

## Frontend Repo:
https://github.com/Alik-da-Geek/AppTrackerFrontend/tree/main

## Model
1. Create a Linear SVM model for multiclass text classification of status column
- Kaggle article explaims SVM well
  - Article describes sentiment analysis on Twitter posts
  - https://www.kaggle.com/code/mehmetlaudatekman/text-classification-svm-explained
- Implement some features that they have into our model
- Probably borrowing sklearn's built in SVC models
2. Create a NER model to extract company name from an email given status != NA
- Can probably use Stanford NLTK library
- Can build our own too if y'all want
  - Build using BERT, LSTMs, or HMMs
  - A ton of datasets from Kaggle that we can use to train

## IMAP Password: 
wataetfrwhhkjgho

## Labels
ENUMS:
SUBMITTED
INPROGRESS
OFFERED
REJECTED

FLAGS:
TEST
SCHEDULE
