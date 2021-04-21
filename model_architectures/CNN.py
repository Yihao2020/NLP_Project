import numpy as np
import pandas as pd

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras.utils import np_utils
from keras.layers.convolutional import Conv1D
from keras.layers import GRU, LSTM
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from sklearn import metrics

from data_preprocessing.nltk_preprocessing import preprocess_file

def embedding(train,test):
    tokenizer = Tokenizer(num_words=20000)
    tokenizer.fit_on_texts(train)
    t=len(tokenizer.word_index)+1
    trainsequences = tokenizer.texts_to_sequences(train)
    traindata = pad_sequences(trainsequences, maxlen=100)
    testsequences = tokenizer.texts_to_sequences(test)
    testdata = pad_sequences(testsequences, maxlen=100)
    return traindata, testdata,t

# Build the CNN model
def buildModel(xtrain,ytrain):
    batch_size=1000
    epochs=5
    model= Sequential()
    model.add(Embedding(20000,32,input_length=100))
    model.add(Conv1D(32,kernel_size=3,padding='same',activation='relu'))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.3))
    model.add(Conv1D(64,kernel_size=3,padding='same',activation='relu'))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.35))
    model.add(Conv1D(128,kernel_size=3,padding='same',activation='relu'))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.4))
    model.add(GRU(50,return_sequences=True))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.45))
    model.add(Dense(6,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.fit(xtrain,ytrain,batch_size=batch_size,epochs=epochs)
    model.save("toxic.h5")
    # figure out how to save the model
    return model

def main():
    # Prepare train and test data
    # df= pd.read_csv(io.BytesIO(uploaded['train.csv']),encoding='latin-1') 
    # df1 = df.head(10000)
    df = preprocess_file('data/train.csv')
    X=df['comment_text']
    classes=["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    y=df[classes].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    xtrain,xtest,vocab_size=embedding(X_train,X_test)
    CNN_model = buildModel(xtrain,y_train)
    y_pred = np.around(CNN_model.predict(xtest))

    # print results
    CNN_acc = metrics.accuracy_score(y_test,y_pred)
    print('The accuracy for CNN model is', CNN_acc)

    CNN_cm = metrics.multilabel_confusion_matrix(y_test, y_pred)
    print('The confusion matrix for each label in ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"] is:')
    print(CNN_cm)

    print(metrics.classification_report(y_test, y_pred, target_names=classes, zero_division=1))


if __name__ == "__main__":
    main()