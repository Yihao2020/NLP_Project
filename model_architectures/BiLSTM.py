"""
BiLSTM model with input as vector embedding (SBERT) , 1 BiLSTM layer and 1 FC Dense Layer and 2 BiLSTM layer and 1 dense layer
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Bidirectional, Input, Embedding, LSTM, TimeDistributed
from keras.regularizers import l2
from keras.losses import mean_squared_error
from keras import Model
import pandas as pd
import numpy as np
import keras
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from keras.callbacks import EarlyStopping
import matplotlib as mpl
import matplotlib.pyplot as plt

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


class BiLSTM1:
    """
    This class implement 1layer Bilstm + 1 FC dense layer model
    """

    def __init__(self, epochs: int = 10):
        self.model = self._build_model()
        self.epochs = epochs
        self.X_test = []
        self.y_test = []
        self.history = ""

    def _build_model(self):
        """
        Function to build 1 layer Bilstm model
        """
        input = Input(shape=(100, 768), dtype="float32")
        bilstm1 = Bidirectional(LSTM(70, return_sequences=True))(input)
        # bilstm2 = Bidirectional(LSTM(10))(bilstm1)
        flatten = Flatten()(bilstm1)
        output = Dense(6, activation='softmax')(flatten)
        model = Model(inputs=input, outputs=output, name="bilstm_model")
        METRICS = [
            keras.metrics.TruePositives(name='tp'),
            keras.metrics.FalsePositives(name='fp'),
            keras.metrics.TrueNegatives(name='tn'),
            keras.metrics.FalseNegatives(name='fn'),
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
        ]

        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=METRICS)
        print(model.summary())
        return model

    def train(self, data_file):
        """
        Function to train the model .
        Input: Embedding vector
        """
        data = pd.read_pickle(data_file)
        X = data["comment_text"].values
        Y = data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values
        X = np.asarray(X)
        X = keras.preprocessing.sequence.pad_sequences(X, maxlen=100)
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, Y, test_size=0.2, random_state=42,
                                                                      stratify=Y[:, 3])

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42,
                                                          stratify=y_train[:, 3])

        # class_weights = class_weight.compute_class_weight('balanced',
        #                                                 np.unique(y_train[:,3]),
        #                                                y_train[:,3])

        callback = EarlyStopping(monitor='val_prc', verbose=1, patience=4, mode='max', restore_best_weights=True)
        self.history = self.model.fit(X_train, y_train, epochs=self.epochs, callbacks=[callback],
                                      validation_data=(X_val, y_val))

        self.model.save("BiLSTM_Model")

    def evaluate(self):
        """
        Function to Evaluate
        """
        predictions = self.model.predict(self.X_test)
        rounded = []
        for pred in predictions:
            rounded.append([round(x) for x in pred])

        rounded = np.asarray(rounded)
        print("Evalution Results on Test Data")
        self.model.evaluate(self.X_test, self.y_test)
        confusion_matrix = multilabel_confusion_matrix(self.y_test, rounded)
        print(confusion_matrix)
        target_names = ['toxic', 'severe toxic', 'obscene', 'threat', 'insult', 'identity hate']
        print(classification_report(self.y_test, rounded, target_names=target_names))

    def plot_metrics(self):
        """
        Function to plot metrics
        """
        history = self.history
        metrics = ['loss', 'prc', 'precision', 'recall']
        for n, metric in enumerate(metrics):
            name = metric.replace("_", " ").capitalize()
            plt.subplot(2, 2, n + 1)
            plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
            plt.plot(history.epoch, history.history['val_' + metric],
                     color=colors[0], linestyle="--", label='Val')
            plt.xlabel('Epoch')
            plt.ylabel(name)
            if metric == 'loss':
                plt.ylim([0, plt.ylim()[1]])
            elif metric == 'auc':
                plt.ylim([0.8, 1])
            else:
                plt.ylim([0, 1])

            plt.legend()
            plt.savefig('line_plot_1bilstm_' + str(metric))


############


class BiLSTM2:
    """
    This class implements 2 layer BiLSTM + 1 layer FC dense Network
    """

    def __init__(self, epochs: int = 10):
        self.model = self._build_model()
        self.epochs = epochs
        self.X_test = []
        self.y_test = []
        self.history = ""

    def _build_model(self):
        """
        Function to build 3 layer Bilstm model
        """
        input = Input(shape=(100, 768), dtype="float32")
        bilstm1 = Bidirectional(LSTM(70, return_sequences=True))(input)
        bilstm2 = Bidirectional(LSTM(10))(bilstm1)
        flatten = Flatten()(bilstm2)
        output = Dense(6, activation='softmax')(flatten)
        model = Model(inputs=input, outputs=output, name="bilstm_model")
        METRICS = [
            keras.metrics.TruePositives(name='tp'),
            keras.metrics.FalsePositives(name='fp'),
            keras.metrics.TrueNegatives(name='tn'),
            keras.metrics.FalseNegatives(name='fn'),
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
        ]

        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=METRICS)
        print(model.summary())
        return model

    def train(self, data_file):
        data = pd.read_pickle(data_file)
        X = data["comment_text"].values
        Y = data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values
        X = np.asarray(X)
        X = keras.preprocessing.sequence.pad_sequences(X, maxlen=100)
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, Y, test_size=0.2, random_state=42,
                                                                      stratify=Y[:, 3])

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42,
                                                          stratify=y_train[:, 3])

        # class_weights = class_weight.compute_class_weight('balanced',
        #                                                 np.unique(y_train[:,3]),
        #                                                y_train[:,3])

        callback = EarlyStopping(monitor='val_prc', verbose=1, patience=4, mode='max', restore_best_weights=True)
        self.history = self.model.fit(X_train, y_train, epochs=self.epochs, callbacks=[callback],
                                      validation_data=(X_val, y_val))

        self.model.save("BiLSTM2_Model")

    def evaluate(self):
        predictions = self.model.predict(self.X_test)
        rounded = []
        for pred in predictions:
            rounded.append([round(x) for x in pred])

        rounded = np.asarray(rounded)
        print("Evalution Results on Test Data")
        self.model.evaluate(self.X_test, self.y_test)
        confusion_matrix = multilabel_confusion_matrix(self.y_test, rounded)
        print(confusion_matrix)
        target_names = ['toxic', 'severe toxic', 'obscene', 'threat', 'insult', 'identity hate']
        print(classification_report(self.y_test, rounded, target_names=target_names))

    def plot_metrics(self):
        history = self.history
        metrics = ['loss', 'prc', 'precision', 'recall']
        for n, metric in enumerate(metrics):
            name = metric.replace("_", " ").capitalize()
            plt.subplot(2, 2, n + 1)
            plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
            plt.plot(history.epoch, history.history['val_' + metric],
                     color=colors[0], linestyle="--", label='Val')
            plt.xlabel('Epoch')
            plt.ylabel(name)
            if metric == 'loss':
                plt.ylim([0, plt.ylim()[1]])
            elif metric == 'auc':
                plt.ylim([0.8, 1])
            else:
                plt.ylim([0, 1])

            plt.legend()
            plt.savefig('line_plot_bilstm2_' + str(metric))