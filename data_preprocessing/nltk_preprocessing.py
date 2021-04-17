from utilities import load_data_csv_to_dataframe
import nltk
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from utilities import  save_data_dataframe_to_csv
nltk.download('punkt')
nltk.download('stopwords')


def preprocess_file(filename):
    data = load_data_csv_to_dataframe(filename)
    for index, row in data.iterrows():
        text = row['comment_text']
        text = remove_puntucation(text)
        text = tokennize(text)
        text = fileter_stopwords(text)
        row['comment_text'] = text
        data.at[index,'comment_text'] = text

    return data


def remove_puntucation(sentence):
    sentence = "".join([char for char in sentence if char not in string.punctuation])
    return sentence


def tokennize(sentece):

    words = word_tokenize(sentece)
    return words

def fileter_stopwords(words):
    stop_words = stopwords.words('english')
    words = [word for word in words if word not in stop_words]
    return words

def stemming(words):
    porter = PorterStemmer()
    words = [porter.stem(word) for word in words]
    return words


