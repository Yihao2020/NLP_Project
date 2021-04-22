import pandas as pd
from sentence_transformers import SentenceTransformer
#from data_preprocessing.nltk_preprocessing import preprocess_file
from utilities import logger, load_data_csv_to_dataframe


def generate_bert_embeddings(filename):
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')
    data = pd.read_pickle("../data/train_data_processed.pkl")

    for index, row in data.iterrows():
        text = row['comment_text']

        emb = model.encode(text)
        data.at[index, 'comment_text'] = emb

    return data


if __name__ == "__main__":
    generate_bert_embeddings("")






