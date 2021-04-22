

import pandas as pd
import logging
import sys

def load_data_csv_to_dataframe(file_path):
    data = pd.read_csv(file_path)
    return data

def save_data_dataframe_to_csv(data,filename = None):
    if(filename):
        data.to_csv(filename,index=False)

    else:
        data.to_csv(index=False)


def get_console_handler():
   console_handler = logging.StreamHandler(sys.stdout)
   console_handler.setFormatter(logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s"))
   return console_handler

logger = logging.getLogger("NLP_PROEJCT")
logger.setLevel(logging.INFO) # better to have too much log than not enough
logger.addHandler(get_console_handler())
logger.propagate = False
