
import pandas as pd
def load_data_csv_to_dataframe(file_path):
    data = pd.read_csv(file_path)
    return data

def save_data_dataframe_to_csv(data,filename = None):
    if(filename):
        data.to_csv(filename,index=False)

    else:
        data.to_csv(index=False)



def evaluate():
    pass
