import os

def load_csv_files(path):
    dir_list = os.listdir(path)
    csv_list =  filter(lambda x: x.endswith('.csv'), dir_list)
    return list(csv_list)