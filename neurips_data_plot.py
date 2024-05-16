import os
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict

def strip_extension(file_name: str) -> str:
    return os.path.splitext(file_name)[0]

def get_Kaiqing_data(directory_name: str) -> dict:
    list_of_names_of_Kaiqing_data_files = [name for name in os.listdir(directory_name) if ".pk" in name]
    number_of_data_files = len(list_of_names_of_Kaiqing_data_files)
    data_class = defaultdict(dict)
    for data_file_name in list_of_names_of_Kaiqing_data_files:
        print(data_file_name)
        data_file = open(directory_name+data_file_name, "rb")
        data = pickle.load(data_file)
        data_class[strip_extension(data_file_name)] = data
    return data_class

def main():

    neurips_data_directory_name = "neurips_data/"
    data_class = get_Kaiqing_data(neurips_data_directory_name)
    print(data_class.keys())
    pass

if __name__ == "__main__":
    main()