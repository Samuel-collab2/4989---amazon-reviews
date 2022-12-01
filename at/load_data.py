import pandas as pd

def load_data(file_path):

    # with open(file_path, "r") as file:
    #
    #     for line in file:
    #         print(line)
    #
    #     # print(data)
    #     pass

    data = pd.read_csv(file_path)
    print(data)

if __name__ == '__main__':
    file_path = "/Users/at/Documents/School/COMP4989/4989_AmazonProjectRawData/AppliancesCsv.csv"
    load_data(file_path)