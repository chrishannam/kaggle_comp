import pandas as pd
from os.path import join, dirname, abspath

data_path = join(dirname(dirname(abspath(__file__))), "data")
print(data_path)
gold_data_path = join(data_path, "train.csv")
gold_data = pd.read_csv(open(gold_data_path, "r"))
print(gold_data[:10])


