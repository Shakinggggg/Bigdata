import pandas as pd

clu_list = []
data = pd.read_csv('clustering.csv')

selected_column = ['X', 'Y', 'Cluster']

X = data[selected_column]

random_data = X.sample(n=200, random_state=42)
for index, row in random_data.iterrows():
    clu_dist = {
        "x/0": row['X'],
        "y/0": row['Y'],
        "s/0": int(row['Cluster'])
    }
    clu_list.append(clu_dist)

print(clu_list)