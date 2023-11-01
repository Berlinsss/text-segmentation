import pandas as pd

df = pd.read_excel('/Users/berlin/CUHK/text-segmentation/sample labeled by claude.xlsx')

path_prefix = '/Users/berlin/CUHK/text-segmentation/input_data'

for index, row in df.iterrows():

    second_column_data = str(row.iloc[1])

    path = path_prefix + '/' + str(index) + '.txt'

    with open(path, 'w+') as file:
        file.write(second_column_data)