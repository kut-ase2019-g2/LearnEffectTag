import pandas as pd
import csv
import pandas as pd
import os
import sys
import glob
from tqdm import tqdm
import collections

csv_folder = 'EffectTagData'
csv_name_list = ['001.csv', '002.csv', '003.csv', '004.csv', '005.csv']
out_csv = 'result.csv'
df_list = []

image_list = glob.glob('img/*')

for i in range(len(csv_name_list)):
    df = pd.read_csv(csv_folder+'/'+csv_name_list[i],
     index_col=0, encoding = "shift-jis", sep=', ', engine = 'python')
    df_list.append(df)

for i in range(len(image_list)):
    image_list[i] = image_list[i].replace('\\', '/')

result_df = pd.DataFrame(index = image_list, columns = csv_name_list)



for i in range(len(image_list)):
    image_path = image_list[i]
    for j in range(len(df_list)):
        df = df_list[j]
        if image_path in df.index:
            result_df.at[image_path, csv_name_list[j]] = df.at[image_path, 'Tag name']
        else:
            result_df.at[image_path, csv_name_list[j]] = 'non'
    result_df.to_csv(out_csv)
