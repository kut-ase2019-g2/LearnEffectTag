from clarifai.rest import ClarifaiApp
import pandas as pd
import os
import sys
import glob
from tqdm import tqdm
import collections

api_key='key'
modelname = 'cui_test'
effect_tag = ['KiraKira', 'Heart', 'Notes', 'Fire', 'Central line', 'Rain',
            'Thunder', 'Lens flare', 'Comicalize', 'Blur'] #10まで
csv_name_list = ['a.csv','b.csv','c.csv','d.csv','e.csv',]
df_list = []

# 学習用画像と解答ラベル
image_list = glob.glob('img/*')

for i in range(len(csv_name_list)):
    df = pd.read_csv(csv_name_list[0], names=('tag',), index_col=0)
    df_list.append(df)


# API setup
app=ClarifaiApp(api_key=api_key)

# model make or load
search_model = app.models.search(modelname)
if modelname != None:
    model = app.models.get(modelname)
else:
    model = app.models.create(modelname, concepts=effect_tag)

# image uploaded
update_img = 0
tag0_img = 0
tag1_img = 0
tag2_img = 0

#pbar = tqdm(total=total_learn_batch, desc='Update_img')
for i in range(len(image_list)):
    image_path = image_list[i]
    for j in range(len(df_list)):
        df = df_list[i]
        tag = dfdf.at[image_path, 'tag']
    c = collections.Counter(tag)
    tag_list = []
    for j in range(len(c)):
        if c[c[j][0]][1] >= 2:
            tag_list.append(c[j][0])
    app.inputs.create_image_from_filename(filename=image_path, concepts=tag_list)
    if len(tag_list) == 0:
        tag0_img += 1
    elif len(tag_list) == 1:
        tag1_img += 1
    elif len(tag_list) == 2:
        tag2_img += 1
    print('Now Update img %d/%d : tag %d' %(i, len(image_list), len(tag_list) ))
    #pbar.update(1)  # プロセスバーを進行

print('===============================')
print('update img : ' + str(update_img))
print(' tag0 img : ' + str(update_img))
print(' tag1 img : ' + str(update_img))
print(' tag2 img : ' + str(update_img))
print('===============================')

# model train
print('train start')
model.train()
print('train end')
