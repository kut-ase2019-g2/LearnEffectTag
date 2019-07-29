import pandas as pd
import csv
import pandas as pd
import os
import sys
import glob
from tqdm import tqdm
import collections

    # u'キラキラ':'KiraKira',
    # u'ハート':'Heart',
    # u'音符':'Notes',
    # u'炎':'Fire',
    # u'集中線':'Central_line',
    # u'波紋':'Rain',
    # u'電撃':'Thunder',
    # u'レンズフレア':'Lens_flare',
    # u'漫画風':'Comicalize',
    # u'振動':'Blur',
    # u'コントラスト':'Change_contrast',
    # u'拡大':'Zoom_in',
    # u'縮小':'Zoom_out'
    # 'KiraKira': 107,
    # 'Change_contrast': 48,
    # 'Lens_flare': 47,
    # 'Rain': 45,
    # 'Central_line': 39,
    # 'Zoom_in': 39,
    # 'Heart': 37,
    # 'Fire': 37,
    # 'Notes': 30,
    # 'Zoom_out': 19,
csvfile = 'result.csv'
outfile = 'SetTag.csv'
# effect_tag = ['KiraKira', 'Change_contrast', 'Lens_flare', 'Rain', 'Central_line',
#     'Zoom_in', 'Heart', 'Fire', 'Notes', 'Zoom_out']
effect_tag = ['KiraKira', 'Change_contrast', 'Lens_flare', 'Rain', 'Central_line',
    'Zoom_in', 'Heart', 'Notes', 'Zoom_out']
columns_list = ['tag1', 'tag2']
val_list = [0 for i in range(len(effect_tag))]
count_tag1 = dict(zip(effect_tag, val_list))
count_tag2 = dict(zip(effect_tag, val_list))

df = pd.read_csv(csvfile, index_col=0)
result_df = pd.DataFrame(index = df.index, columns = columns_list)
# image_list = glob.glob('img/*')
# for i in range(len(image_list)):
#     image_list[i] = image_list[i].replace('\\', '/')
tag0_img = 0
tag1_img = 0
tag2_img = 0

for i in range(len(df)):
    tag_list = []
    double_tag_list = []
    for j in range(len(df.columns)):
        tag = df.iat[i,j]
        if tag in effect_tag:
            if tag in tag_list:
                if tag not in double_tag_list:
                    count_tag2[tag] += 1
                    double_tag_list.append(tag)
            else:
                count_tag1[tag] += 1
                tag_list.append(tag)
    for j in [0,1]:
        if j+1 <= len(double_tag_list):
            result_df.iat[i,j] = double_tag_list[j]
        else:
            result_df.iat[i,j] = 'None'
    if len(double_tag_list) == 0:
        tag0_img += 1
    elif len(double_tag_list) == 1:
        tag1_img += 1
    elif len(double_tag_list) == 2:
        tag2_img += 1
# print(result_df)
result_df.to_csv(outfile)
count_tag1 = dict(sorted(count_tag1.items(), key=lambda x:x[1], reverse=True))
count_tag2 = dict(sorted(count_tag2.items(), key=lambda x:x[1], reverse=True))
print('tag0_img: %d' %tag0_img)
print('tag1_img: %d' %tag1_img)
print('tag2_img: %d' %tag2_img)
print(count_tag1)
print(count_tag2)
