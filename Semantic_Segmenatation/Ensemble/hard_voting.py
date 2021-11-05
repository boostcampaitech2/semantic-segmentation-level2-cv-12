import pandas as pd
import os
import numpy as np 

csvfiles_path = '/opt/ml/segmentation/csvfiles4'
new_csvfile_path = '/opt/ml/segmentation/submission_latest.csv' 

csvfile_list = os.listdir(csvfiles_path)
csvfiles = [0 for i in range(len(csvfile_list))]
for i in range(len(csvfiles)):
    csvfiles[i] = pd.read_csv(os.path.join(csvfiles_path, csvfile_list[i]))

mostbits=[]
data = pd.read_csv(new_csvfile_path)

image_size = len(csvfiles[0]['PredictionString'])
file_size = len(csvfiles[0]['PredictionString'][0])

def most_frequent(data):
    return max(data, key=data.count)

print(csvfile_list)
for images in range(image_size):

    mostbits=[]
    a = csvfiles[0]['PredictionString'][images].split()
    b = csvfiles[1]['PredictionString'][images].split()
    c = csvfiles[2]['PredictionString'][images].split()
    d = csvfiles[3]['PredictionString'][images].split()
    e = csvfiles[4]['PredictionString'][images].split()
    f = csvfiles[5]['PredictionString'][images].split()
    g = csvfiles[6]['PredictionString'][images].split()

    for j in range(65536):
        mostbit = [a[j],b[j],c[j],d[j],e[j],f[j],g[j]]
        if mostbit[0] == ' ':

        else:
            mostbit = list(map(int, mostbit))
            mostbit = most_frequent(mostbit)
            mostbits.append(mostbit)

    new_string = ' '.join(str(_) for _ in(mostbits))

    data["PredictionString"][images] = new_string

data.to_csv(new_csvfile_path, index = False)


