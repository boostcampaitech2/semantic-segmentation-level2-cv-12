import pandas as pd
import os
import numpy as np
import torch
from tqdm import tqdm

def map_confidence_matrix(list_of_accuracy,csv_file):
    image_size = len(csv_file['PredictionString'])
    cls_num = len(list_of_accuracy)
    confidence_img_list = np.zeros((image_size,cls_num,256*256))
    
    for image in tqdm(range(image_size)):
        pixel_list = csv_file['PredictionString'][image].split()
        for pixel in range(len(pixel_list)):
            for clas in range(cls_num):
                if pixel_list[pixel] == str(clas):  
                    confidence_img_list[image,:,pixel] = (1-(list_of_accuracy[clas])**2)/(cls_num-1)
                    confidence_img_list[image,clas,pixel] = (list_of_accuracy[clas])**2
    
    return torch.FloatTensor(confidence_img_list)


if __name__ == '__main__':
    #csv파일 가져오기

    csvfiles_path = '/opt/ml/segmentation/csvfiles'
    new_csvfile_path = '/opt/ml/segmentation/submission_latest.csv'
    
    data = pd.read_csv('/opt/ml/segmentation/baseline_code/submission/sample_submission.csv', index_col=None)

    csv_files_name = [
                   "Fold1.csv","Fold2.csv","Fold3.csv",
                  "Fold4.csv","Fold5.csv"
                     ]
    li_of_acc = [
        [0.9866,0.5853 ,0.9263,0.6984,0.7245,0.8402,0.6893,0.7479,0.9524,0.99,0.8547],
        [0.984,0.5814,0.9368,0.8174,0.7955,0.7142,0.7588, 0.924,0.9415,0.9925,0.9185],
        [0.9848,0.6654,0.9064 ,0.6576,0.6046,0.7035,0.726,0.8515,0.94,0.7885,0.7465],
        [0.9825,0.6105,0.9297,0.8215,0.9273,0.8149,0.7025,0.8648,0.956,0.9948,0.6181],
        [0.9852,0.6058,0.9535,0.966,0.8883,0.9188,0.7787,0.9273,0.9314,0.9095,0.7799]
                ]
    
    model_weight = [0.2,0.2,0.2,0.5,0.2]
    image_size = len(pd.read_csv(os.path.join(csvfiles_path, csv_files_name[0]))['PredictionString'])
    cls_num = len(li_of_acc[0])
    print(model_weight)
    csvfiles=[0 for i in range(len(csv_files_name))]
    for i in range(len(csv_files_name)):
        csvfiles[i] = pd.read_csv(os.path.join(csvfiles_path, csv_files_name[i]))
        if i == 0:
            test_images_matrixes = model_weight[i]*torch.unsqueeze(map_confidence_matrix(li_of_acc[i],csvfiles[i]),0)
            print(f"{i}th file saving done")
        else:
            test_images_matrixes = torch.cat((test_images_matrixes,torch.unsqueeze(model_weight[i]*map_confidence_matrix(li_of_acc[i],csvfiles[i]),0)), axis=0)
            print(f"{i}th file saving done")
    
    result_score = torch.mean(test_images_matrixes,0)
    result_final = torch.argmax(result_score,axis=1)
    
    for images in tqdm(range(result_final.shape[0])):
        data = data.append({"image_id" : csvfiles[0]["image_id"][images], "PredictionString" :' '.join(str(e) for e in list(np.array(result_final[images])))}, 
                                       ignore_index=True)


    data.to_csv(new_csvfile_path, index = False)


    