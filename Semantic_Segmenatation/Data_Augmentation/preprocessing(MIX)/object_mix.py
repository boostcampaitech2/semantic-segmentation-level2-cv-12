import cv2
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
import random
import albumentations as A

json_dir = '../input/data/train.json'
save_image_dir = './dataset/image/'
save_anno_dir = './dataset/annotation/'

coco = COCO(json_dir)

def getWH(ann):
    xlist, ylist = [] ,[]
    seg = ann[0]['segmentation'][0]
    for x,y in [seg[i*2:i*2+2] for i in range(len(seg)//2)]:
        xlist.append(x)
        ylist.append(y)
    xmin=min(xlist)
    ymin=min(ylist)
    xmax=max(xlist)
    ymax=max(ylist)
    w = xmax-xmin
    h = ymax-ymin
    return xmin, ymin, w,h

def loadImage(img_id):
    img_dir = '../input/data/'+coco.loadImgs(img_id)[0]['file_name']
    img= cv2.imread(img_dir)
    return img

def mix_img(annid,target_id):
    
    anns = coco.loadAnns(annid)
    img_id = anns[0]['image_id']
    category_id = anns[0]['category_id']
    xmin, ymin, w,h= getWH(anns)
    object_img= loadImage(img_id)
    
    mask=np.zeros((512,512))
    mask[coco.annToMask(anns[0])!=0]=category_id
    
    #오브젝트 이미지에서 오브젝트만 추출, 이미지 크기 변경
    tem_mask=coco.annToMask(anns[0])
    masked_img = cv2.bitwise_and(object_img, object_img,mask=tem_mask)
    
    object_img = masked_img[ymin:ymin+h,xmin:xmin+w]
    object_mask =np.zeros((512,512),dtype='uint8')
    object_mask = mask[ymin:ymin+h,xmin:xmin+w]
    
    #타겟 이미지 로드
    target_img = loadImage(target_id)
    target_anns = coco.getAnnIds(target_id)
    target_mask = np.zeros((512,512),dtype='uint8')
    
    #타겟 이미지에 오브젝트를 합성
    random_x,random_y= random.randint(0,512-w),random.randint(0,512-h)
    for ann in coco.loadAnns(target_anns):
        target_mask[coco.annToMask(ann) == 1]=ann['category_id']
        
    mask_roi=target_mask[random_y:random_y+h,random_x:random_x+w]
    image_roi= target_img[random_y:random_y+h,random_x:random_x+w]
    
    tem_mask= tem_mask[ymin:ymin+h,xmin:xmin+w]
    back_ground = cv2.bitwise_and(image_roi, image_roi, mask =  cv2.bitwise_not(tem_mask)//255)
    target_img[random_y:random_y+h,random_x:random_x+w]=back_ground + object_img
    
    
    target_mask[random_y:random_y+h,random_x:random_x+w]=mask_roi
    tem2_mask=np.zeros((512,512))
    tem2_mask[random_y:random_y+h,random_x:random_x+w]= tem_mask
    
    target_mask[tem2_mask==1]=category_id
    
    
    return target_img,target_mask



def searchAnno():
    ids=coco.getAnnIds()
    tresh_area=3000
    new_ids =[]
    category_id=[3,4,6]
    for i in ids:
        anno=coco.loadAnns(i)
        if anno[0]['area']> tresh_area and anno[0]['category_id'] in category_id:
            new_ids.append(i)
            
    return new_ids
img_ids= coco.getImgIds()
idx=0

for i in searchAnno():
    random_imgid=random.randint(0,len(img_ids))
    img, mask = mix_img(i,random_imgid)
    img_path = save_image_dir+str(len(img_ids)+idx)+'.jpg'
    mask_path = save_anno_dir+str(len(img_ids)+idx)+'.png'
    idx+=1
    cv2.imwrite(img_path,img)
    cv2.imwrite(mask_path,mask)
    
