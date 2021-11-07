import os
import cv2
from cv2 import imread, imwrite
import csv
import pandas as pd
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import sys
import numpy as np
from PIL import Image
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
from tqdm import tqdm
# Get im{read,write} from somewhere.
try:
    from cv2 import imread, imwrite
except ImportError:
    # Note that, sadly, skimage unconditionally import scipy and matplotlib,
    # so you'll need them if you don't have OpenCV. But you probably have them.
    from skimage.io import imread, imsave
    imwrite = imsave
    # TODO: Use scipy instead.

from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian

# import webcolors
class_colormap = [[0,0,1],
               [192,0,128],
               [0,128,192],
               [0,128,64],
               [128,0,0],
               [64,0,128],
               [64,0,192],
               [192,128,64],
               [192,192,128],
               [64,64,128],
               [128,0,192]]
CLASSES = ("Backgroud", "General trash", "Paper", "Paper pack", "Metal", "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

def create_trash_label_colormap():
    """Creates a label colormap used in Trash segmentation.
    Returns:
        A colormap for visualizing segmentation results.
    """
    colormap = np.zeros((11, 3), dtype=np.uint8)
    for inex, (r, g, b) in enumerate(class_colormap):
        colormap[inex] = [r, g, b]
    
    return colormap

def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
        label: A 2D array with integer type, storing the segmentation label.

    Returns:
        result: A 2D array with floating type. The element of the array
                is the color indexed by the corresponding element in the input label
                to the trash color map.

    Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
              map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_trash_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]

submission_path = "output772.csv"
data_root = "test/"
crf_num = 1
datas = pd.read_csv(submission_path)

for i in tqdm(range(len(datas["PredictionString"]))):

   
    fn_im = data_root+datas["image_id"][i][7]+datas["image_id"][i][-8:]
    img = imread(fn_im)
   
    

    label_img = np.reshape(np.array(list(map(int,datas["PredictionString"][i].split()))),(-1,256))
    colored_label = label_to_color_image(label_img)
    colored_label = Image.fromarray(colored_label)
    colored_label.save("__.png")
    

    # Convert the annotation's RGB color to a single 32-bit integer color 0xBBGGRR
    anno_rgb = imread("__.png").astype(np.uint32)
    anno_lbl = anno_rgb[:,:,0] + (anno_rgb[:,:,1] << 8) + (anno_rgb[:,:,2] << 16)

    # Convert the 32bit integer color to 1, 2, ... labels.
    # Note that all-black, i.e. the value 0 for background will stay 0.
    colors, labels = np.unique(anno_lbl, return_inverse=True)


    # But remove the all-0 black, that won't exist in the MAP!
    HAS_UNK = 0 in colors
    if HAS_UNK:
        # print("Found a full-black pixel in annotation image, assuming it means 'unknown' label, and will thus not be present in the output!")
        # print("If 0 is an actual label for you, consider writing your own code, or simply giving your labels only non-zero values.")
        colors = colors[1:]
    #else:
    #    print("No single full-black pixel found in annotation image. Assuming there's no 'unknown' label!")

    # And create a mapping back from the labels to 32bit integer colors.
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:,0] = (colors & 0x0000FF)
    colorize[:,1] = (colors & 0x00FF00) >> 8
    colorize[:,2] = (colors & 0xFF0000) >> 16

    # Compute the number of classes in the label image.
    # We subtract one because the number shouldn't include the value 0 which stands
    # for "unknown" or "unsure".
    n_labels = len(set(labels.flat)) - int(HAS_UNK)
    # print(n_labels, " labels", (" plus \"unknown\" 0: " if HAS_UNK else ""), set(labels.flat))

    ###########################
    ### Setup the CRF model ###
    ###########################
    # use_2d = False
    use_2d = True
    if n_labels-1 >0:
        if use_2d:
            # print("Using 2D specialized functions")

            # Example using the DenseCRF2D code
            d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)

            # get unary potentials (neg log probability)
            U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)
            d.setUnaryEnergy(U)

            # This adds the color-independent term, features are the locations only.
            d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                                normalization=dcrf.NORMALIZE_SYMMETRIC)

            # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
            d.addPairwiseBilateral(sxy=(5, 5), srgb=(10, 10, 10), rgbim=img,
                                compat=10,
                                kernel=dcrf.DIAG_KERNEL,
                                normalization=dcrf.NORMALIZE_SYMMETRIC)
        else:
            # print("Using generic 2D functions")

            # Example using the DenseCRF class and the util functions
            d = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)

            # get unary potentials (neg log probability)
            # if n_labels-1 > 0:
            U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)
            d.setUnaryEnergy(U)

            # This creates the color-independent features and then add them to the CRF
            feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
            d.addPairwiseEnergy(feats, compat=3,
                                kernel=dcrf.DIAG_KERNEL,
                                normalization=dcrf.NORMALIZE_SYMMETRIC)

            # This creates the color-dependent features and then add them to the CRF
            feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                            img=img, chdim=2)
            d.addPairwiseEnergy(feats, compat=10,
                                kernel=dcrf.DIAG_KERNEL,
                                normalization=dcrf.NORMALIZE_SYMMETRIC)


        ####################################
        ### Do inference and compute MAP ###
        ####################################

        # Run five inference steps.
        Q = d.inference(crf_num)

        # Find out the most probable class for each pixel.
        MAP = np.argmax(Q, axis=0)

        # Convert the MAP (labels) back to the corresponding colors and save the image.
        # Note that there is no "unknown" here anymore, no matter what we had at first.
        MAP = colorize[MAP,:]
        crf_img = MAP.reshape(img.shape)
        crf_img = crf_img.tolist()

        crf_list = []
        reversed_color = [i[::-1] for i in class_colormap]

        for i in range(len(crf_img)):
            for j in range(len(crf_img[i])):
                for k in range(len(reversed_color)):
                    if crf_img[i][j] ==reversed_color[k]:
                        
                        crf_list.append(str(k))
                    
        crf_string = ' '.join(crf_list)
        datas["PredictionString"][i] = crf_string
datas.to_csv(submission_path[:-4]+"_denseCRF"+"_"+str(crf_num)+".csv")