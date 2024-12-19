import numpy as np
from scipy.io import loadmat
import cv2
import json
import os
import pickle

dimx = 220
dimy = 220

def make_dataset():
    
    i = 0
    reqd_train_images = 11000
    curr_train_images = 0
    labels = {} 
    
    images_name = []

    for t in os.listdir('./images/'):
        if '.jpg' in t:
            images_name.append(t)      

    data = open("datampii.json")
    
    joints = ['pelvis', 'thorax', 'upper_neck', 'head_top', 'r_ankle', 'r_knee', 
              'r_hip', 'l_hip', 'l_knee', 'l_ankle', 'r_wrist', 'r_elbow', 
              'r_shoulder', 'l_shoulder', 'l_elbow', 'l_wrist']
   
    for line in data:
        dicta = json.loads(line)
        
        # Check if filename in directory
        if dicta['filename'] not in images_name:
            continue
        
        image = cv2.imread("./images/" + dicta['filename'], 1)     
        
        x = []
        y = []
        
        for joint in joints:
            x.append(dicta['joint_pos'][joint][0])
            y.append(dicta['joint_pos'][joint][1])

        if any(coordinatex < 0 for coordinatex in x) or any(coordinatey < 0 for coordinatey in y):
            print("skipping image due to negative coordinate")
            continue  
            
        if any(coordinatex >= image.shape[1] for coordinatex in x) or any(coordinatey >= image.shape[0] for coordinatey in y):
            print("skipping image due to out of bound coordinate")
            continue 
        
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)   
        
        x_max = int(x.max())
        x_min = int(x.min())
        y_max = int(y.max())
        y_min = int(y.min())
  
        image_crop = image[max(y_min - 50, 0):min(y_max + 50, image.shape[0] - 1), 
                           max(x_min - 50, 0):min(x_max + 50, image.shape[1] - 1), :]
        
        # Translated coordinates
        if x_min - 50 > 0:
            x_trans = x - (x_min - 50) 
        else:
            x_trans = x
            
        if y_min - 50 > 0:
            y_trans = y - (y_min - 50) 
        else:
            y_trans = y
        
        resz_img = cv2.resize(image_crop, (dimx, dimy), interpolation=cv2.INTER_LINEAR)
        
        x_trans = x_trans * dimx / image_crop.shape[1]
        y_trans = y_trans * dimy / image_crop.shape[0] 
        
        # Normalize between -1 to 1
        x_trans = ((x_trans / dimx) - 0.5) * 2
        y_trans = ((y_trans / dimy) - 0.5) * 2        
        
        label_i = {'x': x_trans, 'y': y_trans}
        
        print("image_saved:", i)
        i += 1
        
        # Try-except for images with multiple labelled humans
        try:
            check_ = labels[dicta['filename']]
            labels[str(i) + dicta['filename']] = label_i 
            
            # Train valid split
            if curr_train_images < reqd_train_images:
                cv2.imwrite("./train/" + str(i) + dicta['filename'], resz_img)        
            else:
                cv2.imwrite("./valid/" + str(i) + dicta['filename'], resz_img)
        
        except:    
            labels[dicta['filename']] = label_i  
            
            # Train valid split
            if curr_train_images < reqd_train_images:
                cv2.imwrite("./train/" + dicta['filename'], resz_img)        
            else:
                cv2.imwrite("./valid/" + dicta['filename'], resz_img) 
        
        curr_train_images += 1
        
    return labels  

labels = make_dataset()
# Save label dictionary
outfile = open('labelsdict_mpii', 'wb')
pickle.dump(labels, outfile)
outfile.close()
