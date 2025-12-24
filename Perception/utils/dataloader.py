import cv2
import numpy as np
from torch.utils.data import Dataset
import torch
import random

class LaneDataset(Dataset):
    def __init__(self,imagePath,maskPath,prob=0,transforms=None):
        self.imagePath = imagePath # Array of filepaths for the input images
        self.maskPath = maskPath # Array of filepaths for the mask images
        self.transforms = transforms
        self.prob = prob

    def __len__(self):
        return len(self.imagePath)

    def __getitem__(self,idx):
        img_path = self.imagePath[idx]
        mask_path = self.maskPath[idx]

        image = cv2.imread(img_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)

        edges,edges_inv = self.find_edge_channel(image)
        
        output_image = np.zeros((gray.shape[0],gray.shape[1],3),dtype=np.uint8)
        output_image[:,:,0] = gray
        output_image[:,:,1] = edges
        output_image[:,:,2] = edges_inv
        
        if "Town" in mask_path:
            output_image, mask = self.prob_rotate(output_image,mask)
            output_image, mask = self.prob_flip(output_image,mask)
#             mask = cv2.bitwise_or(cv2.bitwise_and(mask,edges),cv2.bitwise_and(mask,edges_inv))

        if self.transforms != None:
            output_image = self.transforms(output_image)
            mask = self.transforms(mask)
    
        
        mask_binary = (mask>0).type(torch.float)
        
        # print(type(mask_binary))
        # print(type(output_image))
        return (output_image,mask_binary)

    
    def prob_flip(self,img,lbl):
        if random.random() > self.prob:
            return img,lbl
        flip_img = cv2.flip(img,1)
        flip_lbl = cv2.flip(lbl,1)
        return flip_img,flip_lbl
    
    def prob_rotate(self,img,lbl):
        if random.random() > self.prob: 
            return img,lbl
        
        rotations = [-90,-45,45,90,180]
        angle = random.choice(rotations)
        center_img = (img.shape[1]//2,img.shape[0]//2)
        center_lbl = (lbl.shape[1]//2,lbl.shape[0]//2)
        
        rotate_matrix_img = cv2.getRotationMatrix2D(center=center_img,angle=angle,scale=1)
        rotate_matrix_lbl = cv2.getRotationMatrix2D(center=center_lbl,angle=angle,scale=1)
        
        rotated_img = cv2.warpAffine(img,rotate_matrix_img,(img.shape[1],img.shape[0]))
        rotated_lbl = cv2.warpAffine(lbl,rotate_matrix_lbl,(lbl.shape[1],lbl.shape[0]))
        
        return rotated_img,rotated_lbl
    
    def find_edge_channel(self,img):
        edges_mask = np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)
        width = img.shape[1]
        height = img.shape[0]
        
        gray_im = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        # Separate into quadrants
        med1 = np.median(gray_im[:height//2,:width//2])
        med2 = np.median(gray_im[:height//2,width//2:])
        med3 = np.median(gray_im[height//2:,width//2:])
        med4 = np.median(gray_im[height//2:,:width//2])

        l1 = int(max(0,(1-0.205)*med1))
        u1 = int(min(255,(1+0.205)*med1))
        e1 = cv2.Canny(gray_im[:height//2,:width//2],l1,u1)

        l2 = int(max(0,(1-0.205)*med2))
        u2 = int(min(255,(1+0.205)*med2))
        e2 = cv2.Canny(gray_im[:height//2,width//2:],l2,u2)

        l3 = int(max(0,(1-0.205)*med3))
        u3 = int(min(255,(1+0.205)*med3))
        e3 = cv2.Canny(gray_im[height//2:,width//2:],l3,u3)

        l4 = int(max(0,(1-0.205)*med4))
        u4 = int(min(255,(1+0.205)*med4))
        e4 = cv2.Canny(gray_im[height//2:,:width//2],l4,u4)

        # Stitch the edges together
        edges_mask[:height//2,:width//2] = e1
        edges_mask[:height//2,width//2:] = e2
        edges_mask[height//2:,width//2:] = e3
        edges_mask[height//2:,:width//2] = e4
        
        edges_mask_inv = cv2.bitwise_not(edges_mask)
        
        return edges_mask, edges_mask_inv