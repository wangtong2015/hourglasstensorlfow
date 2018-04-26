import configparser
from datagen import DataGenerator
import inference as Inference
from train_launcher import process_config
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import time
from skimage import transform
import scipy.misc as scm
from PIL import Image, ImageDraw,ImageFont
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--CPU",help="trained by cpu",action = "store_true")
args = parser.parse_args()
if args.CPU:
	from hourglass_tiny_CPU import HourglassModel
else:
	from hourglass_tiny import HourglassModel
EXT = ['.jpg','.png','JPG','.PNG']
class TestData:
    def __init__(self, joints_name = None, img_dir=None,output_dir=None):
        """ Initializer
        Args:
            joints_name			: List of joints condsidered
            img_dir				: Directory containing every images
            train_data_file		: Text file with training set data
            remove_joints		: Joints List to keep (See documentation)
        """
        if joints_name == None:
            self.joints_list = ['r_anckle', 'r_knee', 'r_hip', 'l_hip', 'l_knee', 'l_anckle', 'pelvis', 'thorax', 'neck', 'head', 'r_wrist', 'r_elbow', 'r_shoulder', 'l_shoulder', 'l_elbow', 'l_wrist']
        else:
            self.joints_list = joints_name
        self.toReduce = False
        self.output_dir = output_dir
        self.letter = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N']
        self.img_dir = img_dir
        self.images = []
        for filename in os.listdir(img_dir):
            ext = os.path.splitext(filename)[1]
            if ext in EXT:
                self.images.append(filename)
        self.imnp = {}
        self.im = {}
        self.font = ImageFont.truetype("consola.ttf", 12, encoding="unic")
        for name in self.images:
            self.im[name],self.imnp[name] = self.getImg(name)
    def _crop_img(self, img, padding, crop_box):
        """ Given a bounding box and padding values return cropped image
        Args:
            img			: Source Image
            padding	: Padding
            crop_box	: Bounding Box
        """
        img = np.pad(img, padding, mode = 'constant')
        max_lenght = max(crop_box[2], crop_box[3])
        img = img[crop_box[1] - max_lenght //2:crop_box[1] + max_lenght //2, crop_box[0] - max_lenght // 2:crop_box[0] + max_lenght //2]
        return img
    def _crop(self, img, hm, padding, crop_box):
        """ Given a bounding box and padding values return cropped image and heatmap
        Args:
            img			: Source Image
            hm			: Source Heat Map
            padding	: Padding
            crop_box	: Bounding Box
        """
        img = np.pad(img, padding, mode = 'constant')
        hm = np.pad(hm, padding, mode = 'constant')
        max_lenght = max(crop_box[2], crop_box[3])
        img = img[crop_box[1] - max_lenght //2:crop_box[1] + max_lenght //2, crop_box[0] - max_lenght // 2:crop_box[0] + max_lenght //2]
        hm = hm[crop_box[1] - max_lenght //2:crop_box[1] + max_lenght//2, crop_box[0] - max_lenght // 2:crop_box[0] + max_lenght // 2]
        return img, hm
    def _relative_joints(self, box, padding, joints, to_size = 64):
        """ Convert Absolute joint coordinates to crop box relative joint coordinates
        (Used to compute Heat Maps)
        Args:
            box			: Bounding Box 
            padding	: Padding Added to the original Image
            to_size	: Heat Map wanted Size
        """
        new_j = np.copy(joints)
        max_l = max(box[2], box[3])
        new_j = new_j + [padding[1][0], padding[0][0]]
        new_j = new_j - [box[0] - max_l //2,box[1] - max_l //2]
        new_j = new_j * to_size / (max_l + 0.0000001)
        return new_j.astype(np.int32)
    # ---------------------------- Image Reader --------------------------------			
            
    def getImg(self,name):  # 预处理
        if name[-1] in self.letter:
            name = name[:-1]
        im = Image.open(os.path.join(self.img_dir, name)).resize((256,256))
        imnp = np.array(im, np.uint8)
        return (im,imnp)
    def plotImg(self,name,preJoint = None):
        draw =ImageDraw.Draw(self.im[name])
        if preJoint is not None:
            width,height = self.im[name].size
            for i in range(len(self.joints_list)):
                y,x= preJoint[i]
                if(x!=-1 and y!=-1):
                    radius = 3
                    x = width * x/256.0
                    y = height * y/256.0
                    rect = (x-radius,y-radius,x+radius,y+radius)
                    textCoor = (x+3*radius,y)
                    draw.ellipse(rect,'red','black')
                    draw.text(textCoor,self.joints_list[i],'red',self.font)
        self.im[name].save(os.path.join(self.output_dir,name))
        
if __name__ == '__main__':
    params = process_config('config.cfg') 
    testData = TestData(params['joint_list'], params['test_img_directory'],params['image_output'])
    inference = Inference.Inference( config_file = 'config.cfg', model = './results/models/template_model/hg_refined_tiny_200', yoloModel = './results/models/template_model/YOLO_small.ckpt')
    preJoints = {}
    for name in testData.images:
        preJoints[name] = inference.predictJoints(testData.imnp[name])
        testData.plotImg(name,preJoints[name])


    
