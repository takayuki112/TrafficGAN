import os
import cv2
import numpy as np
from PIL import Image

class Preprocess:
    def __init__(self, image_dir, roi_dict = None, resize_dim = None, do_histogram_eq = True, do_sharpening=True, output_dir = None, verbose = True):
        """
        Preprocess the images in a folder and save them in output_dir.
        If output_dir is None, the images get saved in the same directory as the input images.
        
        Parameters:
            image_dir (str): The directory containing the images.
            roi_dict (dict): A dictionary containing the Region of Interest (ROI) for each image.
            resize_dim (tuple): A tuple (width, height) to resize the image.
            do_histogram_eq (bool): Whether to perform histogram equalization on the image.
        """
        self.image_dir = image_dir
        self.roi_dict = roi_dict if roi_dict else {}
        self.resize_dim = resize_dim
        self.do_histogram_eq = do_histogram_eq
        self.do_sharpening = do_sharpening
        self.output_dir = output_dir if output_dir else image_dir
        self.verbose = verbose
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        if verbose:
            print(f"Ready to preprocess images from {self.image_dir} and save them in {self.output_dir}")
            print(f"Number of images: {len(os.listdir(self.image_dir))}")

    def load_and_preprocess(self):
        count = 0
        for image_name in os.listdir(self.image_dir):
            
            if not image_name.lower().endswith((".png", ".jpg", ".jpeg", ".ppm")):
                continue
            
            if self.verbose:
                print(f"Processing {image_name} : {count+1}/{len(os.listdir(self.image_dir))}")
                count += 1
            
            img = Image.open(os.path.join(self.image_dir, image_name)).convert('RGB')
            
            if image_name in self.roi_dict:
                img = img.crop(self.roi_dict[image_name])
                
            if self.resize_dim:
                img = img.resize(self.resize_dim)
                
            if self.do_histogram_eq:
                img = self.hist_eq(img)
            
            if self.do_sharpening:
                img = self.sharpen_image(img)
                
            save_name = image_name.split('.')[0] + '_preprocessed.png'
            img.save(os.path.join(self.output_dir, save_name))
    
    def hist_eq(self, img):
        """
        Applies Histogram Equalization using OpenCV.
        Default Configuration: clipLimit=2.0, tileGridSize=(8, 8) - can be changed as needed.
        """
        img = np.array(img)  

        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_ycrcb[:, :, 0] = clahe.apply(img_ycrcb[:, :, 0])

        img_eq = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2RGB)
        
        return Image.fromarray(img_eq)
    
    def sharpen_image(img):
        """Apply a Laplacian sharpening filter"""
        img = np.array(img)

        sharpen_kernel = np.array([[0, -1, 0],
                                    [-1, 5, -1],
                                    [0, -1, 0]])

        img_sharpened = cv2.filter2D(img, -1, sharpen_kernel)

        return Image.fromarray(img_sharpened)
            

