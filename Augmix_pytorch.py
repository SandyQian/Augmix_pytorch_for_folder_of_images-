#%%
import torch
from torchvision.transforms import v2
# from torchvision.datasets import CIFAR10
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from file_management import list_of_fullpath, list_of_filenames, back_to_forward_slash
import argparse
import os

def load_img(img_path):
    #Load Img with PIL package
    '''A function that load image with PIL package'''
    img_path = img_path.replace("\\", "/")
    img = Image.open(img_path)
    return img

def AugMix_pip():
    '''A function that constructs transformation (AugMix) pipeline '''
    #Automation pipeline 
    transform_pip = v2.Compose([
        v2.AugMix(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True), #Convert to pytorch tensor
        # v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform_pip

def apply_augmix(pipline, img):
    '''A function that returns aug_image'''
    aug_image = pipline(img)
    return aug_image

def save_img(aug_tensor, outdir, save_name):
    '''A function that saves tensor as png image'''
    to_PIL = v2.ToPILImage()
    img_saved = to_PIL(aug_tensor)
    img_saving_path = outdir + './' + save_name
    img_saved.save(img_saving_path)
    print(save_name + ' is saved.')
    # img_saved.save('./output_img.png')
    return 

def process_one_img(in_dir, out_dir, pipeline, save_name):
    img = load_img(in_dir)
    aug_tensor = pipeline(img)
    saved_img = save_img(aug_tensor, output_dir, save_name)
    return 

def process_all_img(in_folder, out_folder, pipeline, save_name_list):
    for i in range(0, len(save_name_list)):
        in_file = in_folder[i]
        save_name = save_name_list[i]
        process_one_img(in_file, out_folder, pipeline, save_name)
    return 


def main(input_dir, output_dir):
    all_file_paths = list_of_fullpath(input_dir, extension='.JPG')
    all_file_name = list_of_filenames(input_dir, extension='.JPG')

    aug_pipline = AugMix_pip()
    process_all_img(all_file_paths, output_dir, aug_pipline, all_file_name)
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images with AugMix pipeline.")
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()

#$indir = "E:\XJTLU intern\2249501_XiaohanXu_Datasets\5_class_roboflow\photo"
#$outdir = "E:\XJTLU intern\2249501_XiaohanXu_Datasets\AugMix_5class"
#python .\AugMix.py  indir outdir


#Show image
# plt.imshow(aug_tensor.permute(1, 2, 0))
# plt.axis('off')  # Turn off axis labels
# plt.show()

# #Save image 
# to_PIL = v2.ToPILImage()
# img_saved = to_PIL(aug_tensor)
# img_saved.save('output_img.png')

