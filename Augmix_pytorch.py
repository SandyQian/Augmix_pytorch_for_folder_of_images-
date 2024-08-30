#%%
import torch
from torchvision.transforms import v2
# from torchvision.datasets import CIFAR10
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import file_management import list_of_fullpath, list_of_filenames, back_to_forward_slash

def load_img(img_path):
    #Load Img with PIL package
    '''A function that load image with PIL package'''
    img_path = img_path("\\", "/")
    img = Image.open(image_path)
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
    img_saved.save(outdir + './' + save_name)
    # img_saved.save('./output_img.png')
    return 


#Upgrade the functions for operation on a list of files 
def load_imgs(img_path_list):
    '''Loading images from a folder'''
    list_of_img = []
    for img_path in img_path_list:
        list_of_img.append(load_img(img_path))
    return list_of_img

def apply_aug_on_list(pipline, img_list):
    aug_tensor_list =[]
    for item in img_list:
        aug_tensor_list.append(pipline(img_list))
    return aug_tensor_list

def save_tensor_as_img(outdir, aug_tensor_list, save_name_list):
    '''A function that saves a list of tensors as png images'''
    for i in range(0, len(aug_tensor_list)):
        save_img(aug_tensor_list[i], outdir, save_name_list[i])
    return 



#img = torch.randint(0, 256, size=(3, H, W), dtype=torch.uint8)
# image_path = 'E:\XJTLU intern\\2249501_XiaohanXu_Datasets\\5_class_roboflow\\photo\\train(25).JPG'

input_dir = "E:\\XJTLU intern\\2249501_XiaohanXu_Datasets\\5_class_roboflow\photo"
output_dir = "E:/XJTLU intern/2249501_XiaohanXu_Datasets/AugMix_5class"

all_file_paths = list_of_fullpath(input_dir)
all_file_name = list_of_filenames(input_dir)

img_list = load_imgs(all_file_paths)
aug_pipline = AugMix_pip()
aug_tensors = apply_aug_on_list(aug_pipline, img_list)
save_tensor_as_img(output_dir, aug_tensors, all_file_name)

#Apply transformation to data set 
# aug_tensor = transform_pip(img)

#Show image
# plt.imshow(aug_tensor.permute(1, 2, 0))
# plt.axis('off')  # Turn off axis labels
# plt.show()

# #Save image 
# to_PIL = v2.ToPILImage()
# img_saved = to_PIL(aug_tensor)
# img_saved.save('output_img.png')
