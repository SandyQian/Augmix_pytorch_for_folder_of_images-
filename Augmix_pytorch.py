#%%
import torch
from torchvision.transforms import v2
# from torchvision.datasets import CIFAR10
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt

#%%
#Img
#img = torch.randint(0, 256, size=(3, H, W), dtype=torch.uint8)

#Load Img with PIL package
image_path = 'E:\XJTLU intern\\2249501_XiaohanXu_Datasets\\5_class_roboflow\\photo\\train(25).JPG'
image_path = image_path.replace("\\", "/")
print(image_path)
img = Image.open(image_path)

#Define automation pipeline 
transform_pip = v2.Compose([
    # transforms.RandomHorizontalFlip(p=0.5),      # Randomly flip the image horizontally
    # v2.RandomRotation(degrees=10),       # Randomly rotate the image by up to 10 degrees
    v2.AugMix(),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True), #Convert to pytorch tensor
    # v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

#Apply transformation to data set 
aug_tensor = transform_pip(img)

#Show image
plt.imshow(aug_tensor.permute(1, 2, 0))
plt.axis('off')  # Turn off axis labels
plt.show()

#Save image 
to_PIL = v2.ToPILImage()
img_saved = to_PIL(aug_tensor)
img_saved.save('output_img.png')
