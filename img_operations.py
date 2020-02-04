import torch
from torch import nn
from torchvision import transforms



def compose_2x2_image(img1, img2, img3, img4):
    image_upp = torch.cat((img1, img2), dim=3)
    image_down = torch.cat((img3, img4), dim=3)
    image = torch.cat((image_upp, image_down), dim=2)
    return image

def compose_2x1_image(img_arr1, img_arr2):
    return torch.cat((img_arr1, img_arr2), dim=2)

def compose_Xx2_image(img_list1, img_list2):
    image_upp = torch.cat(img_list1, dim=3)
    image_down = torch.cat(img_list2, dim=3)
    image = torch.cat((image_upp, image_down), dim=2)
    return image


def compose_3x2_image(img1, img2, img3, img4, img5, img6):
    image_upp = torch.cat((img1, img2, img3), dim=3)
    image_down = torch.cat((img4, img5, img6), dim=3)
    image = torch.cat((image_upp, image_down), dim=2)
    return image

def compose_4x2_image(img1, img2, img3, img4, img5, img6, img7, img8):
    image_upp = torch.cat((img1, img2, img3, img4), dim=3)
    image_down = torch.cat((img5, img6, img7, img8), dim=3)
    image = torch.cat((image_upp, image_down), dim=2)
    return image


def min_max_normalization(tensor, min_value, max_value):
    min_tensor = tensor.min()
    tensor = (tensor - min_tensor)
    max_tensor = tensor.max()
    tensor = tensor / max_tensor
    tensor = tensor * (max_value - min_value) + min_value
    return tensor

def diff(img_pre, img_post):
    diff = torch.abs((img_pre - img_post))
    diff = toGray(diff)
    #diff = nn.Threshold(0.05, 0)(diff)
    return diff

def toGray(img):
    return (torch.sum(img, dim=1, keepdim=True) / 3).repeat(1, 3, 1, 1)


img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda tensor: min_max_normalization(tensor, 0, 1))
])

