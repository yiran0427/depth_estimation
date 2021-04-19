import torch
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as tF

class ToTensor(object):
    def __init__(self):
        self.transform = transforms.ToTensor()
        
    def __call__(self, sample):
        left_img = sample['left_image']
        sample['left_image'] = self.transform(left_img)
        
        if 'right_image' in sample:
            right_img = sample['right_image']
            sample['right_image'] = self.transform(right_img)
            
        return sample


class ToRandomFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob
        
    def __call__(self, sample):
        if random.random() < self.prob:
            return sample
        
        left_img = sample['left_image']
        sample['left_image'] = tF.hflip(left_img)
        
        if 'right_image' in sample:
            right_img = sample['right_image']
            sample['right_image'] = tF.hflip(right_img)

        return sample


class ToResizeImage(object):
    def __init__(self, size=(256, 512)):
        self.size = size
        
    def __call__(self, sample):
        left_img = sample['left_image']
        sample['left_image'] = tF.resize(left_img, self.size)
        
        if 'right_image' in sample:
            right_img = sample['right_image']
            sample['right_image'] = tF.resize(right_img, self.size)

        return sample


class AugumentImagePair(object):
    def __init__(self, prob=0.5):
        self.prob = prob
        
    def __call__(self, sample):
        if 'right_image' not in sample:
            return sample

        left_img = sample['left_image']
        right_img = sample['right_image']
        if random.random() < self.prob:
            # shift gamma
            random_gamma = random.uniform(0.8, 1.2)
            left_img_aug = left_img ** random_gamma
            right_img_aug = right_img ** random_gamma
            
            # shift brightness
            random_brightness = random.uniform(0.5, 2.0)
            left_img_aug = left_img_aug * random_brightness
            right_img_aug = right_img_aug * random_brightness
            
            # shift color
            random_color = random.uniform(0.8, 1.2)
            for i in range(3):
                left_img_aug[i,:,:] *= random_color
                right_img_aug[i,:,:] *= random_color
                
            # saturate
            left_img_aug = torch.clamp(left_img_aug, 0, 1)
            right_img_aug = torch.clamp(right_img_aug, 0, 1)
            
            sample = {
                'left_image': left_img_aug,
                'right_image': right_img_aug
            }
        
        return sample