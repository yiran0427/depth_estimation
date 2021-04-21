import os
from PIL import Image
from torch.utils.data import Dataset


class KittiDataset(Dataset):
    def __init__(self, root_dir, mode, transform=None):
        
        '''
        root_dir (str): path to the dataset
        mode (str): 'train', 'test'
        transform: transformations applied to the data

        return (dict): return left and right images if in train mode, 
                        return only left image if in test mode
        '''

        zip_paths = os.listdir(root_dir)
        
        self.left_paths = []
        self.right_paths = []
        
        for zip_path in zip_paths:
            if zip_path[0] == '.': continue
            left_dir = os.path.join(root_dir, os.path.join(zip_path, 'image_02/data/'))
            left_fnames = os.listdir(left_dir)
            left_fnames.sort()
            
            for left_fname in left_fnames:
                self.left_paths.append(os.path.join(left_dir, left_fname))
        
        if mode == 'train':
            for zip_path in zip_paths:
                if zip_path[0] == '.': continue
                right_dir = os.path.join(root_dir, os.path.join(zip_path, 'image_03/data/'))
                right_fnames = os.listdir(right_dir)
                right_fnames.sort()
                
                for right_fname in right_fnames:
                    self.right_paths.append(os.path.join(right_dir, right_fname))
                
        self.transform = transform
        self.mode = mode


    def __len__(self):
        return len(self.left_paths)


    def __getitem__(self, idx):
        sample = {}
        
        left_image = Image.open(self.left_paths[idx])
        sample['left_image'] = left_image
        
        if self.mode == 'train':
            right_image = Image.open(self.right_paths[idx])
            sample['right_image'] = right_image

        if self.transform:
            sample = self.transform(sample)
                
        return sample
