import torch.utils.data as data

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms
import numpy as np
import os
import os.path
import sys

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(data_dict,dir=None):
        if dir!=None:
            classes = [c for c in sorted(data_dict) if c[0]!='_']
            classes.sort()
            classes.append('background')            #add background class
            class_to_idx = {classes[i]: i for i in range(len(classes)-1)}
            class_to_idx['background'] = -1         #the label of background class is set to -1
        else:
            classes = [c for c in sorted(data_dict) if c[0]!='_']
            classes.sort()
            class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx


def make_dataset(data_dict, dir, class_to_idx):
    images = []
    idx_to_class = {}
    intervals = []
    i0,i1 = 0,0
    
    for catg in sorted(class_to_idx):#classes
        if catg == 'background':
            continue
        for fdir in data_dict[catg]:
            if is_image_file(fdir):
                idx_to_class[i1] = class_to_idx[catg]
                images.append((fdir, class_to_idx[catg]))
                i1 += 1
        intervals.append((i0,i1))
        i0 = i1
    if i0!=i1:
        intervals.append((i0,i1))
        i0=i1
    if dir!=None:    
        for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_image_file(path):
                    idx_to_class[i1] = -1
                    images.append((path, -1))
                    i1 += 1
        if i0!=i1:
            intervals.append((i0,i1))
            print('Background dataset is not empty!\n')
            i0 = i1
        else:
            print('Background dataset is empty!\n')

    return images, intervals, idx_to_class

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def label_transform():
    pass

def folderReader(path):
    pass

class ImageReader(data.Dataset):

    def __init__(self, data_dict, dir=None, transform=None, target_transform=None,
                 loader=default_loader):
        
        classes, class_to_idx = find_classes(data_dict, dir)
        imgs, intervals, idx_to_class = make_dataset(data_dict, dir, class_to_idx)
        
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images!"))

        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx# class name -> 0,1,2,3,4
        self.intervals = intervals
        self.idx_to_class = idx_to_class# img_idx->class
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        origin = img
        origin = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(256),
                                      transforms.ToTensor(),
                                      ])(origin)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return origin, img, target

    def __len__(self):
        return len(self.imgs)
