import cv2
import numpy as np
import torch.utils.data as data
import albumentations as A
from data_augmentations.rand_augment import preprocess_input

def cutout(img, replace=0):
    n_holes = 1
    new_img = img.copy()
    h, w, _ = new_img.shape

    for n in range(n_holes):
        y = np.random.randint(int(0.95*h), h)
        x = np.random.randint(int(0.95*w), w)

        pad_size_x = int(0.04*w/2*np.random.uniform(low=0.5, high=1.0, size=(1,)[0]))
        pad_size_y = int(0.04*h/2*np.random.uniform(low=0.5, high=1.0, size=(1,)[0]))
        y1 = np.clip(y - pad_size_y, 0, h)
        y2 = np.clip(y + pad_size_y, 0, h)
        x1 = np.clip(x - pad_size_x, 0, w)
        x2 = np.clip(x + pad_size_x, 0, w)
        
        new_img[y1:y2, :,:] = replace
        new_img[:, x1:x2,:] = replace

    return new_img

class ImageFolder(data.Dataset):
    def __init__(self, df, default_configs, randaug_magnitude, mode):
        super().__init__()
        df['filename'] = df["filename"]
        self.df = df.reset_index(drop=True)
        self.labels = df["label"].values
        self.mode = mode
        self.df_imgsize = default_configs["image_size"]
        train_real_aug_list = [
            A.RandomResizedCrop(width=self.df_imgsize, height=self.df_imgsize, interpolation=cv2.INTER_CUBIC, p=1),
            A.HorizontalFlip(p=0.5),
        ]

        train_fake_aug_list = [
            A.RandomResizedCrop(width=self.df_imgsize, height=self.df_imgsize, interpolation=cv2.INTER_CUBIC, p=1),
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=.1),
                A.IAAPiecewiseAffine(p=0.3),
            ], p=0.3),
            A.OneOf([
                A.Blur(p=0.5),
                A.MedianBlur(p=0.5)
            ], p=0.3),
        ]
        
        self.train_real_transform = A.Compose(train_real_aug_list)
        self.train_fake_transform = A.Compose(train_fake_aug_list)
        
        self.test_transform = A.Compose([
            A.Resize(height=self.df_imgsize, width=self.df_imgsize, interpolation=cv2.INTER_CUBIC, p=1)
        ]
        )
      
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if self.mode == 'train':
            row = self.df.loc[index]
            file_path = row.filename
            label = self.labels[index]
        elif self.mode == 'test':
            row = self.df.loc[index]
            label = self.labels[index]
            file_path = row.filename

        if self.mode == 'train':
            frame = cv2.imread(file_path)
            
            if label == 0:
                frame = self.train_real_transform(image=frame)["image"]
            else:
                frame = self.train_fake_transform(image=frame)["image"]
                
            frame = preprocess_input(frame, randaug_magnitude=7)
            if label == 1:
                if np.random.rand() < 0.5:
                    frame = cutout(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
        elif self.mode == 'test':
            file_path = file_path.strip('\n')
            frame = cv2.imread(file_path)
            frame = self.test_transform(image=frame)["image"]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame, label, file_path

