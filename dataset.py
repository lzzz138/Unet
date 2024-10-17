import torch
import os
import glob
import torch.utils.data as data
import albumentations as A
import cv2
from torchvision import transforms as T

trans=A.Compose([
    A.VerticalFlip(p=0.5), #垂直旋转
    A.HorizontalFlip(p=0.5),#水平旋转
    A.RandomBrightnessContrast(p=0.2),   # 随机明亮对比度
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, interpolation=1, 
                       border_mode=4, value=None, mask_value=None, always_apply=False, 
                       approximate=False, p=0.5) #随机弹性变换
])

class ISBI_Dataset(data.Dataset):
    def __init__(self,data_path,transform):
        super().__init__()
        self.data_path=data_path
        self.imgs_path=glob.glob(os.path.join(data_path,"image/*.png"))
        self.labels_path=glob.glob(os.path.join(data_path,"label/*.png"))
        
        self.transform=transform
        self.as_tensor=T.ToTensor()

    def __getitem__(self, index):
        img_path=self.imgs_path[index]
        label_path=self.labels_path[index]
        img=cv2.imread(img_path)
        mask=cv2.imread(label_path)
        
        #数据增广
        augments=self.transform(image=img,mask=mask)
        img=augments['image']
        mask=augments['mask']

        #将mask转换为单通道
        mask=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)

        return self.as_tensor(img),self.as_tensor(mask)

    def __len__(self):
        return len(self.imgs_path)


if __name__=="__main__":
    isbi_dataset=ISBI_Dataset("data/train",trans)

    print(len(isbi_dataset))
    isbi_dataloader=data.DataLoader(isbi_dataset,batch_size=2,shuffle=True)
    for imgs,masks in isbi_dataloader:
        print(imgs.shape)
        print(masks.shape)
                                    
        




