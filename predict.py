import glob
import torch
import cv2
import os
from torchvision import transforms as T
from unet import UNet
import numpy as np

if __name__=='__main__':
    net=UNet(input_channels=3,num_classes=1)
    device=torch.device('cuda' if torch.cuda.is_available else 'cpu')
    net.to(device)
    net.load_state_dict(torch.load("model/best_model.pth",map_location=device))
    net.eval()
    #os.listdir获得指定目录下的全部文件名 排好序 glob.glob获得可指定格式的全部文件路径 乱序
    tests_path=os.listdir("data/test/image")
    saves_path="data/test/label"
    for index,test_path in enumerate(tests_path):
        save_path=os.path.join(saves_path,f"{index}.png")
        img=cv2.imread(os.path.join("data/test/image",test_path))

        #torch.from_numpy不会改变HWC T.totensor可以直接改变CHW
        as_tensor=T.ToTensor()
        img=as_tensor(img)
        img=img.unsqueeze(0)
        img=img.to(device,torch.float32)
        
        pred=net(img)
        pred=np.array(pred.data.cpu()[0])[0]

        pred[pred>=0.5]=255
        pred[pred<0.5]=0

        cv2.imwrite(save_path,pred)