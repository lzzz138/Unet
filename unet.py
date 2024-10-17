import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self,input_channels,out_channels):
        super().__init__()
        self.double_conv=nn.Sequential(
            nn.Conv2d(input_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self,input_channels,out_channels):
        super().__init__()
        self.down=nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(input_channels,out_channels)
        )
    def forward(self,x):
        return self.down(x)
    
class Up(nn.Module):
    def __init__(self,input_channels,out_channels):
        super().__init__()
        self.up=nn.ConvTranspose2d(input_channels,input_channels//2,kernel_size=2,stride=2)
        self.conv=DoubleConv(input_channels,out_channels)
    def forward(self,x1,x2):
        x1=self.up(x1)
        diffH=x2.size()[2]-x1.size()[2]
        diffW=x2.size()[3]-x1.size()[3]
        x1=F.pad(x1,(diffW//2,diffW-diffW//2,diffH//2,diffH-diffH//2))
        x1=torch.cat([x1,x2],dim=1)
        return self.conv(x1)

class Out(nn.Module):
    def __init__(self,input_channels,out_channels):
        super().__init__()
        self.out=nn.Conv2d(input_channels,out_channels,kernel_size=1)
    def forward(self,x):
        return self.out(x)



class UNet(nn.Module):
    def __init__(self,input_channels,num_classes):
        super().__init__()
        self.conv=DoubleConv(input_channels,64)
        self.down1=Down(64,128)
        self.down2=Down(128,256)
        self.down3=Down(256,512)
        self.down4=Down(512,1024)
        self.up1=Up(1024,512)
        self.up2=Up(512,256)
        self.up3=Up(256,128)
        self.up4=Up(128,64)
        self.out=Out(64,num_classes)
    def forward(self,x):
        x1=self.conv(x)
        x2=self.down1(x1)
        x3=self.down2(x2)
        x4=self.down3(x3)
        x5=self.down4(x4)
        x6=self.up1(x5,x4)
        x7=self.up2(x6,x3)
        x8=self.up3(x7,x2)
        x9=self.up4(x8,x1)
        return self.out(x9)


if __name__=="__main__":
    net=UNet(3,2)
    a=torch.rand(size=(4,3,512,512))
    b=net(a)
    print(b.shape)
        