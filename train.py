import torch 
import torch.nn as nn
from dice_score import dice_loss
import torch.utils.data as data
import torch.optim as optim
from unet import UNet
from dataset import ISBI_Dataset,trans
import torch.nn.functional as F

def train(net,device,data_path,trans,epochs=40,batch_size=4,lr=0.0001):
    isbi_dataset=ISBI_Dataset(data_path,trans)
    train_loader=data.DataLoader(isbi_dataset,batch_size,shuffle=True)

    optimizer=optim.RMSprop(net.parameters(),lr=lr,weight_decay=1e-8,momentum=0.9)
    criterion=nn.BCEWithLogitsLoss()

    best_loss=float('inf')
    
    for epoch in range(epochs):
        net.train()
        for imgs,labels in train_loader:
            optimizer.zero_grad()

            imgs=imgs.to(device,torch.float32)
            labels=labels.to(device,torch.float32)

            pred=net(imgs)
            loss=criterion(pred,labels)
            loss+=dice_loss(F.sigmoid(pred),labels.float(),multiclass=False)
            print(f"epoch{epoch}:loss:{loss}")
            if loss<best_loss:
                best_loss=loss
                torch.save(net.state_dict(),"model/best_model.pth")
            loss.backward()
            optimizer.step()            
         


if __name__=='__main__':
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net=UNet(3,1)
    net.to(device)
    data_path="data/train"
    train(net,device,data_path,trans)

