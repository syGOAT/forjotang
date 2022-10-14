import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


train_dir = 'data/train'
val_dir = 'data/test'

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Grayscale(), 
])

train_data = datasets.ImageFolder(train_dir, transform=transform)
val_data =   datasets.ImageFolder(val_dir, transform=transform)

batch_size = 64

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader =   DataLoader(val_data, batch_size=batch_size, shuffle=True)

#X, y = next(iter(train_loader))
#print(X, y)

net = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, stride=1), nn.ReLU(), 
    nn.Conv2d(32, 64, kernel_size=3, stride=1), nn.ReLU(), 
    nn.MaxPool2d(kernel_size=2), nn.Dropout(p=0.25), 

    nn.Conv2d(64, 128, kernel_size=3), nn.ReLU(), 
    nn.MaxPool2d(kernel_size=2), 
    nn.Conv2d(128, 128, kernel_size=3), nn.ReLU(), 
    nn.MaxPool2d(kernel_size=2), nn.Dropout(p=0.25), 

    nn.Flatten(), 
    nn.Linear(2048,1024), nn.ReLU(), nn.Dropout(p=0.5), 
    nn.Linear(1024, 7), nn.Softmax(dim=1)
)
net.load_state_dict(torch.load('net.params'))
'''
X = torch.randn(1, 1, 48, 48)
for layer in net:
    X=layer(X)
    print(layer.__class__.__name__,'output shape:\t',X.shape)
'''
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-6)

num_epochs = 100

def train(net, train_loader, val_loader, num_epochs, device, loss, optimizer):
    net.to(device)
    print('train on', device)
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d :
            nn.init.xavier_uniform_(m.weight)
    def acc(loader, device):
        net.eval()
        num_right = 0
        num = 0
        if not device:
            device = next(iter(net.parameters()) ).device
        with torch.no_grad():
            for X, y in loader:
                if isinstance(X, list):
                    X = [x.to(device) for x in X]
                else:
                    X = X.to(device)
                y = y.to(device)
                num += y.numel()
                y_hat = net(X)
                if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
                    y_hat = y_hat.argmax(axis=1) 
                cmp =  y_hat.type(y.dtype) == y
                num_right += int(cmp.type(y.dtype).sum())
        return num_right / num
    net.apply(init_weights)
    for epoch in range(num_epochs):
        net.train()
        for X, y in train_loader:
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            vl = loss(net(X), y)
        print('epoch', epoch, 'train loss =', float(l.mean()), 'train acc =', acc(train_loader, device))
        print('val loss =', float(vl), 'val acc =', acc(val_loader, device))
    torch.save(net.state_dict(),'net.params')    

train(net, train_loader, val_loader, num_epochs, torch.device('cuda:0'), loss, optimizer)


emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

def video_pre():
    # start the webcam feed
    cap = cv2.VideoCapture(0)  
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        bounding_box = cv2.CascadeClassifier('D:/anaconda3/envs/py39/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
                          
        num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)
            
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), 0), 0)
            cropped_img = torch.from_numpy(cropped_img).float()
            emotion_prediction = net(cropped_img)
            
            maxindex = int(torch.argmax(emotion_prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
           
        cv2.imshow('Video', cv2.resize(frame,(1200,860),interpolation = cv2.INTER_CUBIC))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release() 
    cv2.destroyAllWindows()  

video_pre()


