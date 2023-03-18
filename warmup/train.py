import os, sys
from libs import *

from data import ImageDataset
from engines import train_fn

train_loaders = {
    "train":torch.utils.data.DataLoader(
        ImageDataset(
            data_dir = "../datasets/P-ACS/{}/".format("ACS"), 
            augment = True, 
        ), 
        num_workers = 8, batch_size = 16, 
        shuffle = True, 
    ), 
    "val":torch.utils.data.DataLoader(
        ImageDataset(
            data_dir = "../datasets/P-ACS/{}/".format("P"), 
            augment = False, 
        ), 
        num_workers = 8, batch_size = 16, 
        shuffle = False, 
    ), 
}
model = torchvision.models.resnet18(pretrained = True)
model.fc = nn.Linear(
    model.fc.in_features, 7, 
)
optimizer = optim.SGD(
    model.parameters(), weight_decay = 5e-4, 
    lr = 0.002, momentum = 0.9, 
)
lr_scheduler = optim.lr_scheduler.StepLR(
    optimizer, 
    step_size = 20, gamma = 0.1, 
)

save_ckp_dir = "../ckps/P-ACS/{}".format("P")
if not os.path.exists(save_ckp_dir):
    os.makedirs(save_ckp_dir)
train_fn(
    train_loaders, num_epochs = 25, 
    model = model, 
    optimizer = optimizer, 
    lr_scheduler = lr_scheduler, 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
    save_ckp_dir = save_ckp_dir, 
)