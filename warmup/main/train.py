import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__), sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))
from libs import *

from data import ImageDataset
from models import *
from engines import train_fn

train_loaders = {
    "train":torch.utils.data.DataLoader(
        ImageDataset(
            data_dir = "../../datasets/H-D/{}/*/".format("HAM"), 
            augment = True, 
        ), 
        num_workers = 8, batch_size = 16, 
        shuffle = True, 
    ), 
    "val":torch.utils.data.DataLoader(
        ImageDataset(
            data_dir = "../../datasets/H-D/{}/*/".format("DMF"), 
            augment = False, 
        ), 
        num_workers = 8, batch_size = 16, 
        shuffle = False, 
    ), 
}
model = fcn_resnet18(
    num_classes = 7, 
)
optimizer = optim.AdamW(
    model.parameters(), weight_decay = 1e-4, 
    lr = 1e-4, 
)
scheduler = optim.lr_scheduler.StepLR(
    optimizer, 
    step_size = 20, gamma = 0.1, 
)

save_ckp_dir = "../ckps/H-D/{}".format("DMF")
if not os.path.exists(save_ckp_dir):
    os.makedirs(save_ckp_dir)
train_fn(
    train_loaders, num_epochs = 25, 
    model = model, 
    optimizer = optimizer, 
    scheduler = scheduler, 
    device = torch.device("cuda"), 
    save_ckp_dir = save_ckp_dir, 
)