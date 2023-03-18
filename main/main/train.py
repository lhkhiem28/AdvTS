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
            data_dir = "../../datasets/P-ACS/{}/".format("ACS"), 
            augment = True, 
        ), 
        num_workers = 8, batch_size = 32, 
        shuffle = True, 
    ), 
    "val":torch.utils.data.DataLoader(
        ImageDataset(
            data_dir = "../../datasets/P-ACS/{}/".format("P"), 
            augment = False, 
        ), 
        num_workers = 8, batch_size = 32, 
        shuffle = False, 
    ), 
}
FT = torch.load("../../warmup/ckps/P-ACS/P/last.ptl")
for parameter in FT.classifier.parameters():
    parameter.requires_grad = False
models = {
    "FT":FT, "FS":fcn_resnet18(), 
    "GS":fcn_3x64_gctx(), 
}

save_ckps_dir = "../ckps/P-ACS/{}".format("P")
if not os.path.exists(save_ckps_dir):
    os.makedirs(save_ckps_dir)
train_fn(
    train_loaders, num_epochs = 60, 
    models = models, 
    device = torch.device("cuda"), 
    save_ckps_dir = save_ckps_dir, 
)