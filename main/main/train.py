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
            data_dir = "../../datasets/ACS-P/{}/*/".format("ACS"), 
            augment = True, 
        ), 
        num_workers = 8, batch_size = 16, 
        shuffle = True, 
    ), 
    "val":torch.utils.data.DataLoader(
        ImageDataset(
            data_dir = "../../datasets/ACS-P/{}/*/".format("P"), 
            augment = False, 
        ), 
        num_workers = 8, batch_size = 16, 
        shuffle = False, 
    ), 
}
FT = torch.load(
    "../../warmup/ckps/ACS-P/P/last.ptl", 
    map_location = "cpu", 
)
for parameter in FT.parameters():
    parameter.requires_grad = False
models = {
    "FT":FT, "FS":fcn_resnet18(), 
    "GS":fcn_3x64_gctx(), 
}

save_ckps_dir = "../ckps/ACS-P/{}".format("P")
if not os.path.exists(save_ckps_dir):
    os.makedirs(save_ckps_dir)
train_fn(
    train_loaders, num_epochs = 50, 
    models = models, 
    device = torch.device("cuda"), 
    save_ckps_dir = save_ckps_dir, 
)