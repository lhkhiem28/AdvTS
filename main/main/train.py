import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__), sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))
from libs import *

from data import ImageDataset
from models import *
from losses import discrepancy_loss

train_loaders = {
    "train":torch.utils.data.DataLoader(
        ImageDataset(
            data_dir = "../../datasets/P-ACS/{}/".format("ACS"), 
            augment = True, 
        ), 
        num_workers = 8, batch_size = 16, 
        shuffle = True, 
    ), 
    "val":torch.utils.data.DataLoader(
        ImageDataset(
            data_dir = "../../datasets/P-ACS/{}/".format("P"), 
            augment = False, 
        ), 
        num_workers = 8, batch_size = 16, 
        shuffle = False, 
    ), 
}
FT, C = torch.load("../../warmup/ckps/P-ACS/P/last.ptl").backbone.cuda(), torch.load("../../warmup/ckps/P-ACS/P/last.ptl").classifier.cuda()
for parameter in C.parameters():
    parameter.requires_grad = False
FS = resnet18(
    num_classes = 7, 
).backbone.cuda()
GS = fcn_3x64_gctx().cuda()