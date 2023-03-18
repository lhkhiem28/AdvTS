import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__), sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))
from libs import *

from data import ImageDataset
from models import *
from losses import discrepancy

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
optimizer_FS = optim.SGD(
    FS.parameters(), weight_decay = 5e-4, 
    lr = 5e-4, 
)
optimizer_GS = optim.SGD(
    GS.parameters(), weight_decay = 5e-4, 
    lr = 5e-4, 
)
scheduler_FS = optim.lr_scheduler.StepLR(
    optimizer_FS, 
    step_size = 30, gamma = 0.1, 
)
scheduler_GS = optim.lr_scheduler.StepLR(
    optimizer_GS, 
    step_size = 30, gamma = 0.1, 
)

save_ckp_dir = "../ckps/P-ACS/{}".format("P")
if not os.path.exists(save_ckp_dir):
    os.makedirs(save_ckp_dir)
print("\nStart Training ...\n" + " = "*16)
for epoch in range(1, 60 + 1):
    print("epoch {}/{}".format(epoch, 60) + "\n" + " - "*16)

    with torch.autograd.set_detect_anomaly(True):
        for images, labels in tqdm.tqdm(train_loaders["train"]):
            images, labels = images.cuda(), labels.cuda()

            augmented_images = GS(images)
            augmented_features, features,  = FS(augmented_images.float()), FT(images.float()), 
            features_discrepancy = discrepancy(
                augmented_features, features, 
            )

            # Domain Generalized Representation Learning
            loss_FS = features_discrepancy + F.cross_entropy(C(augmented_features), labels)

            features2 = features.clone()
            augmented_features2 = augmented_features.clone()
            labels2 = labels.clone()

            features_discrepancy2 = discrepancy(
                augmented_features2, features2, 
            )
            loss_GS = -torch.minimum(features_discrepancy2 - 0.01, torch.zeros(1).cuda()) + F.cross_entropy(C(augmented_features2), labels2.clone())

            loss_FS.backward(retain_graph = True)
            loss_GS.backward()
            state_dict_FT = FT.state_dict()
            state_dict_FS = FS.state_dict()
            for layer in state_dict_FT:
                state_dict_FT[layer] = 0.999*state_dict_FT[layer] + (1 - 0.999)*state_dict_FS[layer]

            # Learning to Generating Novel Domains
            # loss_GS = -torch.minimum(features_discrepancy - 0.01, torch.zeros(1).cuda()) + F.cross_entropy(C(augmented_features), labels)
            
            optimizer_FS.step()
            optimizer_GS.step()
            optimizer_FS.zero_grad()
            optimizer_GS.zero_grad()

    scheduler_FS.step(), 
    scheduler_GS.step(), 

    with torch.no_grad():
        FT.eval(), C.eval()
        running_loss, running_corrects,  = 0.0, 0.0, 
        for images, labels in tqdm.tqdm(train_loaders["val"]):
            images, labels = images.cuda(), labels.cuda()

            logits = C(FT(images.float()))
            loss = F.cross_entropy(logits, labels)

            running_loss, running_corrects,  = running_loss + loss.item()*images.size(0), running_corrects + torch.sum(torch.max(logits, 1)[1] == labels.data).item(), 
    val_loss, val_accuracy,  = running_loss/len(train_loaders["val"].dataset), running_corrects/len(train_loaders["val"].dataset), 
    print("{:<8} - loss:{:.4f}, accuracy:{:.4f}".format(
        "val", 
        val_loss, val_accuracy, 
    ))

    print("\nFinish Training ...\n" + " = "*16)
    torch.save(
        GS, 
        "{}/GS.ptl".format(save_ckp_dir), 
    )