import os, sys
from libs import *

def compute_discrepancy(
    features_T, features_S, 
):
    normalized_features_T, normalized_features_S,  = features_T/torch.norm(features_T, p = 2), features_S/torch.norm(features_S, p = 2), 
    discrepancy = torch.dist(
        normalized_features_T, normalized_features_S, 
        p = 2, 
    )

    return discrepancy

def train_fn(
    train_loaders, num_epochs, 
    models, 
    device = torch.device("cpu"), 
    save_ckps_dir = ".", 
):
    print("\nStart Training ...\n" + " = "*16)
    FT, FS,  = models["FT"].to(device), models["FS"].to(device), 
    GS = models["GS"].to(device)
    optimizer_FS = optim.SGD(
        FS.parameters(), 
        lr = 5e-4, 
    )
    optimizer_GS = optim.SGD(
        GS.parameters(), 
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

    for epoch in range(1, num_epochs + 1):
        print("epoch {}/{}".format(epoch, num_epochs) + "\n" + " - "*16)

        with torch.autograd.set_detect_anomaly(True):
            FT, FS,  = FT.train(), FS.train(), 
            GS = GS.train()
            for images_T, labels_T in tqdm.tqdm(train_loaders["train"]):
                images_T, labels_T = images_T.to(device), labels_T.to(device)
                images_S = GS(images_T)

                features_T, features_S,  = FT.backbone(images_T.float()), FS.backbone(images_S.float()), 
                discrepancy = compute_discrepancy(
                    features_T, features_S, 
                )

                loss_FS = F.cross_entropy(FT.classifier(features_S), labels_T) + discrepancy
                loss_GS = F.cross_entropy(FT.classifier(features_S), labels_T) - torch.minimum(discrepancy - 0.1, torch.zeros(1).cuda())
                loss_FS.backward(retain_graph = True)
                loss_GS.backward(retain_graph = False)

                # Domain Generalized Representation Learning
                optimizer_FS.step()
                state_dict_FT, state_dict_FS,  = FT.state_dict(), FS.state_dict(), 
                for parameter in state_dict_FS:
                    state_dict_FT[parameter] = 0.999*state_dict_FT[parameter] + (1 - 0.999)*state_dict_FS[parameter]
                FT.load_state_dict(state_dict_FT)

                # Learning to Generate Novel Domains
                optimizer_GS.step()

                optimizer_FS.zero_grad()
                optimizer_GS.zero_grad()

        with torch.no_grad():
            FT, FS,  = FT.eval(), FS.eval(), 
            GS = GS.eval()
            running_loss, running_corrects,  = 0.0, 0.0, 
            for images_T, labels_T in tqdm.tqdm(train_loaders["val"]):
                images_T, labels_T = images_T.to(device), labels_T.to(device)

                logits = FT(images_T.float())
                loss = F.cross_entropy(logits, labels_T)

                running_loss, running_corrects,  = running_loss + loss.item()*images_T.size(0), running_corrects + torch.sum(torch.max(logits, 1)[1] == labels_T.data).item(), 
        val_loss, val_accuracy,  = running_loss/len(train_loaders["val"].dataset), running_corrects/len(train_loaders["val"].dataset), 
        print("{:<8} - loss:{:.4f}, accuracy:{:.4f}".format(
            "val", 
            val_loss, val_accuracy, 
        ))

        scheduler_FS.step(), 
        scheduler_GS.step(), 

        torch.save(
            GS, 
            "{}/GS.ptl".format(save_ckps_dir), 
        )
    print("\nFinish Training ...\n" + " = "*16)