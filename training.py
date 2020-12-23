import numpy as np
import torch
from torch import nn
import argparse
import time
import os
import sys
from torchvision import models
from torch.utils.data import DataLoader
import torch.optim as optim
from utils.util import *
from utils.metrics import get_metrics_values
import torch.nn.functional as F
from datasetcClass import classDataset
from models.hrnet import hrnet


def train(epoch,n_epochs,model,data_loader,criterion,optimizer):
    model.train()
    losses = AverageMeter()
    for i, sample in enumerate(data_loader):
        inputs = sample["image"]
        targets = sample["label"].long()
        if torch.cuda.is_available():
            targets = targets.cuda()
            inputs = inputs.cuda()

        model.zero_grad()
        outputs = model(inputs)["output"]
        if outputs.shape[1] != targets.shape[1] or outputs.shape[2] != targets.shape[2]:
            outputs = F.upsample(input=outputs, size=(targets.shape[1],targets.shape[2]), mode="bilinear")
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        losses.update(loss.item())
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d / %d] [Loss: %f]"
            % (epoch,n_epochs, i, len(data_loader), losses.avg)
        )
    print("Epoch Loss", losses.avg)

        
def validate_model(epoch, model, data_loader, criterion,best_val_iou,n_classes,output_dir):

    model.eval()
    losses = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    with torch.no_grad():
        for i, sample in enumerate(data_loader):
            inputs = sample["image"].type(torch.FloatTensor)
            targets = sample["label"].type(torch.LongTensor)
            if torch.cuda.is_available():
                targets = targets.cuda()
                inputs = inputs.cuda()

            outputs = model(inputs, (targets.shape[1], targets.shape[2]))["output"]
            if outputs.shape[1] != targets.shape[1] or outputs.shape[2] != targets.shape[2]:
                 outputs = F.upsample(input=outputs, size=(targets.shape[1],targets.shape[2]), mode="bilinear")
    
            loss = criterion(outputs, targets)
            
            inters, uni = get_metrics_values(
                targets, [outputs],n_classes)
            intersection_meter.update(inters)
            union_meter.update(uni)

            losses.update(loss.item())
            sys.stdout.write(
                "\r[Epoch %d] [Batch %d / %d]" % (epoch, i, len(data_loader))
            )

    iou = intersection_meter.sum / union_meter.sum

    if best_val_iou < np.mean(iou):
        best_val_iou = np.mean(iou)
        torch.save(
            {"state_dict": model.state_dict(), "iou": np.mean(iou), "epoch": epoch},
            os.path.join(output_dir, "best.pth"),
        )
    print("")
    for i, _iou in enumerate(iou):
        print("class [{}], IoU: {:.4f}".format(i, _iou))

    print("Epoch Loss", losses.avg, "Validation_meanIou", np.mean(iou))
    return best_val_iou
    

def main():
    ####################argument parser#################################
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csvpath",
        type=str,
        required=True,
        default="./",
        help="Location to data csv file",
    )


    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="output directory to save model",
    )

    parser.add_argument(
        "--n_classes",
        type=int,
        help="number of claesses",
    )
    
    parser.add_argument(
        "--height",
        type=int,
        default=360,
        required=True,
        help="height of image",
    )
    
    parser.add_argument(
        "--width",
        type=int,
        default=480,
        required=True,
        help="width of image",
    )
 
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        required=True,
        help="batch size",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
        required=True,
        help="number of workers for data loader",
    )
    args = parser.parse_args()
    
    validate = False

    if os.path.exists(os.path.join(args.csvpath, "valid.csv")):
            validate = True
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #cityscapes

    train_object = classDataset(args.csvpath,'train',args.height,args.width,mean_std)
    train_loader = DataLoader(
        train_object,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
    )

    if validate:
        valid_object = classDataset(args.csvpath,'valid',args.height,args.width,mean_std)
        valid_loader = DataLoader(
            valid_object,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.n_workers,
        )
        
    model = hrnet(args.n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    best_val_iou = 0
    if torch.cuda.is_available():
        model.cuda()
        
    start_epoch = 1 
    n_epochs = 500
    for epoch in range(start_epoch, n_epochs + 1):
        train(epoch,n_epochs,model,train_loader,criterion,optimizer)
        if validate:
                best_val_iou = validate_model(
                    epoch, model, valid_loader, criterion, best_val_iou,args.n_classes,args.output_dir
                )

 
if __name__ == "__main__":
    main()
    
