import os

import mlflow
from losses.class_balanced_loss import CB_loss
from timm.loss import LabelSmoothingCrossEntropy
from data_augmentations.mixup import mixup, cutmix
from optimizer.sam import SAM
from optimizer.adan import Adan
from optimizer.ranger21.ranger21 import Ranger21
from datetime import datetime
import json

import pandas as pd
import numpy as np
import time
import torch
import random

from torch.optim import lr_scheduler

from datetime import datetime
from timm.utils import ModelEma
from model import Net

from tqdm import tqdm

from dataset import ImageFolder
from torch.utils.data import DataLoader

import torch

from eval import eval
import argparse
import gc
from timm.utils import get_state_dict

default_configs = {}

def train_one_fold(fold, samples_per_cls, train_loader, test_loader):
    print("FOLD: ", fold)
    
    DATA_PATH = "train"
    now = datetime.now()
    weight_path = os.path.join("weights", default_configs["a_name"])
    os.makedirs(weight_path, exist_ok=True)
    weight_path = os.path.join(weight_path, str(fold))
    os.makedirs(weight_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    start_epoch = 0
    criterion_class = LabelSmoothingCrossEntropy(default_configs["smoothing_value"])
    
    criterion_class.to(device)
    model = Net(default_configs["backbone"], default_configs["n_frames"], device)

    if default_configs["optimizer"] == "SAM":
        base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
        optimizer_model = SAM(model.parameters(), base_optimizer, lr=default_configs["lr"], momentum=0.9, weight_decay=default_configs["weight_decay"], adaptive=True)
    
    elif default_configs["optimizer"] == "Ranger21":
        optimizer_model = Ranger21(model.parameters(), lr=default_configs["lr"], weight_decay=default_configs["weight_decay"], 
        num_epochs=default_configs["num_epoch"], num_batches_per_epoch=len(train_loader))
    elif default_configs["optimizer"] == "SGD":
        optimizer_model = torch.optim.SGD(model.parameters(), lr=default_configs["lr"], weight_decay=default_configs["weight_decay"], momentum=0.9)
    else:
        optimizer_model = Adan(model.parameters(), lr=default_configs["lr"], weight_decay=default_configs["weight_decay"])

    scheduler = lr_scheduler.OneCycleLR(optimizer_model, default_configs["lr"], steps_per_epoch=len(train_loader), epochs=default_configs["num_epoch"])
    model.to(device)
    
    scaler = torch.cuda.amp.GradScaler()
    best_model_path = ""


    for epoch in range(start_epoch, default_configs["num_epoch"]):
        print("\n-----------------Epoch: " + str(epoch) + " -----------------")
        
        for param_group in optimizer_model.param_groups:
            mlflow.log_metric("lr", param_group['lr'], step=epoch)
            print("LR: ", param_group['lr'])
        if epoch == default_configs['start_ema_epoch']:
            print("Start ema......................................................")
            model_ema = ModelEma(
                model,
                decay=default_configs["model_ema_decay"],
                device=device, resume='')
       
        start = time.time()
        optimizer_model.zero_grad()
        for batch_idx, (imgs, labels, img_paths) in enumerate(tqdm(train_loader)):
            model.train()
            imgs = imgs.to(device).float()
            labels = labels.to(device).long()  

            if torch.rand(1)[0] < 0.5 and (default_configs["use_mixup"] or default_configs["use_cutmix"]):
                rand_prob = torch.rand(1)[0]

                if default_configs["use_mixup"] == True and default_configs["use_cutmix"] == False:
                    mix_images, target_a, target_b, lam = mixup(imgs, labels, alpha=default_configs["mixup_alpha"])
                if default_configs["use_mixup"] == False and default_configs["use_cutmix"] == True:
                    mix_images, target_a, target_b, lam = cutmix(imgs, labels, alpha=default_configs["mixup_alpha"])
                if default_configs["use_mixup"] == True and default_configs["use_cutmix"] == True: 
                    if rand_prob < 0.5:
                        mix_images, target_a, target_b, lam = mixup(imgs, labels, alpha=default_configs["mixup_alpha"])
                    else:
                        mix_images, target_a, target_b, lam = cutmix(imgs, labels, alpha=default_configs["mixup_alpha"])
                
                with torch.cuda.amp.autocast():
                    logits = model(mix_images)
                    if default_configs["use_cbloss"]:
                        loss = lam * CB_loss(target_a, logits, samples_per_cls, 2, default_configs["cbloss_type"], 0.9999, 2.0, device) + \
                        (1 - lam) * CB_loss(target_b, logits, samples_per_cls, 2, default_configs["cbloss_type"], 0.9999, 2.0, device)
                    else:
                        loss = criterion_class(logits, target_a) * lam + \
                        (1 - lam) * criterion_class(logits, target_b)
                    loss /= default_configs["accumulation_steps"]
                
                scaler.scale(loss).backward()
                if ((batch_idx + 1) % default_configs["accumulation_steps"] == 0) or ((batch_idx + 1) == len(train_loader)):
                    scaler.step(optimizer_model)
                    scaler.update()
                    if epoch >= default_configs['start_ema_epoch']:
                        model_ema.update(model)
                    optimizer_model.zero_grad()
            else:
                with torch.cuda.amp.autocast():
                    logits = model(imgs)
                    
                    if default_configs["use_cbloss"]:
                        loss = CB_loss(labels, logits, samples_per_cls, 2,default_configs["cbloss_type"], 0.9999, 2.0, device)
                    else:
                        loss = criterion_class(logits, labels)
                    loss /= default_configs["accumulation_steps"]
                scaler.scale(loss).backward()
                if ((batch_idx + 1) % default_configs["accumulation_steps"] == 0) or ((batch_idx + 1) == len(train_loader)):
                    scaler.step(optimizer_model)
                    scaler.update()
                    if epoch >= default_configs['start_ema_epoch']:
                        model_ema.update(model)
                    optimizer_model.zero_grad()
        
            scheduler.step()

        end = time.time()
        mlflow.log_metric("train_elapsed_time_f{}".format(fold), end - start, step=epoch)
        best_model_path = os.path.join(weight_path, 'checkpoint_{}.pt'.format(epoch))
        torch.save(get_state_dict(model_ema), best_model_path)
    

    mlflow.log_artifact(best_model_path)
    del model
    del model_ema
    torch.cuda.empty_cache()
    gc.collect()


def seed_worker(worker_id):
    worker_seed = 2021
    np.random.seed(worker_seed)
    random.seed(worker_seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train code")
    parser.add_argument("--exp", type=str, default="exp_1")
    args = parser.parse_args()

    f = open(os.path.join('code/configs', "{}.json".format(args.exp)))
    default_configs = json.load(f)
    f.close()
    print(default_configs)

    existing_exp = mlflow.get_experiment_by_name(args.exp)
    if not existing_exp:
        mlflow.create_experiment(args.exp)

    experiment = mlflow.set_experiment(args.exp)
    experiment_id = experiment.experiment_id

    seed = 2021

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    DATA_PATH = "train"
    folds = ["p1", "p2.1", "p2.2"]
    
    train_loader_list = {}
    test_loader_list = {}
    samples_per_cls_list = {}
    g = torch.Generator()
    g.manual_seed(0)
    for fold in folds:
        train_df = pd.read_csv("code/data/train_{}.csv".format(fold))
        samples_per_cls = [0, 0]
        for index, row in train_df.iterrows():
            label = row['label']
            samples_per_cls[label] += 1
        samples_per_cls_list[fold] = samples_per_cls
        
        train_data_1 = ImageFolder(train_df, default_configs, {default_configs["image_size"]: 9}, "train")
        train_loader = DataLoader(train_data_1, batch_size=default_configs["batch_size"], shuffle=True, drop_last=True, pin_memory=True, num_workers=default_configs["num_workers"], worker_init_fn=seed_worker, generator=g)
        train_loader_list[fold] = train_loader
        test_loader_list[fold] = None
    
    with mlflow.start_run(
        experiment_id=experiment_id,
    ) as parent_run:
        mlflow.set_tag("mlflow.runName", "exp")
        mlflow.log_params(default_configs)
        mlflow.log_artifacts("code") 
        for fold in folds:
            with mlflow.start_run(experiment_id=experiment_id,
                description="fold_{}".format(fold),
                tags={
                    mlflow.utils.mlflow_tags.MLFLOW_PARENT_RUN_ID: parent_run.info.run_id
                }, nested=True):
                train_one_fold(fold, samples_per_cls_list[fold], train_loader_list[fold], test_loader_list[fold]) 
                
        mlflow.end_run()
    