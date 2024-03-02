import pandas as pd
import json
from dataset import ImageFolder
from torch.utils.data import DataLoader
from model import Net
import os
import torch
from tqdm.auto import tqdm
import numpy as np
from sklearn import metrics

def load_old_weight(model, weight_path):
    if weight_path is not None:
        pretrained_dict = torch.load(weight_path, map_location=torch.device('cpu'))
        model.load_state_dict(pretrained_dict, strict=True)
    return model

def build_net(default_configs, weights_path, device_id):
    model = Net(default_configs["backbone"], default_configs["n_frames"], device_id).to(device_id)
    # print(model)
    model = load_old_weight(model, weights_path)
    return model

if __name__ == '__main__':
    parts = ["p1", "p2.1", "p2.2"]
    exp = "exp_5"
    epoch = 4
    device_id = "cuda:1"
    f = open(os.path.join('code/configs', "{}.json".format(exp)))
    default_configs = json.load(f)
    default_configs["image_size"] = 252
    f.close()
    weight_paths = {
        "p1": "weights/{}/p1/checkpoint_4.pt".format(exp),
                    "p2.1": "weights/{}/p2.1/checkpoint_1.pt".format(exp),
                    "p2.2": "weights/{}/p2.2/checkpoint_4.pt".format(exp)
    }
    print(weight_paths)
    id_list = []
    prob_list = []
    label_list = []
    for part in parts:
        val_df = pd.read_csv("code/data/dev_{}.csv".format(part))
        test_data = ImageFolder(val_df, default_configs, None, "test")
        test_loader = DataLoader(test_data, batch_size=64, 
                pin_memory=True, num_workers=4, drop_last=False, shuffle=False)
    
        model = build_net(default_configs, weight_paths[part], device_id)
        model.eval()

        predictions = []
        
        n_images = 0
        with torch.no_grad():
            for batch_idx, (imgs, labels, ids) in enumerate(tqdm(test_loader)):   
                imgs = imgs.to(device_id).float()
               
                logits = model(imgs)
                probs = torch.nn.Softmax(dim=1)(logits)
                
                predictions += [probs]
                for j in range(len(ids)):
                    id_list.append(ids[j])
                    label_list.append(labels[j])

                    
            predictions = torch.cat(predictions).cpu().numpy()
            for i in range(predictions.shape[0]):
                prob = predictions[i][1]
                prob_list.append(prob)

    scores = np.array(prob_list)
    # print("Eval score mean: ", np.mean(scores))
    ground_truths = np.array(label_list)
    
    roc_auc = metrics.roc_auc_score(ground_truths, scores)

    fpr, tpr, threshold = metrics.roc_curve(ground_truths, scores, pos_label=1)
    fnr = 1 - tpr
    acer_threshold = threshold[np.nanargmin(np.absolute((fnr + fpr)/2))]
    eer_threshold = threshold[np.nanargmin(np.absolute(fnr - fpr))]
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    index = np.nanargmin(np.absolute((fnr + fpr)/2))
    ACER = (fpr[index] + fnr[index])/2


    print("EER: ", EER)
    print("ACER: ", ACER)
    print('ACER thresh: {}\n'.format(acer_threshold))
    print('EER thresh: ', eer_threshold)
    print("ROC AUC: ", roc_auc)

