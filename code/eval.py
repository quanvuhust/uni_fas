import torch
import numpy as np
import torch.nn.functional as F
from sklearn import metrics

def eval(val_loader, model, criterion, device, n_eval, fold, epoch, is_ema, default_configs):
    if is_ema:
        print("EMA EVAL")
    else:
        print("NORMAL EVAL")
    running_loss = 0
    k = 0
    model.eval()
    predictions = []
    ground_truths = []
    scores = []
    n_images = 0
    img_path_list = []
    val_metric = {"loss": 0, "acc": 0, "roc_auc": 0, "eer": 0, "acer": 0}
    
    with torch.no_grad():
        for batch_idx, (imgs, labels, img_paths) in enumerate(val_loader):   
            imgs = imgs.to(device).float()
            labels = labels.to(device).long()

            logits = model(imgs)
            probs = torch.nn.Softmax(dim=1)(logits)
                
            loss = 0
            running_loss += loss * imgs.size(0)

            k = k + imgs.size(0)

            _, preds = torch.max(probs, 1)
            predictions += [preds]

            scores += [probs[:,1]]
            ground_truths += [labels.detach().cpu()]
            n_images += len(labels)
            for j in range(len(img_paths)):
                img_path_list.append(img_paths[j])
                

        predictions = torch.cat(predictions).cpu().numpy()
        scores = torch.cat(scores).cpu().numpy()
        ground_truths = torch.cat(ground_truths).cpu().numpy()
        cm = metrics.confusion_matrix(ground_truths, predictions)
        acc = metrics.accuracy_score(ground_truths, predictions)
        roc_auc = metrics.roc_auc_score(ground_truths, scores)

        fpr, tpr, threshold = metrics.roc_curve(ground_truths, scores, pos_label=1)
        fnr = 1 - tpr
        acer_threshold = threshold[np.nanargmin(np.absolute((fnr + fpr)/2))]
        eer_threshold = threshold[np.nanargmin(np.absolute(fnr - fpr))]
        EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        index = np.nanargmin(np.absolute((fnr + fpr)/2))
        ACER = (fpr[index] + fnr[index])/2
        val_metric['eer'] = EER
        val_metric['acer'] = ACER
        
        print("EER: ", EER)
        print("ACER: ", ACER)
        print('ACER thresh: {}\n'.format(acer_threshold))
        print('EER thresh: ', eer_threshold)
        print('Acc: ', acc)
        print("confusion matrix: ", cm)
        print("ROC AUC: ", roc_auc)

    val_metric['loss'] = running_loss/k
    val_metric['acc'] = acc 
    val_metric['roc_auc'] = roc_auc   
         
    return val_metric, ground_truths, scores, acer_threshold

