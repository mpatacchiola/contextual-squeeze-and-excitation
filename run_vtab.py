import torch
import torchvision
import numpy as np
import argparse
import os
import time
import random
import collections

from readers.tf_dataset_reader import TfDatasetReader
from readers.image_folder_reader import ImageFolderReader
from metrics import calibration
import backbones


def topk(output, target, ks=(1,)):
  _, pred = output.topk(max(ks), 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))
  return [correct[:k].max(0)[0] for k in ks]
  
def shuffle(images, labels):
    permutation = np.random.permutation(images.shape[0])
    return images[permutation], labels[permutation]


def _get_number_of_batches(batch_size, task_size):
    num_batches = int(np.ceil(float(task_size) / float(batch_size)))
    if num_batches > 1 and (task_size % batch_size == 1):
        num_batches -= 1
    return num_batches
            
def prepare_task(task_dict):
    context_images_np, context_labels_np = task_dict['context_images'], task_dict['context_labels']
    target_images_np, target_labels_np = task_dict['target_images'], task_dict['target_labels']
    # Prepare context
    context_images_np = context_images_np.transpose([0, 3, 1, 2])
    context_images_np, context_labels_np = shuffle(context_images_np, context_labels_np)
    context_images = torch.from_numpy(context_images_np)
    context_labels = torch.from_numpy(context_labels_np)
    # Prepare target
    target_images_np = target_images_np.transpose([0, 3, 1, 2])
    target_images_np, target_labels_np = shuffle(target_images_np, target_labels_np)
    target_images = torch.from_numpy(target_images_np)
    target_labels = torch.from_numpy(target_labels_np).type(torch.LongTensor)
    # Done!
    return context_images, target_images, context_labels, target_labels

def log_write(file, line, mode="a", newline=True, verbose=True):
    with open(file, mode) as f:
         if(newline): f.write(line+"\n")
         else: f.write(line)
         if(verbose): print(line)

def save(backbone, file_path="./checkpoint.dat"):
        backbone_state_dict = backbone.state_dict()
        torch.save({"backbone": backbone_state_dict}, file_path)

def main(args):
    if(args.device==""):
        args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("[INFO] Using device:", str(args.device))

    if(args.adapter == "case"):
        from adapters.case import CaSE
        adapter = CaSE

    if(args.backbone=="ResNet18"):
        from backbones import resnet
        backbone = resnet.resnet18(pretrained=True, progress=True, norm_layer=torch.nn.BatchNorm2d, adaptive_layer=adapter)
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif(args.backbone=="ResNet50"):
        from backbones import resnet
        backbone = resnet.resnet50(pretrained=True, progress=True, norm_layer=torch.nn.BatchNorm2d)
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif(args.backbone=="EfficientNetB0"):
        from backbones import efficientnet
        backbone = efficientnet.efficientnet_b0(pretrained=True, progress=True, norm_layer=torch.nn.BatchNorm2d, adaptive_layer=adapter)
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif(args.backbone=="BiT-S-R50x1"):
        from backbones import bit_resnet
        backbone = bit_resnet.KNOWN_MODELS[args.backbone](adaptive_layer=adapter)
        if(args.resume_from!=""):
            checkpoint = torch.load(args.resume_from)
            backbone.load_state_dict(checkpoint['backbone'])
            print("[INFO] Loaded checkpoint from:", args.resume_from)
        else:
            backbone.load_from(np.load(f"{args.backbone}.npz"))
        normalize = torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    else:
        print(f"[ERROR] backbone {args.backbone} not supported!")
        quit()

    if(args.resume_from!=""):
        checkpoint = torch.load(args.resume_from)
        backbone.load_state_dict(checkpoint['backbone'])
        print("[INFO] Loaded checkpoint from:", args.resume_from)
    backbone = backbone.to(args.device)

    test_transform = torchvision.transforms.Compose([normalize])  

    if(args.model=="uppercase"):
        from models.uppercase import UpperCaSE
        model = UpperCaSE(backbone, adapter, args.device, tot_iterations=500, start_lr=1e-3, stop_lr=1e-5)
    else:
        print("[ERROR] The model", args.model, "is not implemented!")

    print("[INFO] Start evaluating...\n")
    line = "method,backbone,dataset,dataset-loss,dataset-gce,dataset-ece,dataset-ace,dataset-tace,dataset-sce,dataset-rmsce,dataset-top1"
    log_write(args.log_path, line, mode="w", verbose=True)

    context_set_size = 1000
    datasets_list = [
            {'name': "caltech101", 'task': None, 'enabled': True},
            {'name': "cifar100", 'task': None, 'enabled': True},
            {'name': "oxford_flowers102", 'task': None, 'enabled': True},
            {'name': "oxford_iiit_pet", 'task': None, 'enabled': True},
            {'name': "sun397", 'task': None, 'enabled': True},
            {'name': "svhn_cropped", 'task': None, 'enabled': True},
            {'name': "eurosat", 'task': None, 'enabled': True},
            {'name': "resisc45", 'task': None, 'enabled': True},
            {'name': "patch_camelyon", 'task': None, 'enabled': True},
            {'name': "clevr", 'task': "count", 'enabled': True},
            {'name': "clevr", 'task': "distance", 'enabled': True},
            {'name': "smallnorb", 'task': "azimuth", 'enabled': True},
            {'name': "smallnorb", 'task': "elevation", 'enabled': True},
            {'name': "dmlab", 'task': None, 'enabled': True},
            {'name': "kitti", 'task': None, 'enabled': True},
            {'name': "diabetic_retinopathy_detection", 'task': None, 'enabled': True},
            {'name': "dsprites", 'task': "location", 'enabled': True},
            {'name': "dsprites", 'task': "orientation", 'enabled': True},
        ]


    all_ce, all_top1 = [], []
    all_gce, all_ece, all_ace, all_tace, all_sce, all_rmsce = [], [], [], [], [], []
    for dataset in datasets_list:
        dataset_name = dataset['name']
        
        if dataset['enabled'] is False:
                    continue

        if dataset_name == "sun397":  # use the image folder reader as the tf reader is broken for sun397
                    dataset_reader = ImageFolderReader(
                        path_to_images=args.download_path_for_sun397_dataset,
                        context_batch_size=context_set_size,
                        target_batch_size=args.batch_size,
                        image_size=args.image_size,
                        device="cpu")
        else:  # use the tensorflow dataset reader
                    dataset_reader = TfDatasetReader(
                        dataset=dataset['name'],
                        task=dataset['task'],
                        context_batch_size=context_set_size,
                        target_batch_size=args.batch_size,
                        path_to_datasets=args.download_path_for_tensorflow_datasets,
                        image_size=args.image_size,
                        device="cpu")
                    
        # Get the context images/labels
        context_images, context_labels = dataset_reader.get_context_batch()
        context_images = context_images.to("cpu")
        context_labels = context_labels.long().to(args.device)
        # Normalize those images
        context_images = (context_images + 1.0) / 2.0
        context_images = test_transform(context_images)
        tot_classes = torch.amax(context_labels).item()+1

        if(dataset_name=="sun397" and tot_classes<=1):
            raise Exception("[ERROR] Wrong folder for sun397, tot_classes<=1.")

        print(f"\ndataset: {dataset_name}, tot-context-imgs: {context_images.shape[0]}, tot-classes: {tot_classes}")
        model.predict_batch(context_images, context_labels, target_images=None, reset=True)

        # Target images/labels
        test_set_size = dataset_reader.get_target_dataset_length()
        num_batches = _get_number_of_batches(args.batch_size, test_set_size)
        
        print("test_set_size:", test_set_size, "num_batches:", num_batches)
        
        target_log_probs_list = []
        target_labels_list = []
        for batch_idx in range(num_batches):
            target_images, target_labels = dataset_reader.get_target_batch()

            # Normalize the images
            target_images = target_images.to(args.device)
            target_labels = target_labels.long().to(args.device)
            target_images = (target_images + 1.0) / 2.0
            target_images = test_transform(target_images)

            # Prediction
            log_probs = model.predict_batch(context_images=None, context_labels=None, target_images=target_images, reset=False)
            target_log_probs_list.append(log_probs)
            target_labels_list.append(target_labels)
                    
            if(batch_idx%(num_batches//5)==0): 
                print(f"[{batch_idx}|{num_batches}] dataset: {dataset_name}; context-shape: {target_images.shape}")
                    
        target_log_probs = torch.cat(target_log_probs_list, dim=0)
        target_labels = torch.cat(target_labels_list, dim=0)
        nll = torch.nn.NLLLoss(reduction='none')(target_log_probs, target_labels)
        top1, = topk(target_log_probs, target_labels, ks=(1,))
        dataset_top1 = (top1.float().detach().cpu().numpy() * 100.0).mean()
        dataset_nll = nll.mean().detach().cpu().numpy().mean()
        all_top1.append(dataset_top1)
        # Compute the 95% confidence intervals over the tasks accuracies
        # From: https://github.com/cambridge-mlg/LITE/blob/6e6499b3cfe561a963d9439755be0a37357c7729/src/run.py#L287
        accuracies = np.array(top1.float().detach().cpu().numpy())
        dataset_top1_confidence = (196.0 * np.std(accuracies)) / np.sqrt(len(accuracies))
        # Estimate the error metrics for calibration
        target_labels_np = target_labels.detach().cpu().numpy()
        probs_np = torch.exp(target_log_probs).detach().cpu().numpy()
        dataset_gce = calibration.compute_all_metrics(labels=target_labels_np, probs=probs_np, num_bins=15, return_mean=True)
        dataset_ece = calibration.ece(labels=target_labels_np, probs=probs_np, num_bins=15)
        dataset_ace = calibration.ace(labels=target_labels_np, probs=probs_np, num_bins=15)
        dataset_tace = calibration.tace(labels=target_labels_np, probs=probs_np, num_bins=15, threshold=0.01)
        dataset_sce = calibration.sce(labels=target_labels_np, probs=probs_np, num_bins=15)
        dataset_rmsce = calibration.rmsce(labels=target_labels_np, probs=probs_np, num_bins=15)
        all_gce.append(dataset_gce)
        all_ece.append(dataset_ece)
        all_ace.append(dataset_ace)
        all_tace.append(dataset_tace)
        all_sce.append(dataset_sce)
        all_rmsce.append(dataset_rmsce)

        line = f"{args.model},{args.backbone},{dataset_name}," \
               f"{dataset_nll:.5f}," \
               f"{dataset_gce*100:.2f},{dataset_ece*100:.2f}," \
               f"{dataset_ace*100:.2f},{dataset_tace*100:.2f}," \
               f"{dataset_sce*100:.2f},{dataset_rmsce*100:.2f}," \
               f"{dataset_top1:.2f}"
        log_write(args.log_path, line, mode="a", verbose=True)

    # Finished!
    print(f"*[TOTAL] accuracy: {np.mean(all_top1):.2f}, GCE: {np.mean(all_gce)*100.0:.2f}, ECE: {np.mean(all_ece)*100.0:.2f}, ACE: {np.mean(all_ace)*100.0:.2f}, TACE: {np.mean(all_tace)*100.0:.2f}, SCE: {np.mean(all_sce)*100.0:.2f}, RMSCE: {np.mean(all_rmsce)*100.0:.2f}\n")
                          
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", choices=["uppercase"], default="uppercase", help="The model used for the evaluation.")
  parser.add_argument("--backbone", choices=["BiT-S-R50x1", "ResNet18", "ResNet50", "EfficientNetB0"], default="EfficientNetB0", help="The backbone used for the evaluation.")
  parser.add_argument("--adapter", choices=["case"], default="case", help="The adapted used.")
  parser.add_argument("--data_path", default="../datasets", help="Path to Meta-Dataset records.")
  parser.add_argument("--log_path", default="./log.csv", help="Path to log CSV file for the run.")
  parser.add_argument("--checkpoint_path", default="./checkpoints", help="Path to Meta-Dataset records.")
  parser.add_argument("--download_path_for_tensorflow_datasets", type=str, default="", help="Path to TF datasets.")
  parser.add_argument("--download_path_for_sun397_dataset", type=str, default="", help="Path to TF datasets.")
  parser.add_argument("--max_way_train", type=int, default=50, help="Maximum way of meta-train task.")
  parser.add_argument("--max_support_train", type=int, default=500,
                          help="Maximum support set size of meta-train task.")                                 
  parser.add_argument("--image_size", type=int, default=224, help="Image height and width.")
  parser.add_argument("--num_train_tasks", type=int, default=10000, help="Number of train tasks.")
  parser.add_argument("--num_test_tasks", type=int, default=600, help="Number of test tasks.")
  parser.add_argument("--num_validation_tasks", type=int, default=700, help="Number of validation tasks.")
  parser.add_argument("--resume_from", default="", help="Checkpoint path for the backbone.")
  parser.add_argument("--device", default="", help="Device to use.")
  parser.add_argument("--batch_size", type=int, default=50, help="Size of batches loaded during prediction.")

  args = parser.parse_args()
  main(args)
