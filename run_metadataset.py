import torch
import torchvision
import numpy as np
import argparse
import os
import time
import random
import collections

from readers.meta_dataset_reader import MetaDatasetReader
from metrics import calibration
import backbones

# Run these commands before training/testing:
# ulimit -n 50000
# export META_DATASET_ROOT=/path_to_metadataset_folder
#
# Example command for testing:
# python run_metadataset.py --model=uppercase --backbone=EfficientNetB0 --data_path=/path_to_metadataset_records --log_path=./logs/uppercase_EfficientNetB0_seed1_`date +%F_%H%M%S`.csv --image_size=224 --num_test_tasks=1200 --mode=test

def topk(output, target, ks=(1,)):
  _, pred = output.topk(max(ks), 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))
  return [correct[:k].max(0)[0] for k in ks]
  
def shuffle(images, labels):
    permutation = np.random.permutation(images.shape[0])
    return images[permutation], labels[permutation]
    
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

def count_parameters(model, adapter, verbose=True):
        params_total = 0
        params_backbone = 0
        params_adapters = 0
        # Count adapter parameters
        for module_name, module in model.named_modules():
            for parameter in module.parameters():
                if(type(module) is adapter): params_adapters += parameter.numel()
        # Count all parameters
        for parameter in model.parameters():
            params_total += parameter.numel()
        # Subtract to get the backbone parameters (with no adapters)
        params_backbone = params_total - params_adapters
        # Done, printing
        info_str = f"params-backbone .... {params_backbone} ({(params_backbone/1e6):.2f} M)\n" \
                   f"params-adapters .... {params_adapters} ({(params_adapters/1e6):.2f} M)\n" \
                   f"params-total ....... {params_backbone+params_adapters} ({((params_backbone+params_adapters)/1e6):.2f} M)\n"
        if(verbose): print(info_str)
        return params_backbone, params_adapters
        
def train(args, model, dataset, dataset_list, image_transform, eval_every=5000):
    best_accuracy = 0.0
    best_iteration = 0
    train_accuracy_deque = collections.deque(maxlen=100)
    for task_idx in range(args.num_train_tasks):
        # Gather and normalize
        task_dict = dataset.get_train_task()
        context_images, target_images, context_labels, target_labels = prepare_task(task_dict)
        context_images = context_images.to(args.device)
        target_images = target_images.to(args.device)
        context_labels = context_labels.long().to(args.device)
        target_labels = target_labels.long().to(args.device)
        context_images = (context_images + 1.0) / 2.0
        target_images = (target_images + 1.0) / 2.0
        context_images = image_transform(context_images)
        target_images = image_transform(target_images)

        task_way = torch.max(context_labels).item() + 1
        task_tot_images = context_images.shape[0]
        task_avg_shot = task_tot_images / task_way

        log_probs = model.learn(task_idx, args.num_train_tasks, context_images, context_labels, target_images, target_labels)
        nll = torch.nn.NLLLoss(reduction='none')(log_probs, target_labels)

        top1, = topk(log_probs, target_labels, ks=(1,))
        task_top1 = (top1.float().detach().cpu().numpy() * 100.0).mean()
        task_nll = nll.mean().detach().cpu().numpy().mean()
        train_accuracy_deque.append(task_top1)

        line = f"[{task_idx+1}|{args.num_train_tasks}] {args.model}; {args.backbone}; " \
               f"Tot-imgs: {task_tot_images}; Avg-Shot: {task_avg_shot:.1f}; Way: {task_way}; " \
               f"Task-NLL: {task_nll:.5f}; " \
               f"Task-Acc: {task_top1:.1f}; " \
               f"Train-Acc: {np.mean(list(train_accuracy_deque)):.1f}"
        print(line)
        
        # Validation
        if(task_idx%eval_every==0 and task_idx>0):
            print("*Validation...")
            validation_accuracy_list = list()
            for val_idx in range(args.num_validation_tasks):
                # Gather and normalize
                dataset_name = random.choice(dataset_list)
                task_dict = dataset.get_validation_task(dataset_name)
                context_images, target_images, context_labels, target_labels = prepare_task(task_dict)
                context_images = context_images.to(args.device)
                target_images = target_images.to(args.device)
                context_labels = context_labels.long().to(args.device)
                target_labels = target_labels.long().to(args.device)
                context_images = (context_images + 1.0) / 2.0
                target_images = (target_images + 1.0) / 2.0
                context_images = image_transform(context_images)
                target_images = image_transform(target_images)            
                # Evaluate
                log_probs = model.predict(context_images, context_labels, target_images)
                top1, = topk(log_probs, target_labels, ks=(1,))
                task_top1 = (top1.float().detach().cpu().numpy() * 100.0).mean()
                validation_accuracy_list.append(task_top1)
                # Printing stuff
                if((val_idx+1)%(args.num_validation_tasks//10)==0 or (val_idx+1)==args.num_validation_tasks):
                    line = f"*Validation [{val_idx+1}|{args.num_validation_tasks}] " \
                           f"accuracy: {np.mean(validation_accuracy_list):.1f} " \
                           f"(best: {best_accuracy:.1f} at {best_iteration}); "
                    print(line)
            if(np.mean(validation_accuracy_list)>best_accuracy):
                 checkpoint_path = args.checkpoint_path + "/best_" + args.model + "_" + args.backbone + ".dat"
                 print("Best model! Saving in:", checkpoint_path)
                 save(model.backbone, file_path=checkpoint_path)
                 best_accuracy = np.mean(validation_accuracy_list)
                 best_iteration = task_idx+1

def main(args):
    if(args.device==""):
        args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("[INFO] Using device:", str(args.device))

    if(args.adapter == "case"):
        from adapters.case import CaSE
        adapter = CaSE

    train_set = ['ilsvrc_2012', 'omniglot', 'aircraft', 'cu_birds', 'dtd', 'quickdraw', 'fungi', 'mnist']
    validation_set = ['omniglot', 'aircraft', 'cu_birds', 'dtd', 'quickdraw', 'fungi', 'mscoco']
    test_set = ["omniglot", "aircraft", "cu_birds", "dtd", "quickdraw", "fungi", "traffic_sign", "mscoco"]
        
    if(args.backbone=="ResNet18"):
        from backbones import resnet
        backbone = resnet.resnet18(pretrained=True, progress=True, norm_layer=torch.nn.BatchNorm2d, adaptive_layer=adapter)
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif(args.backbone=="EfficientNetB0"):
        from backbones import efficientnet
        backbone = efficientnet.efficientnet_b0(pretrained=True, progress=True, norm_layer=torch.nn.BatchNorm2d, adaptive_layer=adapter)
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif(args.backbone=="BiT-S-R50x1"):
        from backbones import bit_resnet
        backbone = bit_resnet.KNOWN_MODELS[args.backbone](use_adapter=True)
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

    # Print number of params
    count_parameters(backbone, adapter=adapter, verbose=True)
    # Call reset method to impose CaSE -> identity-output
    for name, module in backbone.named_modules():
        if(type(module) is adapter):
            module.reset_parameters() 

    if(args.resume_from!=""):
        checkpoint = torch.load(args.resume_from)
        backbone.load_state_dict(checkpoint['backbone'], strict=True)
        print("[INFO] Loaded checkpoint from:", args.resume_from)
    backbone = backbone.to(args.device)

    test_transform = torchvision.transforms.Compose([normalize])  

    if(args.model=="uppercase"):
        from models.uppercase import UpperCaSE
        model = UpperCaSE(backbone, adapter, args.device, tot_iterations=500, start_lr=1e-3, stop_lr=1e-5)
    else:
        print("[ERROR] The model", args.model, "is not implemented!")

    print("[INFO] Defined a", args.model, "model")
    print("[INFO] Preparing MetaDatasetReader...")
    dataset = MetaDatasetReader(
        data_path=args.data_path,
        mode=args.mode,
        train_set=train_set,
        validation_set=validation_set,
        test_set=test_set,
        max_way_train=args.max_way_train,
        max_way_test=50,
        max_support_train=args.max_support_train,
        max_support_test=500,
        max_query_train=10,
        max_query_test=10,
        image_size=args.image_size)


    if(args.mode=="train" or args.mode=="train_test"):
        print("[INFO] Start training...\n")
        train(args, model, dataset, dataset_list=validation_set, image_transform=test_transform)
        # Saving the checkpoint
        checkpoint_path = args.checkpoint_path + "/" + args.model + "_" + args.backbone + ".dat"
        print("Saving model in:", checkpoint_path)
        save(model.backbone, file_path=checkpoint_path)
    if(args.mode == "train"): quit()

    
    print("[INFO] Start evaluating...\n")
    line = "method,backbone,dataset,task-idx,task-tot-images,task-avg-shot,task-way,task-loss,task-gce,task-ece,task-ace,task-tace,task-sce,task-rmsce,task-top1,all-top1-mean,all-top1-95ci,time"
    log_write(args.log_path, line, mode="w", verbose=True)
    
    for dataset_name in test_set:
        all_ce, all_top1 = [], []
        all_gce, all_ece, all_ace, all_tace, all_sce, all_rmsce = [], [], [], [], [], []
        dataset_time = time.time()
        
        for task_idx in range(args.num_test_tasks):
            task_time = time.time()
            task_dict = dataset.get_test_task(dataset_name)
            context_images, target_images, context_labels, target_labels = prepare_task(task_dict)
            
            context_images = context_images.to(args.device)
            target_images = target_images.to(args.device)
            context_labels = context_labels.long().to(args.device)
            target_labels = target_labels.long().to(args.device)
            
            # Brings back to range [0,1] then normalize
            context_images = (context_images + 1.0) / 2.0
            target_images = (target_images + 1.0) / 2.0
            context_images = test_transform(context_images)
            target_images = test_transform(target_images)
            
            task_way = torch.max(context_labels).item() + 1
            task_tot_images = context_images.shape[0]
            task_avg_shot = task_tot_images / task_way

            log_probs = model.predict(context_images, context_labels, target_images)
            nll = torch.nn.NLLLoss(reduction='none')(log_probs, target_labels)

            top1, = topk(log_probs, target_labels, ks=(1,))
            task_top1 = (top1.float().detach().cpu().numpy() * 100.0).mean()
            task_nll = nll.mean().detach().cpu().numpy().mean()
            all_top1.append(task_top1)
            # Compute the 95% confidence intervals over the tasks accuracies
            # From: https://github.com/cambridge-mlg/LITE/blob/6e6499b3cfe561a963d9439755be0a37357c7729/src/run.py#L287
            accuracies = np.array(all_top1) / 100.0
            all_top1_confidence = (196.0 * np.std(accuracies)) / np.sqrt(len(accuracies))
            # Estimate the error metrics for calibration
            target_labels_np = target_labels.detach().cpu().numpy()
            probs_np = torch.exp(log_probs).detach().cpu().numpy()
            task_gce = calibration.compute_all_metrics(labels=target_labels_np, probs=probs_np, num_bins=15, return_mean=True)
            task_ece = calibration.ece(labels=target_labels_np, probs=probs_np, num_bins=15)
            task_ace = calibration.ace(labels=target_labels_np, probs=probs_np, num_bins=15)
            task_tace = calibration.tace(labels=target_labels_np, probs=probs_np, num_bins=15, threshold=0.01)
            task_sce = calibration.sce(labels=target_labels_np, probs=probs_np, num_bins=15)
            task_rmsce = calibration.rmsce(labels=target_labels_np, probs=probs_np, num_bins=15)
            all_gce.append(task_gce)
            all_ece.append(task_ece)
            all_ace.append(task_ace)
            all_tace.append(task_tace)
            all_sce.append(task_sce)
            all_rmsce.append(task_rmsce)
            
            stop_time = time.time()

            line = f"{args.model},{args.backbone},{dataset_name}," \
                   f"{task_idx+1},{task_tot_images},{task_avg_shot:.1f},{task_way}," \
                   f"{task_nll:.5f}," \
                   f"{task_gce*100:.2f},{task_ece*100:.2f}," \
                   f"{task_ace*100:.2f},{task_tace*100:.2f}," \
                   f"{task_sce*100:.2f},{task_rmsce*100:.2f}," \
                   f"{task_top1:.2f}," \
                   f"{np.mean(all_top1):.2f},{all_top1_confidence:.2f}," \
                   f"{(time.time() - task_time):.2f}"
            log_write(args.log_path, line, mode="a", verbose=True)
            
        # Finished with this dataset, estimate the final statistics
        print(f"*{dataset_name} Accuracy: {np.mean(all_top1):.2f}+-{all_top1_confidence:.2f}, GCE: {np.mean(all_gce)*100.0:.2f}, ECE: {np.mean(all_ece)*100.0:.2f}, ACE: {np.mean(all_ace)*100.0:.2f}, TACE: {np.mean(all_tace)*100.0:.2f}, SCE: {np.mean(all_sce)*100.0:.2f}, RMSCE: {np.mean(all_rmsce)*100.0:.2f}, Episodes: {task_idx+1}, Time: {(time.time() - dataset_time):.2f} sec\n")
                                 
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", choices=["uppercase"], default="uppercase", help="The model used for the evaluation.")
  parser.add_argument("--backbone", choices=["BiT-S-R50x1", "ResNet18", "EfficientNetB0"], default="EfficientNetB0", help="The backbone used for the evaluation.")
  parser.add_argument("--adapter", choices=["case"], default="case", help="The adapted used.")
  parser.add_argument("--data_path", default="../datasets", help="Path to Meta-Dataset records.")
  parser.add_argument("--log_path", default="./log.csv", help="Path to log CSV file for the run.")
  parser.add_argument("--checkpoint_path", default="./checkpoints", help="Path to Meta-Dataset records.")
  parser.add_argument("--mode", choices=["train", "test", "train_test"], default="test",
                          help="Whether to run meta-training only, meta-testing only,"
                               "both meta-training and meta-testing.")
  parser.add_argument("--max_way_train", type=int, default=50, help="Maximum way of meta-train task.")
  parser.add_argument("--max_support_train", type=int, default=500,
                          help="Maximum support set size of meta-train task.")                                 
  parser.add_argument("--image_size", type=int, default=224, help="Image height and width.")
  parser.add_argument("--num_train_tasks", type=int, default=10000, help="Number of train tasks.")
  parser.add_argument("--num_test_tasks", type=int, default=600, help="Number of test tasks.")
  parser.add_argument("--num_validation_tasks", type=int, default=700, help="Number of validation tasks.")
  parser.add_argument("--resume_from", default="", help="Checkpoint path for the backbone.")
  parser.add_argument("--device", default="", help="Device to use.")
        
  args = parser.parse_args()
  print(os.path.abspath(os.environ['META_DATASET_ROOT']))
  main(args)
