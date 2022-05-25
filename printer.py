import numpy as np
import pandas as pd
import argparse

def main(args):

    test_set = ["omniglot", "aircraft", "cu_birds", "dtd", "quickdraw", "fungi", "traffic_sign", "mscoco"]
    
    df = pd.read_csv(args.log_path)
    all_top1, all_loss, all_time = [], [], []
    all_gce, all_ece, all_ace, all_tace, all_sce, all_rmsce= [], [], [], [], [], []
    
    gce_latex =   "GCE & "
    ece_latex =   "ECE & "
    ace_latex =   "ACE & "
    tace_latex =  "TACE & "
    sce_latex =   "SCE & "
    rmsce_latex = "RMSCE & "
        
    for dataset_name in test_set:
        tot_tasks = len(df.loc[df["dataset"]==dataset_name])
        if(tot_tasks>0):
            print("Dataset:", dataset_name)
            print("Tot-Tasks:", tot_tasks)
            if("task-top1" in df.columns):
                top1 = df.loc[df["dataset"]==dataset_name]["task-top1"]
                top1_mean = top1.mean()
                top1_confidence = (196.0 * np.std(top1/100.)) / np.sqrt(len(top1))
                all_top1.append(top1_mean)
                print(f"TOP-1: {top1_mean:.2f} +- {top1_confidence:.2f}")
            if("task-loss" in df.columns):
                loss = df.loc[df["dataset"]==dataset_name]["task-loss"]
                loss_mean = loss.mean()
                loss_confidence = (196.0 * np.std(loss/100.)) / np.sqrt(len(loss))
                all_loss.append(loss_mean)
                print(f"Loss: {loss_mean:.5f} +- {loss_confidence:.2f}")
            if("task-gce" in df.columns):
                gce = df.loc[df["dataset"]==dataset_name]["task-gce"]
                gce_mean = gce.mean()
                gce_confidence = (196.0 * np.std(gce/100.)) / np.sqrt(len(gce))
                all_gce.append(gce_mean)
                if(args.print_latex): gce_latex += str(round(gce_mean,1))+"$\pm$"+str(round(gce_confidence,1)) + " & "
                print(f"GCE: {gce_mean:.2f} +- {gce_confidence:.2f}")
            if("task-ece" in df.columns):
                ece = df.loc[df["dataset"]==dataset_name]["task-ece"]
                ece_mean = ece.mean()
                ece_confidence = (196.0 * np.std(ece/100.)) / np.sqrt(len(ece))
                all_ece.append(ece_mean)
                if(args.print_latex): ece_latex += str(round(ece_mean,1))+"$\pm$"+str(round(ece_confidence,1)) + " & "
                print(f"ECE: {ece_mean:.2f} +- {ece_confidence:.2f}")
            if("task-ace" in df.columns):
                ace = df.loc[df["dataset"]==dataset_name]["task-ace"]
                ace_mean = ace.mean()
                ace_confidence = (196.0 * np.std(ace/100.)) / np.sqrt(len(ace))
                all_ace.append(ace_mean)
                if(args.print_latex): ace_latex += str(round(ace_mean,1))+"$\pm$"+str(round(ace_confidence,1)) + " & "
                print(f"ACE: {ace_mean:.2f} +- {ace_confidence:.2f}")
            if("task-tace" in df.columns):
                tace = df.loc[df["dataset"]==dataset_name]["task-tace"]
                tace_mean = tace.mean()
                tace_confidence = (196.0 * np.std(tace/100.)) / np.sqrt(len(tace))
                all_tace.append(tace_mean)
                if(args.print_latex): tace_latex += str(round(tace_mean,1))+"$\pm$"+str(round(tace_confidence,1)) + " & "
                print(f"TACE: {tace_mean:.2f} +- {tace_confidence:.2f}")
            if("task-sce" in df.columns):
                sce = df.loc[df["dataset"]==dataset_name]["task-sce"]
                sce_mean = sce.mean()
                sce_confidence = (196.0 * np.std(sce/100.)) / np.sqrt(len(sce))
                all_sce.append(sce_mean)
                if(args.print_latex): sce_latex += str(round(sce_mean,1))+"$\pm$"+str(round(sce_confidence,1)) + " & "
                print(f"SCE: {sce_mean:.2f} +- {sce_confidence:.2f}")
            if("task-rmsce" in df.columns):
                rmsce = df.loc[df["dataset"]==dataset_name]["task-rmsce"]
                rmsce_mean = rmsce.mean()
                rmsce_confidence = (196.0 * np.std(rmsce/100.)) / np.sqrt(len(rmsce))
                all_rmsce.append(rmsce_mean)
                if(args.print_latex): rmsce_latex += str(round(rmsce_mean,1))+"$\pm$"+str(round(rmsce_confidence,1)) + " & "
                print(f"RMSCE: {rmsce_mean:.2f} +- {rmsce_confidence:.2f}")
            if("task-tot-images" in df.columns):
                values = df.loc[df["dataset"]==dataset_name]["task-tot-images"]
                values_mean = values.mean()
                print(f"Avg-Images: {values_mean:.1f}")
            if("task-way" in df.columns):
                values = df.loc[df["dataset"]==dataset_name]["task-way"]
                values_mean = values.mean()
                print(f"Avg-Way: {values_mean:.1f}")
            if("task-avg-shot" in df.columns):
                values = df.loc[df["dataset"]==dataset_name]["task-avg-shot"]
                values_mean = values.mean()
                print(f"Avg-Shot: {values_mean:.1f}")
            if("time" in df.columns):
                tot_time = df.loc[df["dataset"]==dataset_name]["time"].sum()
                all_time.append(tot_time)
                print(f"Time: {tot_time/60.0:.1f} min")              
            print("")
     
    # Finished, printing overall statistics
    print("-------------------")   
    if(len(all_top1)>0):   print(f"TOP-1 ... {np.mean(all_top1):.1f}%")
    if(len(all_loss)>0):   print(f"Loss .... {np.mean(all_loss):.5f}")
    if(len(all_gce)>0):    print(f"GCE ..... {np.mean(all_gce):.1f}%")
    if(len(all_ece)>0):    print(f"ECE ..... {np.mean(all_ece):.1f}%")
    if(len(all_ace)>0):    print(f"ACE ..... {np.mean(all_ace):.1f}%")
    if(len(all_tace)>0):   print(f"TACE .... {np.mean(all_tace):.1f}%")
    if(len(all_sce)>0):    print(f"SCE ..... {np.mean(all_sce):.1f}%")
    if(len(all_rmsce)>0):  print(f"RMSCE ... {np.mean(all_rmsce):.1f}%")
    if(len(all_time)>0):   print(f"Time .... {np.sum(all_time)/60.0:.1f} min, {(np.sum(all_time)/60.0)/60.0:.1f} hour")
    print("-------------------")
    
    if(args.print_latex):
        # Removing last char and adding new-line symbol
        gce_latex =   gce_latex[:-2] + "\\" + "\\"
        ece_latex =   ece_latex[:-2] + "\\" + "\\"
        ace_latex =   ace_latex[:-2] + "\\" + "\\"
        tace_latex =  tace_latex[:-2] + "\\" + "\\"
        sce_latex =   sce_latex[:-2] + "\\" + "\\"
        rmsce_latex = rmsce_latex[:-2] + "\\" + "\\"
        
        print("\nLatex strings:")
        print(gce_latex)
        print(ece_latex)
        print(ace_latex)
        print(tace_latex)
        print(sce_latex)
        print(rmsce_latex)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", default="./log.csv", help="Path to CSV file with the test log.")
    parser.add_argument('--print_latex', dest='print_latex', action='store_true', help="Print latex strings.")
    args = parser.parse_args()
    main(args)
