import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from tqdm.notebook import trange
from sousvide.control.pilot import Pilot
from sousvide.instruct.synthesized_data import *
from typing import List,Tuple,Literal
from enum import Enum

def train_roster(cohort_name:str,roster:List[str],
                 mode:Literal["Parameter","Odometry","Commander"],
                 Neps:int,lim_sv:int,
                 lr:float=1e-4,batch_size:int=64):

    for student_name in roster:
        # Load Student
        student = Pilot(cohort_name,student_name)
        student.set_mode('train')

        train_student(cohort_name,student,mode,Neps,lim_sv,lr,batch_size)

def train_student(cohort_name:str,student:Pilot,
                  mode:Literal["Parameter","Odometry","Commander"],
                  Neps:int,lim_sv:int,lr:float,batch_size:int):

    # Pytorch Config
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    criterion = nn.MSELoss(reduction='mean')
    scaler = torch.cuda.amp.GradScaler()

    # Send to GPU
    student.model.to(device)

    # Select model components
    if mode in student.model.get_network:
        model = student.model.get_network[mode]["Train"]
    else:
        print("--------------------------------------------------------------------------")
        print(student.name,"has no model for",mode)
        print("--------------------------------------------------------------------------")

        return

    # Set parameters to optimize over
    opt = optim.Adam(student.model.get_network[mode]["Unlock"].parameters(),lr=lr)
    num_params = sum(p.numel() for group in opt.param_groups for p in group['params'])
    print(f"Total number of parameters Adam is optimizing across all groups: {num_params}")

    # Some Useful Paths
    student_path = student.path
    model_path   = os.path.join(student_path,"model.pth")
    losses_path  = os.path.join(student_path,"losses_"+mode+".pt")

    # Say who we are training
    print("Training Student: ",student.name)
    
    # Setup Loss Variables (load if exists)
    losses: Dict[str, List] = {}
    if os.path.exists(losses_path):
        losses = torch.load(losses_path)

        print("Re-training existing network.")
        print("Previous Epochs: ", sum(losses["Neps"]), losses["Neps"])

        losses["train"].append([]),losses["test"].append([])
        losses["Neps"].append(0),losses["Nspl"].append(0),losses["t_train"].append(0)
    else:
        losses = {
            "train": [None], "test": [None],
            "Neps": [None], "Nspl": [None], "t_train": [None]
        }
        print("Training new network.")

    Loss_train,Loss_tests = [],[]

    # Record the start time
    start_time = time.time()

    # Training + Testing Loop
    with trange(Neps) as eps:
        for ep in eps:
            # Lock/Unlock Networks
            unlock_networks(student,mode)

            # Get Observation Data Files (Paths)
            od_train_files,od_test_files = get_data_paths(cohort_name,student.name)

            # Training
            loss_log_tn =[]
            for od_train_file in od_train_files:
                # Load Datasets
                dataset = generate_dataset(od_train_file,student,mode,device)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last = False)

                # Training
                for input,label in dataloader:
                    # Move to GPU
                    input,label = tuple(tensor.to(device) for tensor in input),label.to(device)

                    # Forward Pass
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        prediction,_ = model(*input)
                        loss = criterion(prediction,label)

                    # Backward Pass
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad()

                    # Save loss logs
                    loss_log_tn.append((label.shape[0],loss.item()))

            # Testing
            log_log_tt = []
            for od_test_file in od_test_files:
                # Load Datasets
                dataset = generate_dataset(od_test_file,student,mode,device)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last = False)

                # Testing
                for input,label in dataloader:
                    # Move to GPU
                    input,label = tuple(tensor.to(device) for tensor in input),label.to(device)

                    # Forward Pass
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        prediction,_ = model(*input)
                        loss = criterion(prediction,label)

                    # Save loss logs
                    log_log_tt.append((label.shape[0],loss.item()))

            # Loss Diagnostics
            Ntn = np.sum([n for n,_ in loss_log_tn])
            Ntt = np.sum([n for n,_ in log_log_tt])
            loss_train = np.sum([n*loss for n,loss in loss_log_tn])/Ntn
            loss_tests = np.sum([n*loss for n,loss in log_log_tt])/Ntt

            eps.set_description('Loss %f' % loss_train)
            Loss_train.append(loss_train)
            Loss_tests.append(loss_tests)

            # Save at intermediate steps and at the end
            if ((ep+1) % lim_sv == 0) or (ep+1==Neps):
                # Lock the networks
                unlock_networks(student,"None")

                # Record the end time
                end_time = time.time()
                t_train = end_time - start_time

                torch.save(student.model,model_path)

                losses["train"][-1] = Loss_train
                losses["test"][-1] = Loss_tests
                losses["Neps"][-1] = ep+1
                losses["Nspl"][-1] = Ntn
                losses["t_train"][-1] = t_train

                # Save Loss
                torch.save(losses,losses_path)

def unlock_networks(student:Pilot,
                  target:Literal["All","None","Parameter","Odometry","Commander"]):
    """
    Locks/Unlocks the networks based on the training mode.
    
    """

    if target == "All":
        for param in student.model.parameters():
            param.requires_grad = True
    else:
        for param in student.model.parameters():
            param.requires_grad = False
        
        if target == "None":
            return
        elif target == "Parameter":
            for param in student.model.get_network["Parameter"]["Unlock"].parameters():
                param.requires_grad = True
        elif target == "Commander":
            for param in student.model.get_network["Commander"]["Unlock"].parameters():
                param.requires_grad = True
        elif target == "Odometry":
            for param in student.model.get_network["Odometry"]["Unlock"].parameters():
                param.requires_grad = True
        else:
            raise ValueError("Invalid Training Mode")
