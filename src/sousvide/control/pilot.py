import numpy as np
import torch
import time
import os
import json
import numpy.typing as npt
import albumentations as A

from typing import Dict,Union,Tuple,Literal
from scipy.spatial.transform import Rotation
from albumentations.pytorch import ToTensorV2

import sousvide.control.policies.generate_networks as gn

class Pilot():
    def __init__(self,cohort_name:str,pilot_name:str,
                 hz:int=20,Ntx:int=11,Nu:int=4):
        """
        Initializes a pilot object. 
        
        Args:
            cohort_name:    Name of the cohort.
            pilot_name:     Name of the pilot.
        
        Variables:
            name:           Name of the pilot.
            hz:             Frequency of the pilot.
            path:           Path to the pilot.
            policy_type:    Type of policy.

            device:         Device to run the pilot on.
            model:          Neural network model.
            advisor:        Oracle model.

            tx_cr:          Current state.
            txu_pr:         Previous state.
            znn_cr:         Current feature vector.
            preprocess:     Image preprocessing function.

            Obj:            Objective vector.
            Img:            Image vector.

            hy_flag:        Flag to check if history is initialized.
            hy_idx:         Index of history.

            DxU:            Delta transforms.
            Znn:            Feature vectors.

            da_cfg:         Data augmentation configuration.

        """

        ## Initial Variables ===============================================================================================

        # Some useful constants
        transform = A.Compose([                                             # Image transformation pipeline
                A.Resize(256, 256),
                A.CenterCrop(224, 224),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
                ])            
        process_image = lambda x: transform(image=x)["image"]               # Image processing
        nhy = 20                                                            # Number of history states to keep
        obj_dim = [18,1]                                                    # Objective dimensions
        img_dim = [3,224,224]                                               # Image dimensions
        
        # Some useful paths
        workspace_path  = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        default_config_path = os.path.join(
            workspace_path,"configs","pilots",pilot_name+".json")
        pilot_path = os.path.join(
            workspace_path,"cohorts",cohort_name,'roster',pilot_name)
        pilot_config_path = os.path.join(
            pilot_path,"config.json")
        
        # Check if config file exists, if not create one
        if os.path.isfile(pilot_config_path):
            pass
        else:
            os.makedirs(pilot_path,exist_ok=True)
            with open(default_config_path) as json_file:
                profile = json.load(json_file)

            with open(pilot_config_path, 'w') as outfile:
                json.dump(profile, outfile, indent=4)

        # Load config file
        with open(pilot_config_path) as json_file:
            profile = json.load(json_file) 

        # Torch intermediate variables
        use_cuda = torch.cuda.is_available()                                    # Check if cuda is available

        ## Class Variables =================================================================================================
        
        # ---------------------------------------------------------------------
        # Pilot Identifier Variables
        # ---------------------------------------------------------------------

        # Necessary Variables for Base Controller -----------------------------
        self.name = pilot_name
        self.hz = hz
        self.nzcr = None

        self.path = pilot_path
        self.policy_type = "None" if profile["type"] is None else profile["type"]

        # ---------------------------------------------------------------------
        # Pilot Model Variables
        # ---------------------------------------------------------------------

        self.device  = torch.device("cuda:0" if use_cuda else "cpu")
        self.model,Nz = gn.policy_factory(pilot_path,profile,self.device)
        # self.advisor = gn.advisor_factory(pilot_path,profile,self.device)

        # ---------------------------------------------------------------------
        # Pilot Observe Variables
        # ---------------------------------------------------------------------

        # Function Variables
        self.txu_pr = torch.zeros(Ntx+Nu).to(self.device)                   # Previous State
        self.znn_cr = torch.zeros(Nz).to(self.device)                       # Current Feature Vector
        self.process_image = process_image                                  # Image Processing Function

        # Network Input Variables
        self.tx_cr = torch.zeros(Ntx).to(self.device)                       # Current State
        self.Obj = torch.zeros(obj_dim).to(self.device)                     # Objective
        self.Img = torch.zeros(img_dim).to(self.device)                     # Image

        # ---------------------------------------------------------------------
        # Pilot Orient Variables
        # ---------------------------------------------------------------------

        # Function Variables
        self.hy_flag,self.hy_idx = False,0

        # Network Input Variables
        self.DxU = torch.zeros((Ntx+Nu,nhy)).to(self.device)                # Delta Transforms
        self.Znn = torch.zeros((Nz,nhy)).to(self.device)                    # Feature Vectors
        
        # ---------------------------------------------------------------------
        # Pilot Training Variables
        # ---------------------------------------------------------------------
                
        # Data Augmentation
        self.da_cfg = {
            "type": profile["data_augmentation"]["type"],
            "mean": profile["data_augmentation"]["mean"],
            "std": profile["data_augmentation"]["std"]
        }

        # ---------------------------------------------------------------------

    def set_mode(self,mode:Literal['train','deploy']):
        """
        Function that switches the pilot between training mode and deployment.

        Args:
            mode: Mode to be switched to: 'train' or 'deploy'.

        Returns:
            None
        """

        if mode == 'train':
            self.model.train()                              # Set model to training mode
        elif mode == 'deploy':
            self.model.eval()                               # Set model to evaluation mode

            xnn = self.decide()                             # Create dummy information to initialize the model
            self.act(xnn)                                   # Do a 'wipe out' to initialize the model
        else:
            raise ValueError("Invalid mode. Must be 'train' or 'deploy'.")
    
    def observe(self,
                upr:Union[np.ndarray,torch.Tensor],
                tcr:Union[float,torch.Tensor],xcr:Union[np.ndarray,torch.Tensor],
                obj:Union[np.ndarray,torch.Tensor],
                icr:Union[npt.NDArray[np.uint8],None,torch.Tensor],zcr:Union[torch.Tensor,None]) -> None:
        """
        Function that performs the observation step of the OODA loop where we take in all the relevant
        flight data.

        Args:
            upr:    Previous control input.
            tcr:    Current flight time.
            xcr:    Current state in world frame (observed).
            obj:    Objective vector.
            img_cr: Current image frame.
            znn_cr: Current feature vector.

        Returns:
            None
        """

        # Convert Non-Torch Tensor Variables to Torch Tensors on GPU
        upr = torch.tensor(upr,dtype=torch.float32).to(self.device)
        tcr = torch.tensor(tcr,dtype=torch.float32).to(self.device)
        xcr = torch.from_numpy(xcr).float().to(self.device)
        obj = torch.from_numpy(obj).float().to(self.device)
        zcr = zcr.to(self.device) if zcr is not None else torch.zeros(0).to(self.device)

        # Process image if it is not downsampled
        if icr is None or icr.shape != self.Img.shape:
            icr = self.process_image(icr)
        else:
            if isinstance(icr,np.ndarray):
                icr = torch.from_numpy(icr).float()

        # Update Function Variables
        self.txu_pr.copy_(torch.cat((self.tx_cr,upr)).flatten())
        self.tx_cr.copy_(torch.cat((tcr.unsqueeze(0),xcr)).flatten())
        self.znn_cr.copy_(zcr)

        # Update Network Input Variables
        self.Obj.copy_(obj.reshape(self.Obj.shape))
        self.Img.copy_(icr)
            
    def orient(self):
        """
        Function that performs the orientation step of the OODA loop where we generate the history
        variables from the flight data.

        Args:
            None

        Returns:
            None
        """

        # Update history data (if it exists)
        if self.hy_flag == False:
            self.hy_flag = True                                                     # Skip transform calculation for first iteration since there is no previous state
        else:
            q_cr = self.tx_cr[7:11].cpu().numpy()                                   # Current Quaternion
            q_pr = self.txu_pr[7:11].cpu().numpy()                                  # Previous Quaternion

            crRi = Rotation.from_quat(q_cr).as_matrix()                             # Current to Initial
            prRi = Rotation.from_quat(q_pr).as_matrix()                             # Previous to Initial
            dq:np.ndarray = Rotation.from_matrix(crRi.T@prRi).as_quat()             # Quaternion difference

            # Update History Information
            self.DxU[0,self.hy_idx] = self.tx_cr[0]-self.txu_pr[0]                  # Update History Time
            self.DxU[1:4,self.hy_idx] = self.tx_cr[4:7]                             # Update History delta position
            self.DxU[4:7,self.hy_idx] = self.tx_cr[4:7]-self.txu_pr[4:7]            # Update History delta velocity
            
            # TODO: Check if this is correct
            self.DxU[7:11,self.hy_idx] = torch.from_numpy(dq.astype(np.float32))    # Update History delta quaternion
            # self.DxU[7:11,self.hy_idx] = self.txu_pr[7:11]                          # Update History Quaternion
            # ==========================================================================================================

            self.DxU[11:15,self.hy_idx] = self.txu_pr[11:15]                        # Update History Input
            self.Znn[:,self.hy_idx] = self.znn_cr                                   # Update History Odometry

            # Increment History Index
            self.hy_idx += 1                                                        # Increment History index
            if self.hy_idx >= self.DxU.shape[1]:                                    # If history blocks are full, reset index
                self.hy_idx = 0
        
    def decide(self) -> Dict[str,torch.Tensor]:
        """
        Function that performs the decision step of the OODA loop where we decide on the terms to put into the
        neural network model. The input terms are extracted from the current state (with time), objective, image
        and history data (delta transforms and feature vectors). The output is a dictionary of the input terms.

        Args:
            None

        Returns:
            xnn:    Input to the neural network model.
        """
        xnn = self.model.extract_inputs(self.tx_cr,self.Obj,self.Img,self.DxU,self.Znn,self.hy_idx)

        return xnn
    
    def act(self,xnn: Dict[str,torch.Tensor]) -> Tuple[np.ndarray,torch.Tensor,Union[np.ndarray,None]]:

        """
        Function that performs a forward pass of the neural network model using the information
        vector as input.

        Args:
            xnn:    Input to the neural network model.

        Returns:
            unn:    Output from the neural network model.
            znn:    Output from the image processing network module.
            adv:    Oracle output
        """
        with torch.no_grad():
            inputs = self.model.get_commander_inputs(xnn)
            unn,znn = self.model(*inputs)

        # Convert inputs to numpy array
        unn = unn.cpu().numpy().squeeze()
        # znn = znn.cpu().numpy().squeeze()
        
        # Advisor Output (empty for now)
        adv = None

        return unn,znn,adv

    def OODA(self,
             upr:np.ndarray,
             tcr:float,xcr:np.ndarray,
             obj:np.ndarray,
             icr:Union[npt.NDArray[np.uint8],None],zcr:Union[torch.Tensor,None]) -> Tuple[
                 np.ndarray,
                 torch.Tensor,
                 Union[np.ndarray,None],
                 Dict[str,torch.Tensor],
                 np.ndarray]:
        
        """
        Function that runs the OODA loop. This is the main function that is called by the
        pilot during flight.

        Args:
            upr:    Previous control input.
            tcr:    Current flight time.
            xcr:    Current state in world frame (observed).
            obj:    Objective vector.
            icr:    Current image frame.
            zcr:    Current odometry information.

        Returns:
            unn:    Output from the neural network model.
            znn:    Output from the image processing network.
            adv:    Oracle output
            xnn:    Input to the neural network model.
            tsol:   Time taken to solve components of the OODA loop in list form.
        """
        
        # Get the current time
        t0 = time.time()

        # Perform the OODA loop
        self.observe(upr,tcr,xcr,obj,icr,zcr)
        t1 = time.time()
        self.orient()
        t2 = time.time()
        xnn = self.decide()
        t3 = time.time()
        unn,znn,adv = self.act(xnn)
        t4 = time.time()

        # Get the total time taken
        tsol = np.array([t1-t0,t2-t1,t3-t2,t4-t3])

        return unn,znn,adv,xnn,tsol
    
    def control(self,
                tcr:float,xcr:np.ndarray,
                upr:np.ndarray,
                obj:np.ndarray,
                icr:Union[npt.NDArray[np.uint8],None],zcr:Union[torch.Tensor,None]) -> Tuple[
                    np.ndarray,
                    torch.Tensor,
                    Union[np.ndarray,None],
                    np.ndarray]:
        """
        Name mask for the OODA control loop. Variable position swap to match generic controllers.
        
        Args:
            tcr:    Current flight time.
            xcr:    Current state in world frame (observed).
            upr:    Previous control input.
            obj:    Objective vector.
            icr:    Current image frame.
            zcr:    Input feature vector.
        
        Returns:
            unn:    Control input.
            znn:    Output feature vector.
            adv:    Oracle output
            tsol:   Time taken to solve components of the OODA loop in list form.
        """
        unn,znn,adv,_,tsol = self.OODA(upr,tcr,xcr,obj,icr,zcr)
        
        return unn,znn,adv,tsol