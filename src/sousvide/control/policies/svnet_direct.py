import torch
from torch import nn
from sousvide.control.policies.ComponentNetworks import *
from typing import Dict,Tuple,Any,Callable

class SVNetDirect(nn.Module):    
    def __init__(self,
                 config:Dict[str,Any],
                 name:str="SVNetDirect"):
        
        """
        Defines a pilot's neural network model.

        Network Description:
        SV-Net with direct history encoding (versus latent). This policy is identical
        to SV-Net but uses a direct estimate of the mass and thrust coefficient as the
        input into the command network (versus the penultimate layer in SV-Net). Just
        like regular SV-Net, the history network is pre-trained (under Parameter) and
        then locked during training of the command network (under Commander).

        Args:
            config:         Configuration dictionary for the model.
            name:           Name of the policy.

        Variables:
            name:           Name of the policy.
            Nz:             Feature vector size.
            network:        Dictionary of the network components.
            get_data:       Dictionary of functions to extract data for the network.
            get_network:    Dictionary of network components for training.
        """

        # Initial Parent Call
        super(SVNetDirect,self).__init__()

        # ----------------------------------------------------------------------------------------------
        # Class Intermediate Variables
        # ----------------------------------------------------------------------------------------------

        history_enc_cfg = config["networks"][0]
        vision_mlp_cfg = config["networks"][1]
        command_sv_cfg = config["networks"][2]

        history_enc_network = DirectHistoryEncoder(
                history_enc_cfg["delta"],
                history_enc_cfg["frames"],
                history_enc_cfg["hidden_sizes"],
                history_enc_cfg["output_size"]
            )
        vision_mlp_network = VisionMLP(
                vision_mlp_cfg["state"],
                vision_mlp_cfg["state_layer"],
                vision_mlp_cfg["hidden_sizes"],
                vision_mlp_cfg["output_size"],
                vision_mlp_cfg["CNN"]
            )
        commander_sv_network = CommandSV(
                command_sv_cfg["state"],
                command_sv_cfg["objective"],
                history_enc_cfg["output_size"],
                vision_mlp_cfg["output_size"],
                command_sv_cfg["hidden_sizes"],
                command_sv_cfg["output_size"]
            )
        
        # ----------------------------------------------------------------------------------------------
        # Class Name Variables
        # ----------------------------------------------------------------------------------------------

        # Network ID
        self.name = name
        self.Nz = 0

        # Network Components           
        self.network = nn.ModuleDict({
            "HistoryEncoder": history_enc_network,
            "VisionMLP": vision_mlp_network,
            "CommanderSV": commander_sv_network
        })

        # Network Callers [Parameter,Odometry,Commander]
        self.get_data:Dict[str,Callable] = {                         
            "Parameter": self.get_parameter_data,
            "Commander": self.get_commander_data,
            "Images": self.get_image_data
        }

        self.get_network:Dict[str,Dict[str,nn.Module]] = {
            "Parameter": {
                "Train" : self.network["HistoryEncoder"],
                "Unlock": self.network["HistoryEncoder"]
            },
            "Commander": {
                "Train" :self,
                "Unlock":nn.ModuleList([self.network["CommanderSV"],self.network["VisionMLP"]])
            }
        }

    def extract_inputs(self,tx:torch.Tensor,Obj:torch.Tensor,
                       Img:torch.Tensor,DxU:torch.Tensor,Znn:torch.Tensor,hy_idx:int) -> Dict[str,torch.Tensor]:
        """
        Takes the basis input variables and extracts the relevant inputs for the model. If the
        model component does not exist, the input is set to None.

        Args:
            tx:     Current state.
            Obj:    Objective.
            Img:    Image.
            DxU:    History deltas.
            Vio:    Odometry.
            hy_idx: History indices.

        Returns:
            xnn:    Extracted inputs for the model

        """

        # Unused Variables
        _ = Znn

        # Extract Indices
        ix_DxU_par = self.network["HistoryEncoder"].ix_DxU
        ix_tx_vis  = self.network["VisionMLP"].ix_tx
        ix_tx_com  = self.network["CommanderSV"].ix_tx
        ix_obj_com = self.network["CommanderSV"].ix_obj

        # Extract Data
        xnn = {
            "tx_com" : tx[ix_tx_com].flatten(),
            "obj_com": Obj[ix_obj_com].flatten(),
            "dxu_par": DxU[ix_DxU_par[0],hy_idx+ix_DxU_par[1]].flatten(),
            "img_vis": Img.clone(),
            "tx_vis" : tx[ix_tx_vis].flatten()
        }

        return xnn

    def get_commander_inputs(self,xnn: Dict[str,torch.Tensor]) -> Tuple[torch.Tensor,...]:
        return (xnn["tx_com"],xnn["obj_com"],xnn["dxu_par"],xnn["img_vis"],xnn["tx_vis"])
    
    def get_parameter_data(self,xnn:Dict[str,torch.Tensor],ynn:Dict[str,torch.Tensor]) -> Tuple[Tuple[torch.Tensor],torch.Tensor]:
        return (xnn["dxu_par"],),ynn["mfn"]
    
    def get_commander_data(self,xnn: Dict[str,torch.Tensor],ynn: Dict[str,torch.Tensor]) -> Tuple[Tuple[torch.Tensor,...],torch.Tensor]:
        return self.get_commander_inputs(xnn),ynn["unn"]

    def get_image_data(self,xnn: Dict[str,torch.Tensor]) -> torch.Tensor:
        return xnn["img_vis"]
    
    def forward(self,tx_com:torch.Tensor,obj_com:torch.Tensor,
                dxu_par:torch.Tensor,
                img_vis:torch.Tensor,tx_vis:torch.Tensor) -> Tuple[torch.Tensor,None]:
        """
        Defines the forward pass of the neural network model.

        Args:
            tx_com:     Current state for commander network.
            obj_com:    Objective for commander network.
            dxu_par:    History deltas for parameter network.
            img_vis:    Image for vision network.
            tx_vis:     Current state for vision network.

        Returns:
            ycm: Command network output.
            zcm: Empty.
        """

        # Parameter Network
        y_par,_ = self.network["HistoryEncoder"](dxu_par)

        # Command Network
        y_vis,_ = self.network["VisionMLP"](img_vis,tx_vis)
        y_com,_ = self.network["CommanderSV"](tx_com,obj_com,y_par,y_vis)

        return y_com,y_par