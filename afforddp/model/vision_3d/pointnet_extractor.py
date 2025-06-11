import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy
import math
from einops import repeat
from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint
from afforddp.model.vision_3d.transformer import TransformerDecoder

def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules

class PointEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(PointEmbedding, self).__init__()
        self.embedding_layer = nn.Linear(3, embedding_dim)

    def forward(self, points):
        embedded_points = self.embedding_layer(points)
        return embedded_points



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=6, ):
        super().__init__()

        # Compute the positional encoding once
        self.pos_enc = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.pos_enc[:, 0::2] = torch.sin(pos * div_term)
        self.pos_enc[:, 1::2] = torch.cos(pos * div_term)
        self.pos_enc = self.pos_enc.unsqueeze(0).to('cuda')

        # Register the positional encoding as a buffer to avoid it being
        # considered a parameter when saving the model
        self.register_buffer('transformer_pos_enc', self.pos_enc)

    def forward(self, x):
        # Add the positional encoding to the input
        x = x + self.pos_enc[:, :x.size(1), :]
        return x

class PointNetEncoderXYZRGB(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256, 512]
        cprint("pointnet use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("pointnet use_final_norm: {}".format(final_norm), 'cyan')
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )
        
       
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    

class PointNetEncoderXYZ(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int=3,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256]
        cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), 'cyan')
        
        assert in_channels == 3, cprint(f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red")
       
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )
        
        
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not use projection", "yellow")
            
        VIS_WITH_GRAD_CAM = False
        if VIS_WITH_GRAD_CAM:
            self.gradient = None
            self.feature = None
            self.input_pointcloud = None
            self.mlp[0].register_forward_hook(self.save_input)
            self.mlp[6].register_forward_hook(self.save_feature)
            self.mlp[6].register_backward_hook(self.save_gradient)
         
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    
    def save_gradient(self, module, grad_input, grad_output):
        """
        for grad-cam
        """
        self.gradient = grad_output[0]

    def save_feature(self, module, input, output):
        """
        for grad-cam
        """
        if isinstance(output, tuple):
            self.feature = output[0].detach()
        else:
            self.feature = output.detach()
    
    def save_input(self, module, input, output):
        """
        for grad-cam
        """
        self.input_pointcloud = input[0].detach()


class AffordMLP(nn.Module):
    """Encoder for AffordMLP
    """

    def __init__(self,
                 in_channels: int=3,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):

        super().__init__()
        block_channel = [64]

        self.mlp = nn.Sequential(nn.Flatten(),
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU()
        )
        
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
 
        VIS_WITH_GRAD_CAM = False
        if VIS_WITH_GRAD_CAM:
            self.gradient = None
            self.feature = None
            self.input_pointcloud = None
            self.mlp[0].register_forward_hook(self.save_input)
            self.mlp[6].register_forward_hook(self.save_feature)
            self.mlp[6].register_backward_hook(self.save_gradient)
         
         
    def forward(self, x):
        x = self.mlp(x)
        x = self.final_projection(x)
        return x
    
    def save_gradient(self, module, grad_input, grad_output):
        """
        for grad-cam
        """
        self.gradient = grad_output[0]

    def save_feature(self, module, input, output):
        """
        for grad-cam
        """
        if isinstance(output, tuple):
            self.feature = output[0].detach()
        else:
            self.feature = output.detach()
    
    def save_input(self, module, input, output):
        """
        for grad-cam
        """
        self.input_pointcloud = input[0].detach()



class AffordanceTransformerEncoder(nn.Module):
    def __init__(self,
                affordance_encoder_size,
                max_seq_len,
                num_attention_heads: Optional[int] = 2,
                num_attention_layers: Optional[int] = 4,
                ff_dim_factor: Optional[int] = 4,
                dropout: Optional[int] = 0.1,
                encoder_activation: Optional[str] = 'gelu',
                use_positional_encoding: Optional[bool] = True):
        
        super().__init__()

        self.use_positional_encoding = use_positional_encoding
        self.embedding = PointEmbedding(affordance_encoder_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, affordance_encoder_size))
        if self.use_positional_encoding:
            self.positional_encoding = PositionalEncoding(affordance_encoder_size, max_seq_len)

        self.sa_layer = nn.TransformerEncoderLayer(
            d_model=affordance_encoder_size, 
            nhead=num_attention_heads, 
            dim_feedforward=ff_dim_factor*affordance_encoder_size, 
            activation=encoder_activation,
            dropout=dropout, 
            batch_first=True, 
            norm_first=True
        )
        self.sa_encoder = nn.TransformerEncoder(self.sa_layer, num_layers=num_attention_layers)

    def forward(self, affordance):

        b, l, c = affordance.shape
        affordance_embedding = self.embedding(affordance)
        cls_token = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        affordance_embedding = torch.concat((cls_token, affordance_embedding), dim=1)
        if self.use_positional_encoding:
            affordance_embedding = self.positional_encoding(affordance_embedding)
        affordance_encoding_tokens = self.sa_encoder(affordance_embedding)
        affordance_encoding_tokens = affordance_encoding_tokens[:,0,:]
        return affordance_encoding_tokens

class AffordanceTransformerDecoder(nn.Module):
    def __init__(self,
                affordance_encoder_size,
                max_seq_len,
                num_attention_heads: Optional[int] = 2,
                num_attention_layers: Optional[int] = 2,
                head_output_size: Optional[int] = 64,
                mlp_hidden_size: Optional[int] = 256,
                dropout: Optional[int] = 0.1,
                use_positional_encoding: Optional[bool] = True):
        
        super().__init__()

        self.use_positional_encoding = use_positional_encoding
        self.embedding = PointEmbedding(affordance_encoder_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, affordance_encoder_size))
        if self.use_positional_encoding:
            self.positional_encoding = PositionalEncoding(affordance_encoder_size, max_seq_len)

        self.sa_encoder = TransformerDecoder(affordance_encoder_size, num_heads=num_attention_heads,
                                                     num_layers=num_attention_layers,head_output_size=head_output_size,
                                                     mlp_hidden_size=mlp_hidden_size,dropout=dropout)

    def forward(self, affordance):

        b, l, c = affordance.shape
        affordance_embedding = self.embedding(affordance)
        cls_token = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        affordance_embedding = torch.concat((cls_token, affordance_embedding), dim=1)
        if self.use_positional_encoding:
            affordance_embedding = self.positional_encoding(affordance_embedding)
        affordance_encoding_tokens = self.sa_encoder(affordance_embedding)
        affordance_encoding_tokens = affordance_encoding_tokens[:,0,:]
        return affordance_encoding_tokens



class DynamicAffordDPEncoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict,
                 afford_space, 
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 afford_mlp_size=(64, 64),afford_mlp_activation_fn=nn.ReLU,
                 affordance_encoder_size = 64,
                 max_seq_len = 5,
                 pointcloud_encoder_cfg=None,
                 num_attention_heads = 2,
                 num_attention_layers = 2,
                 use_pc_color=False,
                 pointnet_type='pointnet',
                 ):
        super().__init__()
        self.imagination_key = 'imagin_robot'
        self.state_key = 'state'
        self.point_cloud_key = 'point_cloud'
        self.rgb_image_key = 'image'
        self.afford_key = 'afford'
        self.n_output_channels = out_channel
        
        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.afford_shape = afford_space
        self.state_shape = observation_space[self.state_key]
     
        cprint(f"[AffordDPEncoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[AffordDPEncoder] state shape: {self.state_shape}", "yellow")


        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        if pointnet_type == "pointnet":
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
                self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
            else:
                pointcloud_encoder_cfg.in_channels = 3
                self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")


        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        if len(afford_mlp_size) == 0:
            raise RuntimeError(f"afford mlp size is empty")
        else:
            afford_net_arch = afford_mlp_size[:-1]

        afford_output_dim = afford_mlp_size[-1]

        self.traj_transformer = AffordanceTransformerEncoder(affordance_encoder_size, max_seq_len, 
                                                               num_attention_heads=num_attention_heads,
                                                               num_attention_layers=num_attention_layers)
        self.afford_mlp = nn.Sequential(*create_mlp(3,afford_output_dim,afford_net_arch,afford_mlp_activation_fn))
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))

        self.n_output_channels  += (output_dim + affordance_encoder_size + afford_output_dim)
        # self.n_output_channels  += (affordance_encoder_size)
    
        cprint(f"[AffordDPEncoder] output dim: {self.n_output_channels}", "red")

    def forward(self, observations: Dict, afford) -> torch.Tensor:
        points = observations[self.point_cloud_key]
        assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")

        afford = afford.to(torch.float32)
        
        # points = torch.transpose(points, 1, 2)   # B * 3 * N
        # points: B * 3 * (N + sum(Ni))
        pn_feat = self.extractor(points)    # B * out_channel  (B,64)
            
        state = observations[self.state_key]
        state_feat = self.state_mlp(state)  # B * 64
        
        traj_feat = self.traj_transformer(afford) # B * N * 3
        contact_feat = self.afford_mlp(afford[:,0,:])
        final_feat = torch.cat([pn_feat, state_feat, contact_feat, traj_feat], dim=1)
        # final_feat = torch.cat([state_feat, afford_feat], dim=1)
        return final_feat

    def output_shape(self):
        return self.n_output_channels