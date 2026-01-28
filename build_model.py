import torch
from torch import nn

from models.mn.model import mobilenet_v3
from helpers.utils import NAME_TO_WIDTH  # exists in EfficientAT repo

def build_model_from_efficientat():
    pretrained_name = "mn10_as"       # must match the checkpoint family
    width = NAME_TO_WIDTH(pretrained_name)  # e.g., 4.0 for mn40_as

    num_classes = 527
    pretrained_name = None
    width_mult = 1.0
    reduced_tail = False
    dilated = False
    strides = (2, 2, 2, 2)
    head_type = "mlp"
    multihead_attention_heads = 4
    input_dim_f = 128
    input_dim_t = 300
    se_dims = 'c'
    se_agg = "max"
    se_r= 4
    
    input_dims = (input_dim_f, input_dim_t)
    dim_map = {'c': 1, 'f': 2, 't': 3}
    assert len(se_dims) <= 3 and all([s in dim_map.keys() for s in se_dims]) or se_dims == 'none'
    if se_dims == 'none':
        se_dims = None
    else:
        se_dims = [dim_map[s] for s in se_dims]
            
    se_conf = dict(se_dims=se_dims, se_agg=se_agg, se_r=se_r)
    
    model = mobilenet_v3(pretrained_name=pretrained_name, num_classes=num_classes,
                     width_mult=width_mult, reduced_tail=reduced_tail, 
                    dilated=dilated, strides=strides,
                     head_type=head_type, multihead_attention_heads=multihead_attention_heads,
                     input_dims=input_dims, se_conf=se_conf
                     )
    
    
    # 2) Load the checkpoint you put into resources/
    ckpt_path = "./resources/mn10_as_mels_64_mAP_461.pt"
    sd = torch.load(ckpt_path, map_location="cpu")
    
    # 3) If head shape mismatches (e.g., your num_classes=50), load non-strict or drop head keys:
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError:
        model.load_state_dict(sd, strict=False)  # classifier will be ignored if shapes differ

    model.classifier[5] = nn.Linear(1280, 50)
    return model
