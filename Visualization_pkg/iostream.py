import os
from Training_pkg.utils import *


def get_figure_outfile_path(outdir, evaluation_type, figure_type,typeName,begindate,enddate, nchannel, **args):
    width = args.get('width', 11)
    height = args.get('height', 11)
    depth = args.get('depth', 3)
    d_model = args.get('d_model', 64)
    n_head = args.get('n_head', 8)
    ffn_hidden = args.get('ffn_hidden', 256)
    num_layers = args.get('num_layers', 6)
    max_len = args.get('max_len', 1000)
    
    if Apply_CNN_architecture:
        Model_structure_type = 'CNNModel'    
        outfile = outdir + '{}_{}_{}_{}_{}_{}x{}_{}-{}_{}Channel{}.png'.format(Model_structure_type, evaluation_type,figure_type,typeName, species, width, height, begindate,enddate,nchannel,description)
    elif Apply_3D_CNN_architecture:
        if MoE_Settings:
            Model_structure_type = '3DCNN_MoE_{}Experts_Model'.format(MoE_num_experts)
        elif MoCE_Settings:
            Model_structure_type = '3DCNN_MoCE_{}Experts_Model'.format(MoCE_num_experts)
        else:
            Model_structure_type = 'CNN3DModel'
        outfile = outdir + '{}_{}_{}_{}_{}_{}x{}x{}_{}-{}_{}Channel{}.png'.format(Model_structure_type, evaluation_type,figure_type,typeName, species, depth,width, height, begindate,enddate,nchannel,description)
    elif Apply_Transformer_architecture:
        Model_structure_type = 'TransformerModel'
        outfile = outdir + '{}_{}_{}_{}_{}_{}dmodel_{}head_{}ffn_{}layers_{}maxlen_{}-{}_{}Channel{}.png'.format(Model_structure_type, evaluation_type,figure_type,typeName, species, d_model,n_head,ffn_hidden,num_layers,max_len, begindate,enddate,nchannel,description)
    return outfile