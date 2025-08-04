import toml
from Net_Architecture_config import cfg

####################################################################################
###                                  net structure                               ###
####################################################################################

# Network Architecture
network_architecture_cfg = cfg['network-architecture']
Apply_CNN_architecture = network_architecture_cfg['Apply_CNN_architecture']
Apply_3D_CNN_architecture = network_architecture_cfg['Apply_3D_CNN_architecture']
Apply_Transformer_architecture = network_architecture_cfg['Apply_Transformer_architecture']

CNN_architecture_cfg = network_architecture_cfg['CNN-architecture']

CNN_CovLayer_padding_mode = CNN_architecture_cfg['CovLayer_padding_mode']
CNN_Pooling_padding_mode = CNN_architecture_cfg['Pooling_padding_mode']

ResNet_Settings = CNN_architecture_cfg['ResNet']['Settings']
ResNet_Blocks = CNN_architecture_cfg['ResNet']['Blocks']
ResNet_blocks_num = CNN_architecture_cfg['ResNet']['blocks_num']
Pooling_layer_type = CNN_architecture_cfg['ResNet']['Pooling_layer_type']

TwoCombineModels_Settings = CNN_architecture_cfg['TwoCombineModels']['Settings']
TwoCombineModels_Variable = CNN_architecture_cfg['TwoCombineModels']['Variable']
TwoCombineModels_threshold = CNN_architecture_cfg['TwoCombineModels']['threshold']

ResNet_MLP_Settings  = CNN_architecture_cfg['ResNet_MLP']['Settings']
ResNet_MLP_Blocks    = CNN_architecture_cfg['ResNet_MLP']['Blocks']
ResNet_MLP_blocks_num  = CNN_architecture_cfg['ResNet_MLP']['blocks_num']

ResNet_Classification_Settings  = CNN_architecture_cfg['ResNet_Classification']['Settings']
ResNet_Classification_Blocks    = CNN_architecture_cfg['ResNet_Classification']['Blocks']
ResNet_Classification_blocks_num  = CNN_architecture_cfg['ResNet_Classification']['blocks_num']
ResNet_Classification_left_bin    = CNN_architecture_cfg['ResNet_Classification']['left_bin']
ResNet_Classification_right_bin   = CNN_architecture_cfg['ResNet_Classification']['right_bin']
ResNet_Classification_bins_number = CNN_architecture_cfg['ResNet_Classification']['bins_number']

ResNet_MultiHeadNet_Settings             = CNN_architecture_cfg['ResNet_MultiHeadNet']['Settings']
ResNet_MultiHeadNet_Blocks               = CNN_architecture_cfg['ResNet_MultiHeadNet']['Blocks']
ResNet_MultiHeadNet_blocks_num           = CNN_architecture_cfg['ResNet_MultiHeadNet']['blocks_num']
ResNet_MultiHeadNet_left_bin             = CNN_architecture_cfg['ResNet_MultiHeadNet']['left_bin']
ResNet_MultiHeadNet_right_bin            = CNN_architecture_cfg['ResNet_MultiHeadNet']['right_bin']
ResNet_MultiHeadNet_bins_number          = CNN_architecture_cfg['ResNet_MultiHeadNet']['bins_number']
ResNet_MultiHeadNet_regression_portion   = CNN_architecture_cfg['ResNet_MultiHeadNet']['regression_portion']
ResNet_MultiHeadNet_classifcation_portion = CNN_architecture_cfg['ResNet_MultiHeadNet']['classifcation_portion']


LateFusion_Settings      = CNN_architecture_cfg['LateFusion']['Settings']
LateFusion_Blocks        = CNN_architecture_cfg['LateFusion']['Blocks']
LateFusion_blocks_num    = CNN_architecture_cfg['LateFusion']['blocks_num']
LateFusion_initial_channels  = CNN_architecture_cfg['LateFusion']['initial_channels']
LateFusion_LateFusion_channels = CNN_architecture_cfg['LateFusion']['LateFusion_channels']

MultiHeadLateFusion_Settings  = CNN_architecture_cfg['MultiHeadLateFusion']['Settings']
MultiHeadLateFusion_Blocks    = CNN_architecture_cfg['MultiHeadLateFusion']['Blocks']
MultiHeadLateFusion_blocks_num      = CNN_architecture_cfg['MultiHeadLateFusion']['blocks_num']
MultiHeadLateFusion_initial_channels = CNN_architecture_cfg['MultiHeadLateFusion']['initial_channels']
MultiHeadLateFusion_LateFusion_channels  = CNN_architecture_cfg['MultiHeadLateFusion']['LateFusion_channels']
MultiHeadLateFusion_left_bin        = CNN_architecture_cfg['MultiHeadLateFusion']['left_bin']
MultiHeadLateFusion_right_bin       = CNN_architecture_cfg['MultiHeadLateFusion']['right_bin']
MultiHeadLateFusion_bins_number    = CNN_architecture_cfg['MultiHeadLateFusion']['bins_number']
MultiHeadLateFusion_regression_portion   = CNN_architecture_cfg['MultiHeadLateFusion']['regression_portion']
MultiHeadLateFusion_classifcation_portion  = CNN_architecture_cfg['MultiHeadLateFusion']['classifcation_portion']



## 3D CNN Architecture
CNN3D_architeture_cfg = network_architecture_cfg['CNN3D-architecture']
CovLayer_padding_mode_3D = CNN3D_architeture_cfg['CovLayer_padding_mode_3D']
Pooling_padding_mode_3D = CNN3D_architeture_cfg['Pooling_padding_mode_3D']

ResCNN3D_Settings = CNN3D_architeture_cfg['ResNet3D']['Settings']
ResCNN3D_Blocks = CNN3D_architeture_cfg['ResNet3D']['Blocks']
ResCNN3D_blocks_num = CNN3D_architeture_cfg['ResNet3D']['blocks_num']
ResCNN3D_output_channels = CNN3D_architeture_cfg['ResNet3D']['output_channels']
Pooling_layer_type_3D = CNN3D_architeture_cfg['ResNet3D']['Pooling_layer_type_3D']
ResNet3D_depth = CNN3D_architeture_cfg['ResNet3D']['Depth']


## Transformer Architecture
Transformer_cfg = network_architecture_cfg['Transformer-architecture']
Transformer_trg_dim = Transformer_cfg['trg_dim']
Transformer_d_model = Transformer_cfg['d_model']
Transformer_n_head = Transformer_cfg['n_head']
Transformer_ffn_hidden = Transformer_cfg['ffn_hidden']
Transformer_num_layers = Transformer_cfg['num_layers']
Transformer_max_len = Transformer_cfg['max_len']
Transformer_spin_up_len = Transformer_cfg['spin_up_len']
Transformer_drop_prob = Transformer_cfg['drop_prob']