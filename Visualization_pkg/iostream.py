import os
from Training_pkg.utils import *


def get_figure_outfile_path(outdir,evaluation_type, figure_type,typeName,begindate,enddate, nchannel, **args):
    width = args.get('width', 11)
    height = args.get('height', 11)
    depth = args.get('depth', 3)

    if Apply_CNN_architecture:
        Model_structure_type = 'CNNModel'    
        outfile = outdir + '{}_{}_{}_{}_{}_{}x{}_{}-{}_{}Channel{}.png'.format(Model_structure_type, evaluation_type,figure_type,typeName, species, width, height, begindate,enddate,nchannel,description)
    elif Apply_3D_CNN_architecture:
        Model_structure_type = 'CNN3DModel'
        outfile = outdir + '{}_{}_{}_{}_{}_{}x{}x{}_{}-{}_{}Channel{}.png'.format(Model_structure_type, evaluation_type,figure_type,typeName, species, depth,width, height, begindate,enddate,nchannel,description)
    return outfile