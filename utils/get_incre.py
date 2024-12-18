import os
import sys
sys.path.append(os.getcwd())
from incre_methods.KANFusion import KANFusion
def get_incre_method(args):
    incre_method = args["incre_method"].lower()
    if incre_method == "kanfusion":
        return KANFusion(args)