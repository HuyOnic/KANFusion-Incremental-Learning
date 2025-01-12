import argparse
import json
from train import train
import torch
 
def main(): 
    args = setup_parser()
    args.config = f"config/{args.incre_method}.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    param = load_json(args.config)
    args = vars(args)
    param.update(args)
    train(param)
    
def load_json(path):
    with open(path) as f:
        param = json.load(f)
    return param
 
def setup_parser():
    parser = argparse.ArgumentParser(description="Config Incremental Learning")
    parser.add_argument('--dataset', type=str, default="mnist", help="Selecting dataset")
    parser.add_argument('--memory_size', type=int, default=100, help="Total Size of examplar set")
    parser.add_argument('--init_cls', type=int, default=5, help="Number classes on 1st task")
    parser.add_argument('--incre_cls', type=int, default=5, help="Number classes per task")
    parser.add_argument('--incre_method', type=str, default="nexusikan", help="Select A Incremental Method")
    parser.add_argument('--model', type=str, default="resnet32", help="Select A Classification Model")
    parser.add_argument('--init_epochs', type=int, default=1, help="Number epochs of first task")
    parser.add_argument('--incre_epochs', type=int, default=1, help="Number epochs of N+1th task")
    return parser.parse_args()

if __name__=="__main__":
    main()