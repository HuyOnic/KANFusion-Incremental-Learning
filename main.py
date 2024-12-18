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
    parser = argparse.ArgumentParser(description="Config incremental learning")
    parser.add_argument('--dataset', type=str, default="cifar100", help="Select a dataset")
    parser.add_argument('--memory_size', type=int, default=10000, help="Size of examplar set")
    parser.add_argument('--init_cls', type=int, default=50, help="Number classes on 1st task")
    parser.add_argument('--increment', type=int, default=10, help="Number classes per task")
    parser.add_argument('--incre_method', type=str, default="kanfusion", help="Select Incremental Method")
    parser.add_argument('--model', type=str, default="resnet32", help="Classification model")
    parser.add_argument('--init_epochs', type=int, default=1, help="Number epochs of first task")
    parser.add_argument('--incre_epochs', type=int, default=5, help="Number epochs of N+1th task")
    return parser.parse_args()

if __name__=="__main__":
    main()