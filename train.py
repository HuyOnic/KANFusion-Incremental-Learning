import torch
import datetime
from utils.data_manager import DataManager
from incre_methods.Nexus_iKAN import NexusiKAN
import logging
import time
def train(args):
    torch.manual_seed(args["seed"])
    start_time = time.time()
    time_str = datetime.datetime.now().strftime('%m%d-%H-%M-%S-%f')[:-3]
    filename = f"runs/{args['incre_method']}_{time_str}.log"
    f = open(filename, "x")
    data_manager = DataManager(
                                dataset_name=args["dataset"], 
                                shuffle=False, 
                                init_cls=args["init_cls"], 
                                increment=args["incre_cls"], 
                                seed=args["seed"])
    
    logging.basicConfig(
    filename=filename,  
    level=logging.INFO,               
    format='%(asctime)s - %(levelname)s - %(message)s',  
    )
    logging.info(args)
    incre_method = get_incre_method(args)
    #Start incremental trainining
    for task in range(data_manager.num_task):
        incre_method.incremental_train(data_manager)
        incre_method.after_task()
        incre_method.eval_task(data_manager)
    cost_time = time.time() - start_time
    logging.info(f'Total time: {cost_time}')
    incre_method.save_task_weights(time_str)   

def get_incre_method(args):
    incre_method = args["incre_method"].lower()
    if incre_method == "nexusikan":
        return NexusiKAN(args) 
    else:
        return f"{incre_method} doesn't exist"    
          
    
    
    



