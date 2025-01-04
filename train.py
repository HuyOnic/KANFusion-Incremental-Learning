import torch
import datetime
from utils.data_manager import DataManager
from incre_methods.KANFusion import KANFusion
import logging
import time
def train(args):
    start_time = time.time()
    time_str = datetime.datetime.now().strftime('%m%d-%H-%M-%S-%f')[:-3]
    filename = f"runs/{args['incre_method']}_{time_str}.log"
    f = open(filename, "x")
    data_manager = DataManager(
                                dataset_name=args["dataset"], 
                                shuffle=False, 
                                init_cls=args["init_cls"], 
                                increment=args["increment"], 
                                seed=args["seed"])
    
    logging.basicConfig(
    filename=filename,  # Log file name
    level=logging.INFO,               # Log level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    )
    logging.info(args)
    incre_method = get_incre_method(args)
    #Start incremental trainining
    for task in range(data_manager.num_task):
        incre_method.incremental_train(data_manager)
        incre_method.after_task()
        incre_method.eval_task(data_manager)
    end_time = time.time()
    cost_time = end_time - start_time
    logging.info(f'Total time: {cost_time}')
    incre_method.save_task_weights(time_str)   

def get_incre_method(args):
    incre_method = args["incre_method"].lower()
    if incre_method == "kanfusion":
        return KANFusion(args)     
          
    
    
    



