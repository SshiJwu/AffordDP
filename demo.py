from afforddp.env.CabinetManip import CabinetManipEnv
from afforddp.retrieval.retrieval_buf import RetrievalBuf
from afforddp.gym_util.utils import read_yaml_config
from afforddp.utils.seed import set_seed
from afforddp.retrieval.affordance_transfer import affordance_transfer
import argparse
import os
import shutil


def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, default='PullDrawer.yaml', help='Environment Cobfig Name')
    parser.add_argument('--obj_id', type=int, default=45290, help='Gapartnet asset id')
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--part_id', type=int, default=-1, help='select part to manipulation')
    parser.add_argument('--data_dir', type=str, default='record', help='expert demonstration data directory')
    parser.add_argument('--memory_dir', type=str, default='data/memory', help='memory file directory')
    parser.add_argument('--save_dir', type=str, default='vis', help='the path where the visualization results are stored')
    args = parser.parse_args()

    return args

def run_demo(args):
    
    set_seed(args.seed)
    
    task_name = args.config_name.split(".")[0]
    if not task_name in ['PullDrawer', 'OpenDoor']:
        raise ValueError(f'Invalid task_type: {task_name}')
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    else:
        shutil.rmtree(args.save_dir)
        os.makedirs(args.save_dir)
        
    config_path = os.path.join(os.getcwd(),"afforddp/config/env",args.config_name)
    cfgs = read_yaml_config(config_path)
    obj_id = args.obj_id
    
    cfgs['asset']['arti']['arti_gapartnet_ids'] = [obj_id]
    
    gym = CabinetManipEnv(cfgs)
    memory_buffer = RetrievalBuf(data_dir=args.data_dir, save_dir=args.memory_dir, task_name=task_name)
    
    gym.reset(to_reset="all")
    gym.get_gapartnet_anno()        
    if not cfgs["HEADLESS"] and True:
        gym.gym.clear_lines(gym.viewer)
    gym.cal_handle(bbox_id=args.part_id)
    
    affordance_transfer(prompt='cabinet', 
                        gym=gym, 
                        save_dir=args.save_dir, 
                        memory_buffer=memory_buffer, 
                        task_name=task_name)
    
    del gym

    
if "__main__" == __name__:
    
    args = parse_args()
    run_demo(args)
