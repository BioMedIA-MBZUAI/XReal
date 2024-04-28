import sys
import os
os.chdir('../')
sys.path.append(f'{os.getcwd()}') # for xreal | yes i know this is not good practice
sys.path.append(f'{os.getcwd()}/xreal') # for ldm
sys.path.append(f'{os.getcwd()}/src') # for clip
sys.path.append(f'{os.getcwd()}/src/taming-transformers') # for taming
import torch
from omegaconf import OmegaConf
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='../saved_models/backbone.ckpt')
parser.add_argument('--output_path', type=str, default='../saved_models/cnet_base.pt')
parser.add_argument('--model_config_path', type=str, default="../configs/xreal-diff-t2i_cnet.yml")
args = parser.parse_args()

input_path = args.input_path
output_path = args.output_path
model_config_path = args.model_config_path

print(f'Input path: {input_path}')
print(f'current dir: {os.getcwd()}')
assert os.path.exists(input_path), 'Input model does not exist.'
assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'

print(os.getcwd())
print(sys.path)

from xreal.ldm import instantiate_from_config


def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model

def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]


# model = create_model(config_path='./models/cldm_v15.yaml')
model = create_model(config_path=model_config_path)

pretrained_weights = torch.load(input_path)
if 'state_dict' in pretrained_weights:
    pretrained_weights = pretrained_weights['state_dict']

scratch_dict = model.state_dict()

target_dict = {}
for k in scratch_dict.keys():
    is_control, name = get_node_name(k, 'control_')
    if is_control:
        copy_k = 'model.diffusion_' + name
    else:
        copy_k = k
    if copy_k in pretrained_weights:
        target_dict[k] = pretrained_weights[copy_k].clone()
    else:
        target_dict[k] = scratch_dict[k].clone()
        print(f'These weights are newly added: {k}')

model.load_state_dict(target_dict, strict=True)
torch.save(model.state_dict(), output_path)
print('Done.')
