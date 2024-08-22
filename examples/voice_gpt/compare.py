import torch
import sys
path=sys.argv[1]
path1=sys.argv[2]

for i in range(4):

    x=torch.load(f'{path}/mp_rank_0{i}/model_optim_rng.pt')
    x1=torch.load(f'{path1}/mp_rank_0{i}/model_optim_rng.pt')
    for k in x['model'].keys():
        if k.endswith('_extra_state'):
            continue
        print(f' {i}: compare {k}')
        if not torch.all(x['model'][k].cpu()==x1['model'][k].cpu()):
            print(f'release/mp_rank_0{i}/model_optim_rng.pt {k}')
            #import pdb;pdb.set_trace()
            raise
    for k in x1['model'].keys():
        if k.endswith('_extra_state') or k.endswith('encoder.weight') or k.endswith('encoder.bias'):
            continue
        print(f' {i}: compare {k}')
        if not torch.all(x['model'][k].cpu()==x1['model'][k].cpu()):
            print(f'release/mp_rank_0{i}/model_optim_rng.pt {k}')
            #import pdb;pdb.set_trace()
            raise
    #import pdb;pdb.set_trace()
