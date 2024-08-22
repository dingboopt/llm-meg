import torch

path='qwen32b8tp2pp'
path1='qwen32b8tp2pp_wrong'
    
for i in range(8):

    x=torch.load(f'{path}/release/mp_rank_0{i}/model_optim_rng.pt')
    x1=torch.load(f'{path1}/release/mp_rank_0{i}/model_optim_rng.pt')
    for k in x['model'].keys():
        print(f' {i}: compare {k}')
        if not torch.all(x['model'][k].cpu()==x1['model'][k].cpu()):
            print(f'release/mp_rank_0{i}/model_optim_rng.pt {k}')
            import pdb;pdb.set_trace()
            raise
    import pdb;pdb.set_trace()
