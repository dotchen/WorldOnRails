import tqdm
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader, WeightedRandomSampler
from .ego_dataset import EgoDataset
from .main_dataset import LabeledMainDataset

def data_loader(data_type, config):

    if data_type == 'ego':
        dataset = EgoDataset(config.data_dir, T=config.ego_traj_len)
    elif data_type == 'main':
        dataset = LabeledMainDataset(config.data_dir, config.config_path)
    else:
        raise NotImplementedError(f'Unknown data type {data_type}')

    if config.balanced_cmd:
        # cmds = defaultdict(list)
        # for idx in tqdm.tqdm(range(0,len(dataset),3), 'Computing class weights'):
        # # for idx in tqdm.tqdm(range(0,300,3), 'Computing class weights'):
        #     cmd = dataset[idx][-1]
        #     cmds[cmd].append(idx)

        # weights = np.zeros(len(dataset))
        # for cmd, idxes in cmds.items():
        #     print (cmd, len(idxes))
        #     np_idxes = np.array(idxes)
        #     weights[np_idxes] = len(dataset)/len(idxes)
        #     weights[np_idxes+1] = len(dataset)/len(idxes)
        #     weights[np_idxes+2] = len(dataset)/len(idxes)
        
        # import pickle
        # pickle.dump(weights, open('weights.pkl', 'wb'))
        import pickle
        weights = pickle.load(open('weights.pkl', 'rb'))
        
        return DataLoader(dataset, batch_size=config.batch_size, num_workers=config.num_workers, drop_last=True, sampler=WeightedRandomSampler(weights, len(weights)))
    else:
        return DataLoader(dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True, drop_last=True)

__all__ = ['data_loader']