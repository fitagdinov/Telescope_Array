import numpy as np
import h5py as h5
import random as rd
from tqdm import tqdm
import os
import logging
from glob import glob
logger = logging.getLogger(__name__)
logging.basicConfig(filename='myapp.log', level=logging.INFO)
MC_dir_path = '/home3/rfit/Telescope_Array/phd_work/data/filtered/'

ps = ['pr']
ms = ['q4', 'epos', 'sibyll']
es = ['e1']
# h5s = [ p+'_'+m+'_14yr_'+e+'_0110_excl_sat_F_excl_geo_F.h5' for p in ps for m in ms for e in es ]

models_ids = np.array([ m for p in ps for m in ms for e in es ]).astype('<S6')
# h5s = [ MC_dir_path+p for p in h5s ]
h5s = glob(MC_dir_path + '*1895*')
logging.info(h5s)
h5dir = '/home3/rfit/Telescope_Array/phd_work/data/merged/'
os.makedirs(h5dir, exist_ok=True)
h5_out = '_'.join(ps+ms+es) + '_0110_excl_sat_F_excl_geo_F.h5'
h5_out = os.path.join(h5dir, h5_out)
write_step = 500000
keys_to_pull = ['ev_ids','reco_rubtsov','reco_rubtsov_params','mc_params','dt_params','dt_wfs','dt_ids', 'dt_mask'] #bdt_params
keys_to_pull += ['reco_ivanov','reco_ivanov_params']

# get nums evs
nums_evs = {}
total_num = 0
for h5f in h5s:
    with h5.File(h5f,'r') as hf:
        nums_evs[h5f] = hf['reco_ivanov'].shape[0]
        total_num += nums_evs[h5f]

# sample
shfl = np.array( rd.sample( range(total_num), k=total_num ) )

# read and write
with h5.File(h5_out,'w') as ho:
    # ev starts
    ev_lens = []
    nums = []
    for h5f in h5s:
        with h5.File(h5f,'r') as hf:
            ev_lens.append( np.diff( hf['ev_starts'][:] ) )
            nums.append( hf['reco_ivanov'].shape[0] ) 
    ev_lens = np.concatenate( ev_lens )
    ev_starts = np.concatenate( ([0],np.cumsum(ev_lens)) )
    ev_lens_new = ev_lens[shfl]
    ev_starts_shfl = np.concatenate( ([0],np.cumsum(ev_lens_new)) )
    # init
    ho.create_dataset('idxs_shuffl' , data=shfl )
    h5_ds = {}
    for key in keys_to_pull:
        with h5.File(h5s[0],'r') as hf:
            shape = hf[key].shape[1:]
            dtype = hf[key].dtype
        if key not in ['dt_params','dt_wfs','dt_ids', 'dt_mask']:
            h5_ds[key] = ho.create_dataset(key, shape=np.concatenate(([total_num,],shape)), dtype=dtype)
        else:
            h5_ds[key] = ho.create_dataset(key, shape=np.concatenate(([ev_starts[-1],],shape)), dtype=dtype)
    ho.create_dataset('ev_starts' , data=ev_starts_shfl )
    ho.create_dataset('models_ids' , data=np.repeat( models_ids, nums )[shfl] )
    # others
    for key in tqdm(keys_to_pull[:]):
        proc = 0
        proc_hits = 0
        while proc<total_num:
            print(proc,total_num, key)
            # init
            step = min(write_step,total_num-proc)
            shfl_cycle = shfl[proc:proc+step]
            tot = 0
            step_hits = np.sum( ev_lens_new[proc:proc+step] )
            # sort: idea is to read all, ordered -> shuffle properly
            idxs_sort = np.argsort( shfl_cycle )
            rev_idxs = np.argsort( idxs_sort )
            shfl_sorted = shfl_cycle[idxs_sort]
            # init
            vals = []
            for h5f in h5s:
                with h5.File(h5f,'r') as hf:
                    mask_h5 = np.logical_and( shfl_sorted>=tot, shfl_sorted<tot+nums_evs[h5f] )
                    read_h5 = np.full( (nums_evs[h5f],), False )
                    read_h5[shfl_sorted[mask_h5]-tot] = True
                    if key not in ['dt_params','dt_wfs','dt_ids', 'dt_mask']:
                        vals.append( hf[key][()][read_h5] )               
                    else:
                        h5_ev_lens = ev_lens[tot:tot+nums_evs[h5f]]
                        read_h5_hits = np.repeat( read_h5, h5_ev_lens )
                        read_ev_lens = h5_ev_lens[read_h5]
                        read_ev_starts = np.concatenate( ([0],np.cumsum(read_ev_lens)) )
                        temp = hf[key][()][read_h5_hits]
                        vals += [ temp[read_ev_starts[k]:read_ev_starts[k+1]] for k in range(len(read_ev_lens)) ]
                tot += nums_evs[h5f]
            if key not in ['dt_params','dt_wfs','dt_ids', 'dt_mask']:
                vals = np.concatenate(vals, axis=0)
                h5_ds[key][proc:proc+step] = vals[rev_idxs]
            else:
                vals = [ vals[i] for i in rev_idxs ]
                vals = np.concatenate(vals, axis=0)
                h5_ds[key][proc_hits:proc_hits+step_hits] = vals
            proc +=step
            proc_hits += step_hits
