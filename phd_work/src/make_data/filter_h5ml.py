import numpy as np
import h5py as h5
import os
from tqdm import tqdm
MC_dir_path = '/home/rfit/Telescope_Array/phd_work/data/'
h5_in = 'fe_q4_14yr_e1.h5'

# passed cuts: anisotropy, spectrum, composition, gamma
# anisotropy, spectrum, composition, gamma
filt_cuts = np.array([0,1,1,0])

exclude_saturated = False
exclude_geom_fit = False

keys_to_pull = ['ev_ids','reco_rubtsov','reco_rubtsov_params','bdt_params','dt_params','dt_wfs','dt_ids', 'dt_mask']
keys_to_pull += ['mc_params','reco_ivanov','reco_ivanov_params']

h5_out = h5_in[:-3]+'_'+''.join([str(f) for f in filt_cuts])+'_excl_sat_'+str(exclude_saturated)[0]+'_excl_geo_'+str(exclude_geom_fit)[0]+'.h5'
h5_in = MC_dir_path+'h5s/'+h5_in
h5_out = MC_dir_path+'filtered/'+h5_out

# ensure that there are no anomaleous events
E_cut = 2e3
time_up = 0.9*1e5
iter_step = 500000
print(os.path.exists(h5_in), h5_out, h5_in)
with h5.File(h5_in,'r') as hi, h5.File(h5_out,'w') as ho:
    # print(hi['passed_cuts'].value)
    num_evs = hi['pass_cuts'].shape[0]
    num_dets = hi['dt_params'].shape[0]
    # impose cuts
    pass_cuts = hi['pass_cuts'][()]
    passed_cuts = np.all( (pass_cuts-filt_cuts)>=0, axis=1 )
    print(passed_cuts.sum()/len(passed_cuts))
    passed_es = hi['reco_rubtsov'][:,-1]<E_cut
    mask_evs = np.logical_and( passed_cuts, passed_es )
    # make mask for detectors
    mask_hits = np.full( num_dets, True )
    # if exclude_saturated: # исключение по энергии
    #     not_pass = hi['dt_mask'][:,0]
    #     mask_hits = np.logical_and( mask_hits, ~not_pass )
    # if exclude_geom_fit: # исключение по геом. фиту
    #     not_pass = hi['dt_mask'][:,1]
    #     mask_hits = np.logical_and( mask_hits, ~not_pass )
    ev_starts = hi['ev_starts'][:]
    ev_lens = np.diff( ev_starts )
    mask_evs_to_hits = np.repeat( mask_evs, ev_lens )
    mask_hits = np.logical_and( mask_hits, mask_evs_to_hits )
    # read and write
    for key in tqdm(keys_to_pull):
        if key[:3]=='dt_':
            mask = mask_hits
            to_proc = num_dets
        else:
            mask = mask_evs
            to_proc = num_evs
        # init
        proc = 0
        wrtn = 0
        shape = hi[key].shape[1:]
        num_entrs = np.sum(mask)
        ds = ho.create_dataset( key, shape=np.concatenate( ([num_entrs],shape) ), dtype=hi[key].dtype )
        while proc<to_proc:
            step = min(iter_step,to_proc-proc)
            l_mask = mask[proc:proc+step]
            to_write = np.sum(l_mask)
            # correct times
            if key=='dt_params':
                data = hi[key][proc:proc+step][l_mask]
                # TODO (check)
                data[:,-1] = np.where( data[:,-1]<time_up, data[:,-1], data[:,-1]-1e6 )
                ds[wrtn:wrtn+to_write] = data
            else:
                ds[wrtn:wrtn+to_write] = hi[key][proc:proc+step][l_mask]
            proc += step
            wrtn += to_write
    # recalc ev length
    new_ev_lens = np.array([ np.sum( mask_hits[ev_starts[i]:ev_starts[i+1]] ) for i in range(num_evs) ])
    new_ev_lens = new_ev_lens[mask_evs]
    new_ev_starts = np.concatenate( ([0], np.cumsum(new_ev_lens)) )
    ho.create_dataset('ev_starts', data=new_ev_starts, dtype=np.int64)
