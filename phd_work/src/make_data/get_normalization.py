import numpy as np
import h5py as h5

use_wf_16 = True
h5_in = 'pr_q4_epos_sibyll_e1_0110_excl_sat_F_excl_geo_F.h5'
h5f = '/home3/rfit/Telescope_Array/phd_work/data/bundled/' + h5_in
num_evs = 200000
keys_evs = ['recos','det_max_wf','det_max_params']
keys_hits = ['dt_params','wfs_flat']

with h5.File(h5f,'a') as hf:
    # event-wise
    print(hf.keys())
    for key in keys_evs:
        data = hf[key][:num_evs]
        mean = np.mean( data, dtype=np.float64, axis=tuple(range(len(data.shape)-1)) )
        std = np.std( data, dtype=np.float64, axis=tuple(range(len(data.shape)-1)) )
        hf.create_dataset('norm_param/'+key+'/mean', data=mean, dtype=np.float32 )
        hf.create_dataset('norm_param/'+key+'/std', data=std, dtype=np.float32 )
    # detector-wise
    last_ev = hf['ev_starts'][num_evs+1]
    for key in keys_hits:
        data = hf[key][:last_ev]
        mean = np.mean(data, dtype=np.float64, axis=tuple(range(len(data.shape)-1)))
        std = np.std(data, dtype=np.float64, axis=tuple(range(len(data.shape)-1)))
        if key=='wfs_flat' and use_wf_16:
            dtype = np.float16
        else:
            dtype = np.float32
        hf.create_dataset('norm_param/'+key+'/mean', data=mean, dtype=dtype )
        hf.create_dataset('norm_param/'+key+'/std', data=std, dtype=dtype )
    # dt bundle
    if False:
        data = hf['dt_bundle'][:num_evs]
        mask = data[:,:,:,-1].astype(bool)
        data = data[:,:,:,:-1][mask]
        mean = np.mean(data, axis=tuple(range(len(data.shape)-1)), dtype=np.float64)
        std = np.std(data, axis=tuple(range(len(data.shape)-1)), dtype=np.float64)
        mean = np.concatenate((mean,[0]))
        std = np.concatenate((std,[1]))
        hf.create_dataset('norm_param/dt_bundle/mean', data=mean, dtype=np.float32 )
        hf.create_dataset('norm_param/dt_bundle/std', data=std, dtype=np.float32 )
    
    print(hf.keys())
