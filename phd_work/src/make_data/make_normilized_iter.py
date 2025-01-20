import numpy as np
import h5py as h5
import os
h5_in = '/home3/rfit/Telescope_Array/phd_work/data/bundled/pr_fe_q4_e1_0110_excl_sat_F_excl_geo_F.h5'
h5_out = '/home3/rfit/Telescope_Array/phd_work/data/normed/pr_fe_q4_e1_0110_excl_sat_F_excl_geo_F.h5'
os.makedirs('/home3/rfit/Telescope_Array/phd_work/data/normed/', exist_ok=True)
print(os.path.exists(h5_in))
iter_step = 250000

aux_vals = np.array( [-3., -3., 0., 0., 0., 0., 0.] ).astype(np.float32)

frac_train = 0.9
frac_test = 0.1
frac_val = 0.

keys_evs = ['recos'] #,'dt_bundle'
keys_hits = ['dt_params','wfs_flat']
keys_pull = ['ev_ids','mc_params', 'dt_mask']
#keys_pull += ['models_ids']

dsets = ['train','test','val']

proc = 0
with h5.File(h5_in,'r') as hi, h5.File(h5_out,'w') as ho:
    # get from where to start
    
    print(hi.keys())
    ev_starts = hi['ev_starts'][()]
    num_tot = ev_starts.shape[0]-1
    num_tr = int(frac_train*num_tot)
    num_te = int(frac_test*num_tot)
    num_va = min(int(frac_val*num_tot),num_tot-num_tr-num_te)
    nums = [num_tr,num_te,num_va]
    for (ds,num) in zip(dsets,nums):
        l_proc = 0
        l_proc_h = 0
        hf_ds = {}
        mean = {}
        std = {}
        # init dsets and norm params
        for key in keys_evs+keys_hits+keys_pull:
            if key in keys_evs+keys_pull:
                n = num
            else:
                n = ev_starts[proc+num] - ev_starts[proc]
            shape = np.concatenate( ([n,],hi[key].shape[1:]) )
            dtype = hi[key].dtype
            hf_ds[key] = ho.create_dataset(ds+'/'+key, shape=shape, dtype=dtype)
            if key not in keys_pull:
                mean[key] = hi['norm_param/'+key+'/mean'][()]
                std[key] = hi['norm_param/'+key+'/std'][()]
        while l_proc<num:
            print( l_proc, num)
            step = min(num-l_proc,iter_step)
            f_ev = proc+l_proc
            l_ev = proc+l_proc+step
            for key in keys_evs:
                data = (hi[key][f_ev:l_ev]-mean[key])/std[key]
                if key=='dt_bunlde':
                    mask = data[:,:,:,-1].astype(bool)
                    data[~mask] = aux_vals
                hf_ds[key][l_proc:l_proc+step] = data
            for key in keys_pull:
                hf_ds[key][l_proc:l_proc+step] = hi[key][f_ev:l_ev]
            f_hit = ev_starts[proc+l_proc]
            l_hit = ev_starts[proc+l_proc+step]
            n_hits = l_hit-f_hit
            for key in keys_hits:
                data = (hi[key][f_hit:l_hit]-mean[key])/std[key]
                hf_ds[key][l_proc_h:l_proc_h+n_hits] = data
            l_proc += step
            l_proc_h += n_hits
        l_ev_starts = ev_starts[proc:proc+num+1] - ev_starts[proc]
        ho.create_dataset(ds+'/ev_starts', data=l_ev_starts)            
        proc += num
    # write norm params
    for key in keys_evs:
        ho.create_dataset('norm_param/'+key+'/mean', data=hi['norm_param/'+key+'/mean'][()])
        ho.create_dataset('norm_param/'+key+'/std', data=hi['norm_param/'+key+'/std'][()])
    for key in keys_hits:
        ho.create_dataset('norm_param/'+key+'/mean', data=hi['norm_param/'+key+'/mean'][()])
        ho.create_dataset('norm_param/'+key+'/std', data=hi['norm_param/'+key+'/std'][()])
