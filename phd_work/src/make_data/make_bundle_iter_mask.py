# CHOOSE between reco rubtsov and ivanov
import numpy as np
import h5py as h5
import os
from tqdm import tqdm
take_log_wf = True
reco_key = 'reco_ivanov'
iter_step = 100000
h5_in = 'pr_photon_0001_excl_sat_F_excl_geo_F.h5'
h5_out = h5_in#[:-3]+'_bundled.h5'
MC_dir_path_ROBERT = '/home3/rfit/Telescope_Array/phd_work/data/'
h5_in = os.path.join(MC_dir_path_ROBERT, 'merged/', h5_in)
os.makedirs('/home3/rfit/Telescope_Array/phd_work/data/bundled', exist_ok=True)
h5_out = os.path.join(MC_dir_path_ROBERT, 'bundled', h5_out)

take_bdt_params = ()#0,1,2,5,8,10,11,12,13)
take_reco = (0,1,4,5)
aux_vals = np.array( [-5., -5., 0., 0., 0., 0.] ).astype(np.float32)

proc = 0
if os.path.exists(h5_out):
    print('Output file already exists, remove it to continue.')
    os.remove(h5_out)
print(os.path.exists(h5_in), h5_out)
with h5.File(h5_in,'r') as hi, h5.File(h5_out,'w') as ho:
    ev_starts = hi['ev_starts'][:]
    # init datasets
    ho_ds = {}
    num_evs = hi['mc_params'].shape[0]
    num_hits = hi['dt_params'].shape[0]
    for key in ['mc_params','ev_ids']:
        ho_ds[key] = ho.create_dataset(key, shape=np.concatenate( ([num_evs,],hi[key].shape[1:]) ), dtype=hi[key].dtype )
    ho_ds['wfs_flat'] = ho.create_dataset('wfs_flat', shape=np.concatenate( ([num_hits,],hi['dt_wfs'].shape[1:]) ), dtype=hi['dt_wfs'].dtype )
    ho_ds['dt_params'] = ho.create_dataset('dt_params', shape=np.concatenate( ([num_hits,],hi['dt_params'].shape[1:]) ), dtype=hi['dt_params'].dtype )
    ho_ds['recos'] = ho.create_dataset('recos', shape=(num_evs,len(take_reco)+2+len(take_bdt_params)), dtype=hi[reco_key].dtype )
    # ho_ds['dt_bundle'] = ho.create_dataset('dt_bundle', shape=(num_evs,6,6,7), dtype=np.float32 )
    ho_ds['det_max_wf'] = ho.create_dataset('det_max_wf', shape=(num_evs,128,2), dtype=np.float32 )
    ho_ds['det_max_params'] = ho.create_dataset('det_max_params', shape=(num_evs,6), dtype=np.float32 )
    ho_ds['dt_mask'] = ho.create_dataset('dt_mask', shape=(num_hits,2), dtype=np.float32 ) # mask for saturated and geo-excluded dets
    # ho_ds['dt_bunlde_mask'] = ho.create_dataset('dt_bunlde_mask', shape=(num_evs,6,6,2), dtype=np.float32 )
    while proc<num_evs:
        print(num_evs, proc)
        step = min(iter_step,num_evs-proc)
        f_ev = proc
        l_ev = proc + step
        f_hit = ev_starts[proc]
        l_hit = ev_starts[proc+step]
        l_ev_starts = ev_starts[f_ev:l_ev+1] - ev_starts[f_ev]
        wfs = hi['dt_wfs'][f_hit:l_hit]
        dts = hi['dt_params'][f_hit:l_hit]
        mask_dets = hi['dt_mask'][f_hit:l_hit]
        for key in ['ev_ids','mc_params']:
            ho_ds[key][f_ev:l_ev] = hi[key][f_ev:l_ev]
        # flat wfs
        # логарифмирование и замена отрицательных
        if take_log_wf:
            wfs += 1
            wfs = np.where( wfs>0, wfs, 1e-5 )
            wfs = np.log( wfs )
        ho_ds['wfs_flat'][f_hit:l_hit] = wfs
        ho_ds['dt_params'][f_hit:l_hit] = dts
        # сшивка всех параметров иванова + всех параметров рубцова + каких-то bdt_params
        ho_ds['recos'][f_ev:l_ev] = np.concatenate( (hi[reco_key][f_ev:l_ev][:,take_reco],hi['reco_rubtsov_params'][f_ev:l_ev]), axis=1 ) # hi['bdt_params'][f_ev:l_ev][:,take_bdt_params]
        # most active
        idxs_mostQ = np.array([l_ev_starts[i]+np.argmax(dts[l_ev_starts[i]:l_ev_starts[i+1],3]) for i in range(step)])
        mostQ_params = np.array([ dts[idx] for idx in idxs_mostQ ])
        mostQ_wfs = np.stack([ wfs[idx] for idx in idxs_mostQ ])
        ho_ds['det_max_wf'][f_ev:l_ev] = mostQ_wfs
        ho_ds['det_max_params'][f_ev:l_ev] = mostQ_params
        # geom bundle
        # make grid of DT ids
        dt_ids = hi['dt_ids'][f_hit:l_hit,0]
        x_ids = dt_ids // 100
        y_ids = dt_ids % 100
        mostQ_xids = np.array([ x_ids[idx] for idx in idxs_mostQ ])
        mostQ_yids = np.array([ y_ids[idx] for idx in idxs_mostQ ])
        Q_x_grid = np.array([ 3 if x>0 else 2 for x in mostQ_params[:,0] ])
        Q_y_grid = np.array([ 3 if y>0 else 2 for y in mostQ_params[:,1] ])
        tl_x = np.array([ mostQ_xids[i]-Q_x_grid[i] for i in range(step) ])
        tl_y = np.array([ mostQ_yids[i]-Q_y_grid[i] for i in range(step) ])
        xs_ids = np.array([ [ x+i for i in range(6) ] for x in tl_x ])[:,:,np.newaxis] # ev, 6, 1
        xs_ids = np.repeat( xs_ids, 6, axis=2 ) # ev, 6, 6
        ys_ids = np.array([ [ y+i for i in range(6) ] for y in tl_y ])[:,np.newaxis,:] # ev, 1, 6
        ys_ids = np.repeat( ys_ids, 6, axis=1 ) # ev, 6, 6
        grids = xs_ids*100 + ys_ids # ev, 6, 6
        # get mask for writing
        grids = grids.reshape(step,36)
        coinc = [ [ np.nonzero(dt_ids[l_ev_starts[ev]:l_ev_starts[ev+1]]==g_id) for g_id in grids[ev] ] for ev in range(step) ] # ev, num_dt
        mask_flat = [ [ l_ev_starts[ev]+coinc[ev][j][0][0] for j in range(36) if len(coinc[ev][j][0])>0 ] for ev in range(step) ]
        mask_flat = np.concatenate( mask_flat )
        mask_grid = np.array([ [ True if len(coinc[ev][j][0])>0 else False for j in range(36) ] for ev in range(step) ])
        mask_grid = mask_grid.reshape(step,6,6)
        # write
        dt_grid = np.full( (step,6,6,6), aux_vals )
        dt_grid[mask_grid] = dts[mask_flat]
        dt_grid = np.concatenate( (dt_grid,np.expand_dims(mask_grid, axis=-1)), axis=-1 )
        # ho_ds['dt_bundle'][f_ev:l_ev]= dt_grid
        dt_mask_dets = np.full( (step,6,6,2), False )
        dt_mask_dets[mask_grid] = mask_dets[mask_flat]
        # ho_ds['dt_bunlde_mask'][f_ev:l_ev] = dt_mask_dets
        ho_ds['dt_mask'][f_hit:l_hit] = mask_dets
        proc += step
    ho.create_dataset('ev_starts', data=ev_starts)
    ho.create_dataset('models_ids', data=hi['models_ids'][()])
