# Pipline 

1. filter_h5ml.py - накладываются фильтры.

2. merge_and_shuffle_iter.py - перемешивание событий.

3. make_bundle_iter_mask.py - преобразует данные к нужному формату.

4. Нормировка - get_normalization.py и make_normilized_iter.py.



## Data Fromat

dt_bunlde (num_dets,6,6,7):
parameters of detectors bundle
0. detector x relative to shower core, 1200 m units
1. detector y relative to shower core, 1200 m units
2. detector z, 1200 m units
3. detector signal MIP
4. time of the plane front arrival, mks
5. time of the waveform relative to the plane front, mks
6. mask for trigered detectors

recos (num_evs,15):
0. theta
1. phi
2. S_800
3. E_gamma
4. d_border, km
5. chi2/ndof for joint fit
6. Linsley front curvature
7. Area-over-peak (AOP) 1200
8. AOP slope
9. S_b parameter, b=2.5 (arXiv:1104.3399, arXiv:1305.7439)
10. S_b parameter, b=4.0
11. sum of all signals in all detectors
12. asymmetry  of  the  summed signal  at  the  upper  and  lower layers of detectors
13. total number of peaks in event
14. number of peaks for the detector with the largest signal

ev_ids (num_evs,3):
0. date
1. time
2. particle id (Z for nuclei, 0 for gamma, -1 for real data)

mc_params (num_evs,10):
0. mc_event_num
1. mc_parttype (CORSIKA, 1 - gamma, 14 - proton, 5626 - Fe)
2. mc_corecounter, closest to core detector number
3. mc_E (for primaries other than photon energy is rescaled by 1/1.27, i.e. to proton FD energy scale)
4. mc_theta
5. mc_phi
6. mc_height_1st_inter, km
7. mc_xcore
8. mc_ycore
9. mc_border_distance, km 

dt_params (num_dets,6)
0. detector x relative to shower core, 1200 m units
1. detector y relative to shower core, 1200 m units
2. detector z, 1200 m units
3. detector signal MIP
4. time of the plane front arrival, mks
5. time of the waveform relative to the plane front, mks

wfs_flat (num_dets,128,2): log(1+wf), float16
waveforms for all detectors
0: upper layer signal
1: lower layer signal

det_max_wf and det_max_params:
wfs and its params for the most acive detector

ev_starts (num_evs):
For event number i, ev_starts[i] is the first entry in dt_<anything> related to the event, ev_starts[i+1]-1 is the last.
For example, dt_params[ ev_starts[10]:ev_starts[11] ] corresponds to dt_params for 10th events