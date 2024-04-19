import tensorflow as tf
import math
import copy
import tqdm
import numpy as np
pi = tf.constant(math.pi,dtype=tf.float32)
UNIT=1
dist=tf.constant(1.2,dtype=tf.float32)# min dist between 2 detectors in km
c=tf.constant(299792.458,dtype=tf.float32)# # km\s
NSEC= tf.constant(1e9/c,dtype=tf.float32)# in rubsov's code is a time for 1.2 km 1.2/c*1e9
R_L=tf.constant(30e-3,dtype=tf.float32)#
LINSLEY_r0=tf.constant(0.030,dtype=tf.float32)#
DET_AREA=tf.constant(3,dtype=tf.float32)#
s_min = tf.constant([[0.3]],dtype=tf.float32)
s_max = tf.constant([[1.8]],dtype=tf.float32)
t_err_res=tf.constant(c/1e6,dtype=tf.float32)#
t0_err=tf.constant(0.03,dtype=tf.float32)#
dist=tf.constant(1.2,dtype=tf.float32)
tf_type = tf.float32
R_error = 0.15
def detectors_init(data):
    batch = data.shape[0]
    x = tf.cast(tf.repeat(tf.expand_dims(tf.range(0,6),0),6,axis=0),tf_type)
    y = tf.cast(tf.repeat(tf.expand_dims(tf.range(0,6),1),6,axis=1),tf_type)
    x = tf.expand_dims(x,-1)
    y = tf.expand_dims(y,-1)
    # x.shape = (6,6,1)
    detectors = tf.concat([x,y],axis=-1) * dist
    detectors = tf.repeat(tf.expand_dims(detectors,0),batch,0)
    return detectors
def core_(detectors,signal):# shape (batch,6,6,2)
    sum_signal=tf.reduce_sum(signal,axis=(1,2)) # shape (batch ,1)
    core=tf.reduce_sum(detectors * signal, axis=(1,2))/sum_signal
    return core
def expand_dims(vec):
    return tf.expand_dims(tf.expand_dims(vec,-1),-1)
def create_matrix(x,y,t,mask):
    a11=expand_dims(tf.reduce_sum(x*x,axis=1))
    a12=expand_dims(tf.reduce_sum(x*y,axis=1))
    a13=expand_dims(tf.reduce_sum(x,axis=1))
    a22=expand_dims(tf.reduce_sum(y*y,axis=1))
    a23=expand_dims(tf.reduce_sum(y,axis=1))
    a33=expand_dims(tf.reduce_sum(mask,axis=1))
    a1=tf.concat([a11,a12,a13],axis=2)
    a2=tf.concat([a12,a22,a23],axis=2)
    a3=tf.concat([a13,a23,a33],axis=2)
    A=tf.concat([a1,a2,a3],axis=1)

    b1=expand_dims(tf.reduce_sum(x*t,axis=1))
    b2=expand_dims(tf.reduce_sum(y*t,axis=1))
    b3=expand_dims(tf.reduce_sum(t,axis=1))
    b=tf.concat([b1,b2,b3],axis=1)
    return A,b
def place_sol(detectors,real_time,mask):
    detectors = detectors * mask
    x = tf.reshape(detectors[:,:,:,0],(-1,36))
    y = tf.reshape(detectors[:,:,:,1],(-1,36))
    t = tf.reshape(real_time,(-1,36))
    mask = tf.reshape(mask,(-1,36))
    A,b=create_matrix(x,y,t,mask)
    return tf.linalg.solve(A,b)
def place_params(detectors,real_time,mask):
    sol = place_sol(detectors,real_time,mask)
    #t_0=b+(r_cor;n)
    b=sol[:,2:,0]
    n=sol[:,:2,0]
#     core = self.core_()
#     mul=n[:,0]*core[:,0]+n[:,1]*core[:,1]
    t0=b
    a_x=n[:,0]
    a_y=n[:,1]
#         print(a_x.dtype,(1e6/c).dtype,c.dtype)
    a_z=tf.math.pow(1e6/c,2) - (tf.math.pow(a_x,2) + tf.math.pow(a_y,2))
    a_z=tf.where(a_z>0,tf.math.sqrt(a_z),0)
    cos_theta=a_z*(c/1e6)
    theta=tf.math.acos(cos_theta)
#     tg_phi=a_x/(-a_y)
#     atan=tf.math.atan(tg_phi)
#     phi=tf.where(a_y>0,atan+pi,atan-pi)
#     phi=tf.where(a_x<0,atan,phi)
    phi = tf.math.atan2( a_y, a_x ) + pi
    return t0,tf.expand_dims(theta,-1),tf.expand_dims(phi,-1)
def detectors_core(detectors,core):
    detectors_c=detectors[:,:,:,:2]-tf.reshape(core,(-1,1,1,2))
    return tf.concat([detectors_c,detectors[:,:,:,2:3]],axis=-1)
def place_reconstruction(detectors,mask,t0,theta,phi,use_z=False):
    t0=expand_dims(t0) # shape (batch,1,1,1)
    theta=expand_dims(theta)
    phi =expand_dims(phi)
    if use_z:
        n=-tf.concat([tf.math.cos(phi)*tf.math.sin(theta),tf.math.sin(phi)*tf.math.sin(theta),tf.math.cos(theta)],axis=-1)
    else:
        n=-tf.concat([tf.math.cos(phi)*tf.math.sin(theta),tf.math.sin(phi)*tf.math.sin(theta)],axis=-1)
    
    n=tf.cast(n,tf.float32)
    # print('n',n[3])
    t_place =  tf.expand_dims(tf.reduce_sum(detectors*n,axis=-1),-1)*(1e6/c)
    t_place = t_place*mask
    return t_place
def eta_fun(theta):
    x=theta*180/3.14

    e1 = 3.97 - 1.79*(tf.math.abs(1.0/tf.math.cos(theta)) - 1.0)
    e2 = ((((((-1.71299934e-10*x + 4.23849411e-08)*x -3.76192000e-06)*x
               + 1.35747298e-04)*x -2.18241567e-03)*x + 1.18960682e-02)*x
             + 3.70692527e+00)
    res = tf.where(x<62.7,e1,e2)
#     res =tf.where(res>0,res,0)
    return res
def linsley_t(r,S):
    return 0.67*tf.math.pow((1 + r/LINSLEY_r0), 1.5)*tf.math.pow(S, -0.5)*1e-3
def s_profile_tasimple(r_ta,theta,fl=False):
    # убрать 1,2
    UNIT = 1000.0
    r = r_ta * UNIT 
    eta=eta_fun(theta)# batch,1,1
    eta=tf.repeat(eta,6,axis=1)
    eta=tf.repeat(eta,6,axis=2)
#     print('eta',eta.shape)
    # eta shape is batch,6,6
    Rm = tf.constant(90,dtype=tf.float32)# убрал 1.2 из-за Unit
    R1 = tf.constant(1000,dtype=tf.float32)
#     print('shape sprofile',r.shape,Rm.shape,R1.shape,eta.shape,theta.shape)
    return (tf.math.pow((r/Rm),-1.2)*tf.math.pow((1+r/Rm), -(eta-1.2))*tf.math.pow(1+(tf.math.pow(r,2)/R1/R1),-0.6))

def s_profile(r_ta, theta):
    f800=s_profile_tasimple(expand_dims(tf.constant(0.8)), theta,fl=False)
    return s_profile_tasimple(r_ta, theta)/f800
def courve_reconstruction(detectors,t0,theta,phi,courve):
    # u can read from t_place if in place_reconstruction use core shift
#     detectors = detectors_core()
    t0=expand_dims(t0) # shape (batch,1,1,1)
    theta=tf.cast(expand_dims(theta),tf.float32)
    phi = tf.cast(expand_dims(phi),tf.float32)
    n=-tf.concat([tf.math.cos(phi)*tf.math.sin(theta),tf.math.sin(phi)*tf.math.sin(theta),tf.math.cos(theta)],axis=-1)
    n=tf.cast(n,tf.float32)
    t_place = detectors[:,:,:,0:1]*n[:,:,:,0:1] + detectors[:,:,:,1:2]*n[:,:,:,1:2] + detectors[:,:,:,2:3]*n[:,:,:,2:3]
    # print('r_plane',t_place[3,:,:,0])
    dist_core = tf.expand_dims(tf.reduce_sum(tf.math.pow(detectors,2),axis=-1),axis=-1) - tf.math.pow(t_place,2)
    dist_core = tf.where(dist_core>0,tf.math.sqrt(dist_core),0)
    dist_core = tf.where(dist_core<R_error,R_error,dist_core)
    # print('dist_core',dist_core[3,:,:,0])
    LDF=s_profile(dist_core,theta)

    # print('s_profile',LDF[3,:,:,0])
    td=expand_dims(courve)*linsley_t(dist_core,LDF)
    # print('td',td[3,:,:,0])
    return td,LDF,dist_core
def pfs__pps(detectors,theta,phi,signal,mask):
#     t0,theta,phi = self.place_params()
    # u can read from t_place if in place_reconstruction use core shift
#     detectors = detectors_core(detectors,core)
    theta=expand_dims(theta)
    phi = expand_dims(phi)
    n=-tf.concat([tf.math.cos(phi)*tf.math.sin(theta),tf.math.sin(phi)*tf.math.sin(theta),tf.math.cos(theta)],axis=-1)
    t_place = detectors[:,:,:,0:1]*n[:,:,:,0:1] + detectors[:,:,:,1:2]*n[:,:,:,1:2] + detectors[:,:,:,2:3]*n[:,:,:,2:3] # not has t0
    # end t_place's part
    dist_core = tf.expand_dims(tf.reduce_sum(tf.math.pow(detectors,2),axis=-1),axis=-1) - tf.math.pow(t_place,2)
    dist_core = tf.where(dist_core>0,tf.math.sqrt(dist_core),0)
    cond_dist = tf.cast(tf.where(tf.logical_and(dist_core>s_min,dist_core<s_max),mask,0.),tf_type)
    pfs = tf.reduce_sum(signal*cond_dist,axis=(1,2))
    pps = tf.reduce_sum(s_profile(dist_core,theta)*cond_dist,axis=(1,2))
    return (pfs,pps)
def a_ivanov_fun(theta):
    DEG=pi/180
    threshold1=25*DEG
    threshold2=35*DEG
    # переписать для обнавления масива
    res1=tf.where(theta<threshold1,3.3836 - 0.01848*theta/DEG,0)
    res3=tf.where(theta>threshold2,tf.math.exp(-3.2e-2*theta/DEG + 2.0),0)
    a=(0.6511268210e-4*(theta/DEG-0.2614963683))*(theta/DEG*theta/DEG-134.7902422*theta/DEG+4558.524091)
    res2=tf.where(tf.math.logical_and(theta > threshold1,theta < threshold2),a,0)
    return res1+res2+res3
    
def courve_fun(detectors,core,t0,theta,phi,signal,mask):
    a_ivanov = a_ivanov_fun(theta)
    pfs,pps = pfs__pps(detectors,theta,phi,signal,mask)
    S_X = tf.where(pps>1e-10,pfs/pps,1)[:,0] # S_800
    S_X=tf.expand_dims(S_X,-1)
    courve = a_ivanov*1.3/tf.math.sqrt(S_X)
    courve = courve
#     S_X=tf.expand_dims(S_X,-1)
    return courve,S_X
def get_linsley_s(r, S):
    return 1.3*0.29*tf.math.pow((1 + r/R_L), 1.5)*tf.math.pow(S+1e-8, -0.3)*1e-3
def logPua(n,nbar):
    # print(n.shape,nbar.shape)
    last_part = 2*(n*tf.math.log(nbar/(n+1e-8)) + (n - nbar))

    nbar_logical=tf.where(nbar < 1e-90,True,False)
    n_logical1 = tf.where(n>1e-90,True,False)
    res = tf.zeros_like(n)
    res=tf.where(tf.logical_and(nbar_logical,n_logical1),-1e-6,res)

    else_nbar_logical = tf.logical_not(nbar_logical) 
    n_logical2 = tf.where(n<1e-20,True,False)

    res=tf.where(tf.logical_and(else_nbar_logical,n_logical2),-2*nbar,res)
    res=tf.where(tf.logical_and(else_nbar_logical,tf.logical_not(n_logical2)),last_part,res)
    return res
def chi2L(S_X,s_prof,mask,signal):
    s_fit = (expand_dims(S_X)*s_prof*mask) #/ DET_AREA
    # print('s_fit',s_fit[0,:,:,0])
    qs=signal#/ DET_AREA
    s_sigma2_huge = ( 2*qs/DET_AREA + tf.math.pow( 0.15*qs, 2 ) + 1e-6 )
    s_sigma2_small = 1.0/DET_AREA/DET_AREA
    s_sigma2 = tf.where(qs<0.1,s_sigma2_small,s_sigma2_huge)
    # print('s_err',s_sigma2[0,:,:,0])
    maskL2 = tf.where(s_fit>4.0,mask,0)
    # print('shapes L2',qs.shape,s_fit.shape,s_sigma2.shape,((qs - s_fit)*(qs - s_fit)/s_sigma2*maskL2).shape)
    chi2L2=tf.reduce_sum((qs - s_fit)*(qs - s_fit)/s_sigma2*maskL2,axis=(1,2))

    maskL3 = tf.where(s_fit<4.0,mask,0)
    # >0.1 just mask/don't work other
    chi2L3 = -0.4*tf.reduce_sum(2*tf.where(maskL3>0.1,qs*DET_AREA*tf.math.log(s_fit/qs) + DET_AREA*(qs-s_fit),0.0),axis=(1,2))
    # print('log',tf.math.log(s_fit/qs)[0,:,:,0])
    # print('chi2L3',chi2L3[0])
    # print('chi2L3 L2',chi2L3.shape,chi2L2.shape)
    N=tf.reduce_sum(maskL2,axis = (1,2,3))#+tf.reduce_sum(maskL3,axis = (1,2,3))
    # print('N L',N[0])
    return chi2L2, N ,s_sigma2 #+chi2L3
#TODO add mask
def chiT_by_param(real_time, detectors,detectors_z,t0,theta,phi,courve,mask,S_800):
    flat_reco = place_reconstruction(detectors,mask,t0,theta,phi,True)
    td,LDF,dist_core = courve_reconstruction(detectors_z,t0,theta,phi,courve)
    time_reco = t0 + flat_reco + td
    # print('time_reco',time_reco[3,:,:,0])
    lin_s = get_linsley_s(dist_core, expand_dims(S_800)*LDF)
    t_s = expand_dims(courve*tf.math.sqrt(S_800))*lin_s
    t_s = td
    t_sigma2=tf.math.sqrt(t0_err*t0_err + t_s*t_s) 
    chi2T = tf.reduce_sum(tf.math.pow((time_reco-real_time)/t_sigma2*mask,2),axis=(1,2)) #
    return chi2T,LDF
def optimization(data,iterats,num,detectors_rub=None,
                 add_mask = None, l_r=0.001, use_core=False,
                 use_L=True,S800_rub=None,optim_name = "Adam"):
    
    if optim_name == "Adam":
        optimizer = tf.keras.optimizers.Adam(l_r)
    elif optim_name == "SGD":
        optimizer = tf.keras.optimizers.SGD(l_r)
    elif optim_name == "Nadam":
        optimizer = tf.keras.optimizers.Nadam(l_r)
    else:
        raise OptimizerNameExcept("Wrong name optimizer")
    mask=data[:,:,:,3:4]
    #add mask
    if not(add_mask is None):
        mask = tf.where(~add_mask[:,:,:,1:2],mask,0)
    #
    
    signal = data[:,:,:,0:1]*mask
    # без этого лучше работатет хи-Т
#     mask = tf.where(signal==0,0,mask)
    real_time = (data[:,:,:,1:2]+data[:,:,:,2:3])*mask
    batch = data.shape[0]
    detectors_z = detectors_rub.copy()
    detectors=detectors_z[:,:,:,:2]
    core = tf.zeros((batch,2))
    use_z=True
    

    
    signal = tf.cast(signal,tf.float32)
    real_time = tf.cast(real_time,tf.float32)
    mask = tf.cast(mask,tf.float32)
    
    t0,theta,phi = place_params(detectors,real_time,mask)
    courve,S_X = courve_fun(detectors_z,core,t0,theta,phi,signal,mask)
    chi_list=[]
    if S800_rub is None:
        par = [t0,theta,phi,courve,S_X]
    else:
        par = [t0,theta,phi,courve]
    if use_core:
        par.append(core)
    params=[tf.Variable(p, True) for p in par]
    params_list=[]
    params_list.append(copy.deepcopy(params))
    for i in tqdm.notebook.tqdm_notebook(range(iterats)):
        with tf.GradientTape() as gr:  
            gr.watch(params)
            t0=params[0]
            theta=tf.math.abs(params[1])
            phi=params[2]
            courve=params[3]
            if not S800_rub is None:
                S_X=S800_rub
            else:
                S_X=params[4]
            if use_core:
                core=params[5][:,np.newaxis,np.newaxis,:]
                q=tf.math.reduce_sum(tf.math.pow(core,2),axis=-1)
                print('core: ', tf.math.reduce_mean(core), tf.math.reduce_std(core),tf.math.reduce_sum(tf.where(q>0.5*0.5,1,0)))
                detectors_z = tf.concat([detectors_rub[:,:,:,0:2]-core, detectors_rub[:,:,:,2:3]], axis= -1) 
            chi_T,LDF = chiT_by_param(real_time, detectors_rub,detectors_z,expand_dims(t0),theta,phi,courve,mask,S_X)
#             mask_ = tf.where(signal==0,0,mask)
            if not(add_mask is None):
                mask_ = tf.where(~add_mask[:,:,:,0:1],mask,0)
            signal_ = signal*mask_
            
            chi_L, N_L, _=chi2L(S_X,LDF*mask,mask_,signal_)
            if not(use_L):
                chi_L*=tf.constant(0,dtype=tf.float32)
            N_t = tf.reduce_sum(mask,axis=(1,2,3))
            N=tf.expand_dims(N_L+N_t,1)
            global_n = tf.where(N>7,N-7,1)
            chi = (chi_T +chi_L)/global_n
            print(tf.reduce_mean(chi_T/global_n),tf.reduce_mean(chi_L/global_n),tf.reduce_mean(chi),end='\n')
            grad=gr.gradient(chi,params)
            optimizer.apply_gradients(zip(grad, params))
            chi_list.append(chi)
            params_list.append(copy.deepcopy(params))
    for s1,p1 in enumerate(params_list):
        p2=tf.concat(p1,axis=1)
        params_list[s1]=p2
    params_list = np.array(params_list)
    return np.array(chi_list), params_list