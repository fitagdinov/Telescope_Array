import tensorflow as tf
import math
pi = tf.constant(math.pi,dtype=tf.float32)
UNIT=1
dist=tf.constant(1.2,dtype=tf.float32)# min dist between 2 detectors in km
c=tf.constant(299792.458,dtype=tf.float32)# # km\s
NSEC= tf.constant(1e9/c,dtype=tf.float32)# in rubsov's code is a time for 1.2 km 1.2/c*1e9
R_L=tf.constant(30e-3,dtype=tf.float32)#
LINSLEY_r0=tf.constant(0.025,dtype=tf.float32)#
DET_AREA=tf.constant(3,dtype=tf.float32)#
s_min = tf.constant([[0.3]],dtype=tf.float32)
s_max = tf.constant([[1.8]],dtype=tf.float32)
t_err_res=tf.constant(c/1e6,dtype=tf.float32)#
t0_err=tf.constant(30,dtype=tf.float32)#
dist=tf.constant(1.2,dtype=tf.float32)
tf_type = tf.float32
R_error = 0.15
def init(data,tf_type = tf.float32):
    data=tf.cast(data,tf_type)
    mask = tf.expand_dims(data[:,:,:,3],-1)
    signal = tf.expand_dims(data[:,:,:,0],-1)*self.mask
    mask = tf.where(self.signal==0,0,self.mask)
    real_time = tf.expand_dims((data[:,:,:,1] + data[:,:,:,2]),-1)*self.mask 
    signal = tf.cast(self.signal,self.tf_type)
    real_time = tf.cast(self.real_time,self.tf_type)
    mask = tf.cast(self.mask,self.tf_type)
    batch = data.shape[0]
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
    phi = tf.math.atan2( -a_x, -a_y )
    return t0,tf.expand_dims(theta,-1),tf.expand_dims(phi,-1)
def detectors_core(detectors,core):
    detectors_c=detectors-tf.reshape(core,(-1,1,1,2))
    return detectors_c
def place_reconstruction(detectors,mask,t0,theta,phi):
#     t0,theta,phi = place_params(detectors,real_time,mask)
    t0=expand_dims(t0) # shape (batch,1,1,1)
    theta=expand_dims(theta)
    phi =expand_dims(phi)
    n=-tf.concat([tf.math.cos(phi)*tf.math.sin(theta),tf.math.sin(phi)*tf.math.sin(theta)],axis=-1)*(1e6/c)
#     change the order of succession
    t_place =  t0 + detectors[:,:,:,1:2]*n[:,:,:,0:1] + detectors[:,:,:,0:1]*n[:,:,:,1:2]
    t_place = t_place*mask
    return t_place
def eta_fun(theta):
    x=theta*180/3.14

    e1 = 3.97 - 1.79*(tf.math.abs(1.0/tf.math.cos(theta)) - 1.0)
    e2 = ((((((-1.71299934e-10*x + 4.23849411e-08)*x -3.76192000e-06)*x
               + 1.35747298e-04)*x -2.18241567e-03)*x + 1.18960682e-02)*x
             + 3.70692527e+00)
    res = tf.where(x<62.7,e1,e2)
    res =tf.where(res>0,res,0)
    return res
def s_profile_tasimple(r_ta,theta):
    r = r_ta
    eta=eta_fun(theta)# batch,1,1
    eta=tf.repeat(eta,6,axis=1)
    eta=tf.repeat(eta,6,axis=2)
    # eta shape is batch,6,6
    Rm = tf.constant(0.09,dtype=tf.float32)
    R1 = tf.constant(1,dtype=tf.float32)
    return (tf.math.pow((r/Rm),-1.2)*tf.math.pow((1+r/Rm), -(eta-1.2))*tf.math.pow(1+(tf.math.pow(r,2)/R1/R1),-0.6))
def s_profile(r_ta, theta):
    #r_ta shape batch,6,6
    f800=s_profile_tasimple(expand_dims(tf.constant(0.8)), theta)
    return s_profile_tasimple(r_ta, theta)/f800
def pfs__pps(detectors,core,t0,theta,phi,signal,mask):
#     t0,theta,phi = self.place_params()
    # u can read from t_place if in place_reconstruction use core shift
#     detectors = detectors_core(detectors,core)
    t0=expand_dims(t0) # shape (batch,1,1,1)
    theta=expand_dims(theta)
    phi = expand_dims(phi)
    n=-tf.concat([tf.math.cos(phi)*tf.math.sin(theta),tf.math.sin(phi)*tf.math.sin(theta)],axis=-1)
    t_place = detectors[:,:,:,1:2]*n[:,:,:,0:1] + detectors[:,:,:,0:1]*n[:,:,:,1:2] # not has t0
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
    pfs,pps = pfs__pps(detectors,core,t0,theta,phi,signal,mask)
    S_X = tf.where(pps>1e-10,pfs/pps,1)[:,0]
    S_X=tf.expand_dims(S_X,-1)
    courve = a_ivanov*1.3/tf.math.sqrt(S_X)
    courve = courve
#     S_X=tf.expand_dims(S_X,-1)
    return courve,S_X
def linsley_t(r,S):
    return 0.67*tf.math.pow((1 + r/LINSLEY_r0), 1.5)*tf.math.pow(S, -0.5)/1e3
def courve_reconstruction(detectors,t0,theta,phi,courve):
    # u can read from t_place if in place_reconstruction use core shift
#     detectors = detectors_core()
    t0=expand_dims(t0) # shape (batch,1,1,1)
    theta=expand_dims(theta)
    phi = expand_dims(phi)
    n=-tf.concat([tf.math.cos(phi)*tf.math.sin(theta),tf.math.sin(phi)*tf.math.sin(theta)],axis=-1)
    t_place = detectors[:,:,:,1:2]*n[:,:,:,0:1] + detectors[:,:,:,0:1]*n[:,:,:,1:2]
    dist_core = tf.expand_dims(tf.reduce_sum(tf.math.pow(detectors,2),axis=-1),axis=-1) - tf.math.pow(t_place,2)
    dist_core = tf.where(dist_core>R_error*R_error,tf.math.sqrt(dist_core),R_error)
    LDF=s_profile(dist_core,theta)
    td=expand_dims(courve)*linsley_t(dist_core,LDF)/NSEC
    return td,LDF
def logPua(n,nbar):
    print(n.shape,nbar.shape)
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
    s_fit = expand_dims(S_X)*s_prof*mask
    qs=signal
    s_sigma2 = ( 2*qs/DET_AREA + tf.math.pow( 0.15*qs, 2 ) + 1e-6 )
    maskL2 = tf.where(qs>4.0,mask,0)
    S=S_X
    chi2L2=tf.reduce_sum((qs - s_fit)*(qs - s_fit)/s_sigma2*maskL2,axis=(1,2))
#     maskL3 = tf.where(s_fit<4.0,mask,0)
#         chi2L3 = -tf.reduce_sum(0.4*self.logPua(S*self.DET_AREA, s_fit*self.DET_AREA)*maskL3,axis=(1,2))
#         print(chi2L2.shape,chi3L.shape)
    return chi2L2  #+ chi3L
def optimization(data,iterats,num):
    Adam = tf.keras.optimizers.Adam()
    signal = data[:,:,:,0:1]
    real_time = data[:,:,:,1:2]+data[:,:,:,2:3]
    mask=data[:,:,:,3:4]
    
    #detectors
    detectors_orig  = detectors_init(data)
    core = core_(detectors_orig ,signal)
    detectors = detectors_core(detectors_orig ,core)
    t0,theta,phi = place_params(detectors,real_time,mask)
    # ??? don't work without that 
#     pfs,pps= pfs__pps(detectors,core,t0,theta,phi,signal)
    courve,S_X = courve_fun(detectors,core,t0,theta,phi,signal,mask)
    chi_list=[]
#     t0=t0
#     theta=tf.expand_dims(theta,-1)
#     phi=tf.expand_dims(phi,-1)
    print(t0.shape,theta.shape,phi.shape,courve.shape,core.shape,S_X.shape)
    par = [t0,theta,phi,courve,core,S_X]
    params=[tf.Variable(p, True) for p in par]
#         params=tf.concat([t0,theta,phi,courve,core,S_X],axis=1)
    params_list=[]
    params_list.append(params)
    for i in tqdm.notebook.tqdm_notebook(range(iterats)):
        with tf.GradientTape() as gr:  
            gr.watch(params)
            
            t0=params[0]
            theta=params[1]
            phi=params[2]
            courve=params[3]
            core=params[4]
            S_X=params[5]
            detectors = detectors_core(detectors_orig ,core)
            t_place = place_reconstruction(detectors,mask,t0,theta,phi)
            td,s_prof = courve_reconstruction(detectors,t0,theta,phi,courve)  #update LDF <==> s_profile
            t_sigma2=(t0_err*t0_err + td*td) * t_err_res
            time_reco = t_place +td
            chi_T=tf.reduce_sum(tf.math.pow((time_reco-real_time)*mask,2)/t_sigma2,axis=(1,2))
            chi_L=chi2L(S_X,s_prof,mask,signal)
            chi = chi_T +chi_L
#                 print(tf.reduce_mean(chi_T),tf.reduce_mean(chi_L),end='\r')
#             if num:
#                 print(chi_T[num],chi_L[num],[np.array(i[num]) for i in params],end='\r')
            chi_list.append(chi)
            grad=gr.gradient(chi,params)
            Adam.apply_gradients(zip(grad, params))
            params_list.append(params)
    for s1,p1 in enumerate(params_list):
        p2=tf.concat(p1,axis=1)
        params_list[s1]=p2
    params_list = np.array(params_list)
    return np.array(chi_list), params_list