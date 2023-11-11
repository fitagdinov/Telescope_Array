pi = tf.constant(math.pi)
UNIT=1
dist=tf.constant(1.2,dtype=tf.float32)# min dist between 2 detectors in km
c=tf.constant(299792.458,dtype=tf.float32)# # mk\s
NSEC=1e-9
R_L=30e-3
LINSLEY_r0=0.025
DET_AREA=1
t_sigma2=1
s_sigma2=1
tf_type=tf.float32
def xy_core(data,tf_type=tf.float32):
    mask=data[:,:,:,3]
    signal=data[:,:,:,0]*mask
    x=tf.cast(tf.repeat(tf.expand_dims(tf.range(0,6),0),6,axis=0),tf_type)
    y=tf.cast(tf.repeat(tf.expand_dims(tf.range(0,6),1),6,axis=1),tf_type)
    sum_signal=tf.cast(tf.reduce_sum(signal,axis=(1,2)),tf_type)
#     print(signal*x)
    cm_x=tf.reduce_sum(signal*x,axis=(1,2))/sum_signal
    cm_y=tf.reduce_sum(signal*y,axis=(1,2))/sum_signal
    return tf.concat([tf.expand_dims(cm_x,1),tf.expand_dims(cm_y,1)],axis=1)*dist
def expand_dims(vec):
    return tf.expand_dims(tf.expand_dims(vec,-1),-1)
# def create_matrix(x,y,t,mask,weight):
#     a11=expand_dims(tf.reduce_sum(x*x*weight,axis=1))
#     a12=expand_dims(tf.reduce_sum(x*y*weight,axis=1))
#     a13=expand_dims(tf.reduce_sum(x*weight,axis=1))
#     a22=expand_dims(tf.reduce_sum(y*y*weight,axis=1))
#     a23=expand_dims(tf.reduce_sum(y*weight,axis=1))
#     a33=expand_dims(tf.reduce_sum(mask*weight,axis=1))
#     a1=tf.concat([a11,a12,a13],axis=2)
#     a2=tf.concat([a12,a22,a23],axis=2)
#     a3=tf.concat([a13,a23,a33],axis=2)
#     A=tf.concat([a1,a2,a3],axis=1)
    
#     b1=expand_dims(tf.reduce_sum(x*t*weight,axis=1))
#     b2=expand_dims(tf.reduce_sum(y*t*weight,axis=1))
#     b3=expand_dims(tf.reduce_sum(t*weight,axis=1))
#     b=tf.concat([b1,b2,b3],axis=1)
# #     print(A,b)
#     return A,b
def create_matrix(x,y,t,mask,weight):
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
#     print(A,b)
    return A,b
def weight_fun(data):
    mask=data[:,:,:,3]
    signal=data[:,:,:,0]*mask
    sum_signal=tf.cast(tf.reduce_sum(signal,axis=(1,2)),tf_type)
    return signal/expand_dims(sum_signal)
def time_roconstruction(data,tf_type=tf.float32):
    real_time=tf.cast(data[:,:,:,2],tf_type)+tf.cast(data[:,:,:,1],tf_type)
    mask=tf.cast(data[:,:,:,3],tf_type)
    x=tf.cast(tf.repeat(tf.expand_dims(tf.range(0,6),0),6,axis=0),tf_type)*dist # add dist 
    y=tf.cast(tf.repeat(tf.expand_dims(tf.range(0,6),1),6,axis=1),tf_type)*dist # add dist
    # reshape transfrom like: 1st raw,2nd raw, ...
    # transform to 1 dim in one event
#     print(x)
    mask=tf.reshape(mask,(len(data),-1))
    x=tf.reshape(x,(1,-1))*mask
    y=tf.reshape(y,(1,-1))*mask
    t=tf.reshape(real_time,(len(data),-1))*mask
    weight=weight_fun(data)
    weight=tf.reshape(weight,(len(data),-1))
    A,b=create_matrix(x,y,t,mask,weight)
    return tf.linalg.solve(A,b)
def t0_fun(r_core,sol,dtype=tf.float32):
    #t_0=b+(r_cor;n)
    b=sol[:,2,0]
    n=sol[:,:2,0] # shape = batch 2
    #r_core= xy_core(data,tf_type=tf.float32)  shape = batch 2
    mul=n[:,0]*r_core[:,0]+n[:,1]*r_core[:,1]
    
    return tf.cast(tf.expand_dims(b,1)+tf.expand_dims(mul,1),dtype=dtype)
def find_phi(x,y):
    return tf.where(y>0,tf.math.acos(x),tf.math.acos(x)+pi)


# def find_angles(sol):
#     # solution from time_roconstruction
#     a_x=sol[:,0,0] #shape (batch,2)
#     a_y=sol[:,1,0]
#     tg_phi=a_y/(-a_x)
#     atan=tf.math.atan(tg_phi)
#     phi=tf.where(a_y>0,atan+pi,atan-pi)
#     phi=tf.where(a_x<0,atan,phi)
#     cos_theta=a_x/(dist*(1e6/c)*tf.math.cos(phi))
#     theta=tf.math.acos(cos_theta)
#     return theta,phi
def find_angles(sol):
    # solution from time_roconstruction
    a_x=sol[:,0,0] #shape (batch,2)
    a_y=sol[:,1,0]
    tg_phi=a_y/(-a_x)
    atan=tf.math.atan(tg_phi)
    phi=tf.where(a_y>0,atan+pi,atan-pi)
    phi=tf.where(a_x<0,atan,phi)
    a_z=tf.math.pow(tf.math.pow(1e6/c,2)-(tf.math.pow(a_x,2) + tf.math.pow(a_y,2)),0.5)
    cos_theta=a_z*(c/1e6)
    theta=tf.math.acos(cos_theta)
    return tf.expand_dims(theta,-1),tf.expand_dims(phi,-1)
def eta_fun(theta):
    return 3.97-1.79*(1/tf.math.cos(theta)-1)

def s_profile_tasimple(r_ta,theta):
    #change constant
    # I CHANGED. WAS 90e2 and 1000.0e2
    r = r_ta*UNIT
    eta=eta_fun(theta)# batch,1,1
    eta=tf.repeat(eta,6,axis=1)
    eta=tf.repeat(eta,6,axis=2)
    Rm = tf.constant(90e-3,dtype=tf.float32)
    R1 = tf.constant(1000e-3,dtype=tf.float32)
    return (tf.math.pow((r/Rm),-1.2)*tf.math.pow((1+r/Rm), -(eta-1.2))*tf.math.pow(1+(tf.math.pow(r,2)/R1/R1),-0.6))
def s_profile(r_ta, theta,r_plane):
    #r_ta shape batch,6,6
    # заменить r_plane на r_X 
    f800=s_profile_tasimple(expand_dims(tf.constant(0.8)), theta)
#     f800=tf.repeat(f800,r_ta.shape[0],0)
    print(s_profile_tasimple(r_ta, theta))
    return s_profile_tasimple(r_ta, theta)/s_profile_tasimple(r_plane, theta)

def s_s_pfs___s_s_pps(data,theta_,phi_,tf_type=tf.float32):
    mask=data[:,:,:,3]
    theta=tf.expand_dims(theta_,-1)
    phi=tf.expand_dims(phi_,-1)
    signal=data[:,:,:,0]*mask
    s_s_pfs=tf.reduce_sum(signal,axis=(1,2))
    
    
    x=tf.cast(tf.repeat(tf.expand_dims(tf.range(0,6),0),6,axis=0),tf_type)
    y=tf.cast(tf.repeat(tf.expand_dims(tf.range(0,6),1),6,axis=1),tf_type)
    
    x=tf.repeat(tf.expand_dims(x,0),len(data),0)*dist
    y=tf.repeat(tf.expand_dims(y,0),len(data),0)*dist
    
    print(x.shape,phi.shape)
    №
    r_plane=tf.math.sin(theta)*(tf.math.cos(phi)*x + tf.math.sin(phi*y))
    r=tf.math.sqrt(tf.math.pow(x,2)+tf.math.pow(y,2)-tf.math.pow(r_plane,2))
    #change eta
    s_s_pps=tf.reduce_sum(s_profile(r,theta,r_plane),axis=(1,2))
    return tf.cast(s_s_pfs,tf_type), tf.cast(s_s_pps,tf_type)
def S_X_fun(s_s_pfs, s_s_pps):
    res=tf.where(s_s_pps>1e-10,s_s_pfs/s_s_pps,1)
    return res

def a_ivanov_fun(theta):
    DEG=180*pi
    threshold1=25/DEG
    threshold2=35/DEG
    res1=tf.where(theta<threshold1,3.3836 - 0.01848*theta/DEG,0)
    res3=tf.where(theta>threshold2,tf.math.exp(-3.2e-2*theta/DEG + 2.0),0)
    a=(0.6511268210e-4*(theta/DEG-0.2614963683))*(theta/DEG*theta/DEG-134.7902422*theta/DEG+4558.524091)
    res2=tf.where(tf.math.logical_and(theta > threshold1,theta < threshold2),a,0)
    return res1+res2+res3
def courve_fun(a_ivanov,S_X):
    return a_ivanov*1.3/tf.math.sqrt(tf.expand_dims(S_X,-1))
def final_courve(data,theta,phi):
    s_s_pfs, s_s_pps=s_s_pfs___s_s_pps(data,theta,phi)
    print( s_s_pfs.shape, s_s_pps.shape)
    S_X=S_X_fun(s_s_pfs, s_s_pps)
    print(S_X.shape,'S_X')
    a_ivanov=a_ivanov_fun(theta)
    courve=courve_fun(a_ivanov,S_X)
    return courve
def linsley_t(r,S):
    return 0.67*tf.math.pow((1 + r/LINSLEY_r0), 1.5)*tf.math.pow(S, -0.5)*NSEC
def desaturate(S):
    #(S>200)?(exp(log(200)*(-0.7/0.3)+log(S)*1/0.3)):S;
    res=tf.where(S>200,tf.math.exp(math.log(200)*(-0.7/0.3)+tf.math.log(S)*1/0.3),S)
    return res
def chi_2_optimisation(data,alpha=0.1,iterats=20,tf_type=tf.float32):
    r_core=xy_core(data)
    sol=time_roconstruction(data)
    theta,phi=find_angles(sol)
    t0=t0_fun(r_core,sol)
    courve=final_courve(data,theta,phi)
    aprime=courve
    params=tf.concat([t0,theta,phi,aprime],axis=1)
    print(params.shape,'params')
    print(params)
#     params=tf.Variable([t0,theta,phi,aprime])
    

    for i in range(iterats):
#         with tf.GradientTape() as gr:  
#             gr.watch(params)
            
        #chi2 bloke
        n=tf.concat([tf.math.cos(phi),tf.math.sin(phi)],axis=1)*tf.math.sin(theta)
        real_time=tf.cast(data[:,:,:,2],tf_type)+tf.cast(data[:,:,:,1],tf_type)
        mask=tf.cast(data[:,:,:,3],tf_type)
        x=tf.cast(tf.repeat(tf.expand_dims(tf.range(0,6),0),6,axis=0),tf_type)
        y=tf.cast(tf.repeat(tf.expand_dims(tf.range(0,6),1),6,axis=1),tf_type)
        mask=tf.reshape(mask,(len(data),-1))
        x=tf.repeat(tf.expand_dims(x,0),len(data),0)*dist
        y=tf.repeat(tf.expand_dims(y,0),len(data),0)*dist
        r_core_66=tf.expand_dims(tf.expand_dims(r_core,2),3) #(10, 2, 1, 1)
        r_pl=(x-r_core_66[:,0])*tf.expand_dims(tf.expand_dims(n[:,0],-1),-1)+(y-r_core_66[:,1])*tf.expand_dims(tf.expand_dims(n[:,1],-1),-1)
#             print('r',tf.math.pow(x,2)+tf.math.pow(y,2)-tf.math.pow(r_pl,2))
        r=tf.math.sqrt(tf.math.pow(x,2)+tf.math.pow(y,2)-tf.math.pow(r_pl,2))
        r=tf.where(tf.math.is_nan(r),0,r)
        s_prof=s_profile(r,tf.expand_dims(theta,-1),r_pl) #LDF
        td=tf.expand_dims(aprime,-1)*linsley_t(r,s_prof)

        # КАК В СТАТЬЕ

        r_dist=r_pl
        print(s_prof(r_dist,theta).shape)
        td=tf.expand_dims(aprime,-1)*s_prof(r_dist,theta)*tf.math.pow((1+r/R_L),1.5)

        print('t_d',td)
        # КОНЕЦ
        return s_profile_tasimple(r_pl, tf.expand_dims(theta,-1))[0]
#             chi_T=tf.reduce_sum(tf.math.pow((tf.expand_dims(t0,-1)+r_pl+td-real_time),2)/t_sigma2,axis=(1,2))
            
#             # for no geometry fit
# #             s_s_pfs, s_s_pps=s_s_pfs___s_s_pps(data)
# #             S_X=fun_S_X(s_s_pfs, s_s_pps)
# #             s_fit = tf.reshape(S_X,(-1,1,1))*s_prof
# #             #dvalue[3] S 
# #             S=data[:,:,:,0]
# #             x_desaturate = desaturate(S)
# #             chi_L1=tf.reduce_sum(tf.math.pow(0.25*(x - s_fit),2)/s_sigma2,axis=(1,2))
            
#             print(chi_T)
#         grad=gr.gradient(chi_T,params)
#         print(grad)
#         params=params-alpha*grad
#         print(chi_T)