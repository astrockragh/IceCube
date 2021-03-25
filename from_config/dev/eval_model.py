import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.math import sin, cos, acos, abs, reduce_mean, subtract, square
import warnings
from scipy.stats import norm
warnings.filterwarnings("ignore")

eps=1e-5

def azi_alpha(y_true, y_reco):
    diffs = tf.minimum(abs(y_true[:, 2] - y_reco[:, 2]), abs(y_true[:, 2] - y_reco[:, 2])%(np.pi))
    u_azi = 180 / np.pi * tfp.stats.percentile(diffs, [50-34,50,50+34, 68])
    return u_azi.numpy()


def zeni_alpha(y_true, y_reco):
    diffs = tf.minimum(abs(y_true[:, 1] - y_reco[:, 1]), abs(y_true[:, 1] - y_reco[:, 1])%(np.pi))
    u_zen = 180 / np.pi * tfp.stats.percentile(diffs, [50-34,50,50+34, 68])
    return u_zen.numpy()

def azi_res(y_true, y_reco):
    diffs = tf.minimum(abs(y_true[:, 2] - y_reco[:, 2]), abs(y_true[:, 2] - y_reco[:, 2])%(2*np.pi))
    return diffs.numpy()


def zeni_res(y_true, y_reco):
    diffs = tf.minimum(abs(y_true[:, 1] - y_reco[:, 1]), abs(y_true[:, 1] - y_reco[:, 1])%(np.pi))
    return diffs.numpy()    

def alpha_from_angle(y_reco, y_true):
    zep, zet, azp, azt = y_reco[:,1], y_true[:,1], y_reco[:,2], y_true[:,2]
    cosalpha=abs(sin(zep))*cos(azp)*sin(zet)*cos(azt)+abs(sin(zep))*sin(azp)*sin(zet)*sin(azt)+cos(zep)*cos(zet)
    cosalpha-=tf.math.sign(cosalpha) * eps
    alpha=acos(cosalpha)
    return alpha


##consider making pull_plot

def performance_e_alpha(loader, test_step, metrics, save=False, save_path=''):
    '''Function to test and plot performance of Graph DL
    input should be dom pos x,y,z , time, charge(log10)
    target should be energy(log10),zenith angle, azimuthal angle, NOT unit vec 
    '''
    loss = 0
    prediction_list, target_list = [], []
    for batch in loader:
        inputs, targets = batch
        predictions, targets, out = test_step(inputs, targets)
        loss           += out
        
        prediction_list.append(predictions)
        target_list.append(targets)

    y_reco  = tf.concat(prediction_list, axis = 0).numpy()
    y_true  = tf.concat(target_list, axis = 0)
    y_true  = tf.cast(y_true, tf.float32).numpy()

    energy = y_true[:, 0]
    counts, bins = np.histogram(energy, bins = 10)

    xs = (bins[1:] + bins[: -1]) / 2

    w_energies, u_angles = [], []
    e_sig, alpha_sig = [], [] 
    old_energy, old_alpha = [], []
    zenith, azimuth = [], []
    for i in range(len(bins)-1):
        idx = np.logical_and(energy > bins[i], energy < bins[i + 1])

        w, u_angle, old = metrics(y_reco[idx, :], y_true[idx, :])
        old_energy.append(old[0])
        old_alpha.append(old[1])
        w_energies.append(w[1])
        u_angles.append(u_angle[1])
        e_sig.append([w[0], w[2]])
        alpha_sig.append([u_angle[0], u_angle[2]])
        zeni, azi=zeni_alpha(y_reco[idx,:], y_true[idx,:]), azi_alpha(y_reco[idx,:], y_true[idx,:])
        zenith.append(zeni)
        azimuth.append(azi)
    zenith, azimuth =  np.array(zenith), np.array(azimuth)
    fig, ax = plt.subplots(ncols = 3, nrows = 4, figsize = (12, 20))
    axesback=[(0,0), (1,0), (2,0), (3,0)]
    for i,j in axesback:
        a_ = ax[i][j].twinx()
        a_.step(xs, counts, color = "gray", zorder = 10, alpha = 0.7, where = "mid")
        a_.set_yscale("log")
        ax[i][j].set_xlabel("Log(E)")

    #structure: my metrics, old metrics, histogram
    
    # Energy reconstruction
    ax_top = ax[0]

    ax_top[0].errorbar(xs, w_energies,yerr=np.array(e_sig).T, fmt='k.',capsize=2,linewidth=1,ecolor='r',label='data')
    ax_top[0].plot(xs, old_energy, 'bo', label=r"$w(\Delta log(E))$"+'(old metric)')
    ax_top[0].set_title("Energy Performance")
    ax_top[0].set_ylabel(r"$\Delta log(E)$")

    # pull_e=(y_reco[:,0]-tf.reduce_mean(y_reco[:,0]))*np.sqrt(np.abs(y_reco[:,3]))
    # ax_top[1].hist(pull_e, label='Pull plot', bins=50, histtype='step')
    # ax_top[1].set_title("Solid angle pull plot)")
    # ax_top[1].set_title("Energy Performance (old metric)")
    # ax_top[1].set_ylabel(r"$w(\Delta log(E))$")

    ax_top[1].hist2d(y_true[:,0], y_reco[:,0], bins=100,\
                   range=[np.percentile(y_true[:,0],[1,99]), np.percentile(y_reco[:,0],[1,99])])
    ax_top[1].set_title("ML Reco/True")
    ax_top[1].set(xlabel="Truth (log(E))", ylabel="ML Reco (log(E))")
    res_e=abs(y_true[:,0]-y_reco[:,0])
    ax_top[2].hist2d(np.abs(y_reco[:,3]), res_e, bins=100, \
                   range=[np.percentile(np.abs(y_reco[:,3]),[1,99]), np.percentile(res_e,[1,99])])
    ax_top[2].set_title("ML Kappa correlation with Energy error")
    ax_top[2].set(xlabel=r"$\kappa$", ylabel=r"$\Delta E$")
    for axi in ax_top:
        axi.legend()
    #Zenith reconstructi

    # Alpha reconstruction
    ax_m=ax[1]

    ax_m[0].errorbar(xs, u_angles,yerr=np.array(alpha_sig).T, fmt='k.',capsize=2,linewidth=1,ecolor='r',label=r'Median $\pm \sigma$')
    ax_m[0].plot(xs, old_alpha, 'bo', label=r"$w(\Omega)$"+'(old metric)')
    ax_m[0].set_title("Angle Performance") 
    ax_m[0].set_ylabel(r"$\Delta \Omega$")
    
    alphas=alpha_from_angle(y_reco, y_true)

    pull_alpha=np.array(alphas-tf.reduce_mean(alphas))*np.sqrt(np.abs(y_reco[:,3]))
    pull_alpha=np.reshape(pull_alpha, -1)
    vals, x , _ = ax_m[1].hist(pull_alpha, label='Pull plot', bins=50, histtype='step', density=1)
    ax_m[1].plot(x, norm.pdf(x,0,1))
    ax_m[1].set_title("Solid angle pull plot)")
    # ax_m[1].set_ylabel(r"$w(\Omega)$")

    
    ax_m[2].hist2d(np.abs(y_reco[:,3]), alphas, bins=100, \
                   range=[np.percentile(np.abs(y_reco[:,3]),[1,99]), np.percentile(alphas,[1,99])])
    ax_m[2].set_title("ML Kappa correlation with angle error")
    ax_m[2].set(xlabel=r"$\kappa$", ylabel=r"$\Delta \Omega$")
    for axi in ax_m:
        axi.legend()
    #Zenith reconstruction
    ax_z=ax[2]

    ax_z[0].errorbar(xs, zenith[:,1],yerr=[zenith[:,0], zenith[:,2]], fmt='k.',capsize=2,linewidth=1,ecolor='r',label=r'Median $\pm \sigma$')
    ax_z[0].set_title("Zenith Performance")
    ax_z[0].plot(xs, zenith[:,3], 'bo', label='68th')
    ax_z[0].set_ylabel(r"$\Delta \Theta$")
    
    reszeni=np.abs(y_reco[:, 1]%(np.pi/2)-y_true[:,1])
    ax_z[1].hist(reszeni, label = "ML reco - Truth", histtype = "step", bins = 50)
    ax_z[1].hist(y_reco[:, 1]%(np.pi/2), label = "ML reco", histtype = "step", bins = 50)
    ax_z[1].hist(y_true[:, 1], label = "Truth", histtype = "step", bins = 50)
    
    ax_z[1].set_title("Zenith Perfomance")
    ax_z[1].set_ylabel(r"$\Theta$")
    
    ax_z[2].hist2d(np.abs(y_reco[:,3]), reszeni, bins=100,\
                  range=[np.percentile(np.abs(y_reco[:,3]),[1,99]), np.percentile(reszeni,[1,99])])
    ax_z[2].set_title("ML Kappa correlation with zenith error")
    ax_z[2].set(xlabel=r"$\kappa$", ylabel=r"$\Delta \Theta$")
    for axi in ax_z:
        axi.legend()
    #Azimuth reconstruction
    
    ax_az=ax[3]

    ax_az[0].errorbar(xs, azimuth[:,1],yerr=[azimuth[:,0], azimuth[:,2]], fmt='k.',capsize=2,linewidth=1,ecolor='r',label=r'Median $\pm \sigma$')
    ax_az[0].set_title("Azimuth Performance")
    ax_az[0].plot(xs, azimuth[:,3], 'bo', label='68th')
    ax_az[0].set_ylabel(r"$\Delta \phi$")
    
    resazi=np.abs(y_reco[:, 2]%(2*np.pi)-y_true[:,2])
    ax_az[1].hist(resazi, label = "ML reco - Truth", histtype = "step", bins = 50)
    ax_az[1].hist(y_reco[:, 2]%(2*np.pi), label = "ML reco", histtype = "step", bins = 50)
    ax_az[1].hist(y_true[:, 2], label = "Truth", histtype = "step", bins = 50)
    
    ax_az[1].set_title("Azimuth Perfomance")
    ax_az[1].set_ylabel(r"$\phi$")
    
    ax_az[2].hist2d(np.abs(y_reco[:,3]), resazi, bins=100,\
                  range=[np.percentile(np.abs(y_reco[:,3]),[1,99]), np.percentile(resazi,[1,99])])
    ax_az[2].set_title("ML Kappa correlation with azimuth error")
    ax_az[2].set(xlabel=r"$\kappa$", ylabel=r"$\Delta \phi$")
    for axi in ax_az:
        axi.legend()
    fig.tight_layout()
    if save:
        plt.savefig(save_path)
    return fig, ax

def performance_zeniazi(loader, test_step, metrics, save=False, save_path=''):

    '''Function to test and plot performance of Graph DL
    input should be dom pos x,y,z , time, charge(log10)
    target should be energy(log10),zenith angle, azimuthal angle, NOT unit vec 
    '''
    loss = 0
    prediction_list, target_list = [], []
    for batch in loader:
        inputs, targets = batch
        predictions, targets, out = test_step(inputs, targets)
        loss           += out
        
        prediction_list.append(predictions)
        target_list.append(targets)

    y_reco  = tf.concat(prediction_list, axis = 0).numpy()
    y_true  = tf.concat(target_list, axis = 0)
    y_true  = tf.cast(y_true, tf.float32).numpy()

    energy = y_true[:, 0]
    counts, bins = np.histogram(energy, bins = 10)
    xs = (bins[1:] + bins[: -1]) / 2
    w_energies, u_angles = [], []
    e_sig, alpha_sig = [], [] 
    old_energy, old_alpha = [], []
    zenith, azimuth = [], []
    for i in range(len(bins)-1):
        idx = np.logical_and(energy > bins[i], energy < bins[i + 1])
        e, old_e, angle, zeni, azi = metrics(y_reco[idx, :], y_true[idx, :])
        old_energy.append(old_e)
        old_alpha.append(angle[3])
        w_energies.append(e[1])
        u_angles.append(angle[1])
        e_sig.append([e[0], e[2]])
        alpha_sig.append([angle[0], angle[2]])
        zenith.append(zeni)
        azimuth.append(azi)
    zenith, azimuth =  np.array(zenith), np.array(azimuth)
    fig, ax = plt.subplots(ncols = 4, nrows = 3, figsize = (20, 15))
    axesback=[(0,0), (0,2), (1,0), (2,0)]
    for i,j in axesback:
        a_ = ax[i][j].twinx()
        a_.step(xs, counts, color = "gray", zorder = 10, alpha = 0.7, where = "mid")
        a_.set_yscale("log")
        ax[i][j].set_xlabel("Log(E)")

    
    # Energy reconstruction
    ax_top = ax[0]

    ax_top[0].errorbar(xs, w_energies,yerr=np.array(e_sig).T, fmt='k.',capsize=2,linewidth=1,ecolor='r',label='data')
    ax_top[0].plot(xs, old_energy, 'bo', label=r"$w(\Delta log(E))$"+'(old metric)')
    ax_top[0].set_title("Energy Performance")
    ax_top[0].set_ylabel(r"$\Delta log(E)$")


    ax_top[1].hist2d(y_true[:,0], y_reco[:,0], bins=100,\
                   range=[np.percentile(y_true[:,0],[1,99]), np.percentile(y_reco[:,0],[1,99])])
    ax_top[1].set_title("ML Reco/True")
    ax_top[1].set(xlabel="Truth (log(E))", ylabel="ML Reco (log(E))")
    ax_top[1].plot([np.percentile(y_true[:,0],[1]), np.percentile(y_true[:,0],[99])], [np.percentile(y_true[:,0],[1]), np.percentile(y_true[:,0],[99])], 'w--')

    ax_top[2].errorbar(xs, u_angles,yerr=np.array(alpha_sig).T, fmt='k.',capsize=2,linewidth=1,ecolor='r',label=r'Median $\pm \sigma$')
    ax_top[2].plot(xs, old_alpha, 'bo', label=r"$w(\Omega)$"+'(old metric)')
    ax_top[2].set_title("Angle Performance") 
    ax_top[2].set_ylabel(r"$\Delta \Omega$")


    for axi in ax_top:
        axi.legend()
    #Zenith reconstructi
    ax_z=ax[1]

    ax_z[0].errorbar(xs, zenith[:,1],yerr=[zenith[:,0], zenith[:,2]], fmt='k.',capsize=2,linewidth=1,ecolor='r',label=r'Median $\pm \sigma$')
    ax_z[0].set_title("Zenith Performance")
    ax_z[0].plot(xs, zenith[:,3], 'bo', label='68th')
    ax_z[0].set_ylabel(r"$\Delta \Theta$")
    
    reszeni=zeni_res(y_true, y_reco)
    ax_z[1].hist(reszeni, label = "ML reco - Truth", histtype = "step", bins = 50)
    ax_z[1].hist(y_reco[:, 1], label = "ML reco", histtype = "step", bins = 50)
    ax_z[1].hist(y_true[:, 1], label = "Truth", histtype = "step", bins = 50)
    
    ax_z[1].set_title("Zenith Perfomance")
    ax_z[1].set_ylabel(r"$\Theta$")

    ax_z[2].hist2d(y_true[:,1], y_reco[:,1], bins=100,range=[[0,np.pi], [0,np.pi]])
    ax_z[2].set_title("Zenith truth/reco correlation")
    ax_z[2].set(xlabel=r"True", ylabel=r"ML reco")
    ax_z[2].plot([0, np.pi], [0,np.pi], 'w--')

    ax_z[3].hist2d(np.abs(y_reco[:,3]), reszeni, bins=100,\
                  range=[np.percentile(np.abs(y_reco[:,3]),[1,99]), np.percentile(reszeni,[1,99])])
    ax_z[3].set_title("ML Kappa correlation with zenith error")
    ax_z[3].set(xlabel=r"$\kappa$", ylabel=r"$\Delta \Theta$")
    for axi in ax_z:
        axi.legend()
    #Azimuth reconstruction
    
    ax_az=ax[2]

    ax_az[0].errorbar(xs, azimuth[:,1],yerr=[azimuth[:,0], azimuth[:,2]], fmt='k.',capsize=2,linewidth=1,ecolor='r',label=r'Median $\pm \sigma$')
    ax_az[0].set_title("Azimuth Performance")
    ax_az[0].plot(xs, azimuth[:,3], 'bo', label='68th')
    ax_az[0].set_ylabel(r"$\Delta \phi$")
    
    resazi=azi_res(y_true,y_reco)
    ax_az[1].hist(resazi, label = "ML reco - Truth", histtype = "step", bins = 50)
    ax_az[1].hist(y_reco[:, 2]%(2*np.pi), label = "ML reco", histtype = "step", bins = 50)
    ax_az[1].hist(y_true[:, 2], label = "Truth", histtype = "step", bins = 50)
    
    ax_az[1].set_title("Azimuth Perfomance")
    ax_az[1].set_ylabel(r"$\phi$")

    ax_az[2].hist2d(y_true[:,2], y_reco[:,2], bins=100,\
                  range=[[0,2*np.pi], [0,2*np.pi]])
    ax_az[2].set_title("Zenith truth/reco correlation")
    ax_az[2].set(xlabel=r"True", ylabel=r"ML reco")
    ax_az[2].plot([0, 2*np.pi], [0,2*np.pi], 'w--')
    
    
    ax_az[3].hist2d(np.abs(y_reco[:,3]), resazi, bins=100,\
                  range=[np.percentile(np.abs(y_reco[:,3]),[1,99]), np.percentile(resazi,[1,99])])
    ax_az[3].set_title("ML Kappa correlation with azimuth error")
    ax_az[3].set(xlabel=r"$\kappa$", ylabel=r"$\Delta \phi$")
    for axi in ax_az:
        axi.legend()
    fig.tight_layout()
    if save:
        plt.savefig(save_path)
    return fig, ax

def performance_vM2D(loader, test_step, metrics, save=False, save_path=''):
    '''Function to test and plot performance of Graph DL
    input should be dom pos x,y,z , time, charge(log10)
    target should be energy(log10),zenith angle, azimuthal angle, NOT unit vec 
    '''
    loss = 0
    prediction_list, target_list = [], []
    for batch in loader:
        inputs, targets = batch
        predictions, targets, out = test_step(inputs, targets)
        loss           += out
        
        prediction_list.append(predictions)
        target_list.append(targets)

    y_reco  = tf.concat(prediction_list, axis = 0).numpy()
    y_true  = tf.concat(target_list, axis = 0)
    y_true  = tf.cast(y_true, tf.float32).numpy()

    energy = y_true[:, 0]
    counts, bins = np.histogram(energy, bins = 10)
    xs = (bins[1:] + bins[: -1]) / 2
    w_energies, u_angles = [], []
    e_sig, alpha_sig = [], [] 
    old_energy, old_alpha = [], []
    zenith, azimuth = [], []
    for i in range(len(bins)-1):
        idx = np.logical_and(energy > bins[i], energy < bins[i + 1])
        e, old_e, angle, zeni, azi = metrics(y_reco[idx, :], y_true[idx, :])
        old_energy.append(old_e)
        old_alpha.append(angle[3])
        w_energies.append(e[1])
        u_angles.append(angle[1])
        e_sig.append([e[0], e[2]])
        alpha_sig.append([angle[0], angle[2]])
        zenith.append(zeni)
        azimuth.append(azi)
    zenith, azimuth =  np.array(zenith), np.array(azimuth)
    fig, ax = plt.subplots(ncols = 4, nrows = 3, figsize = (20, 15))
    axesback=[(0,0), (0,2), (1,0), (2,0)]
    for i,j in axesback:
        a_ = ax[i][j].twinx()
        a_.step(xs, counts, color = "gray", zorder = 10, alpha = 0.7, where = "mid")
        a_.set_yscale("log")
        ax[i][j].set_xlabel("Log(E)")

    
    # Energy reconstruction
    ax_top = ax[0]

    ax_top[0].errorbar(xs, w_energies,yerr=np.array(e_sig).T, fmt='k.',capsize=2,linewidth=1,ecolor='r',label='data')
    ax_top[0].plot(xs, old_energy, 'bo', label=r"$w(\Delta log(E))$"+'(old metric)')
    ax_top[0].set_title("Energy Performance")
    ax_top[0].set_ylabel(r"$\Delta log(E)$")


    ax_top[1].hist2d(y_true[:,0], y_reco[:,0], bins=100,\
                   range=[np.percentile(y_true[:,0],[1,99]), np.percentile(y_reco[:,0],[1,99])])
    ax_top[1].set_title("ML Reco/True")
    ax_top[1].set(xlabel="Truth (log(E))", ylabel="ML Reco (log(E))")
    ax_top[1].plot([np.percentile(y_true[:,0],[1]), np.percentile(y_true[:,0],[99])], [np.percentile(y_true[:,0],[1]), np.percentile(y_true[:,0],[99])], 'w--')

    ax_top[2].errorbar(xs, u_angles,yerr=np.array(alpha_sig).T, fmt='k.',capsize=2,linewidth=1,ecolor='r',label=r'Median $\pm \sigma$')
    ax_top[2].plot(xs, old_alpha, 'bo', label=r"$w(\Omega)$"+'(old metric)')
    ax_top[2].set_title("Angle Performance") 
    ax_top[2].set_ylabel(r"$\Delta \Omega$")


    for axi in ax_top:
        axi.legend()
    #Zenith reconstructi
    ax_z=ax[1]

    ax_z[0].errorbar(xs, zenith[:,1],yerr=[zenith[:,0], zenith[:,2]], fmt='k.',capsize=2,linewidth=1,ecolor='r',label=r'Median $\pm \sigma$')
    ax_z[0].set_title("Zenith Performance")
    ax_z[0].plot(xs, zenith[:,3], 'bo', label='68th')
    ax_z[0].set_ylabel(r"$\Delta \Theta$")
    
    reszeni=zeni_res(y_true, y_reco)
    ax_z[1].hist(reszeni, label = "ML reco - Truth", histtype = "step", bins = 50)
    ax_z[1].hist(y_reco[:, 1], label = "ML reco", histtype = "step", bins = 50)
    ax_z[1].hist(y_true[:, 1], label = "Truth", histtype = "step", bins = 50)
    
    ax_z[1].set_title("Zenith Perfomance")
    ax_z[1].set_ylabel(r"$\Theta$")

    ax_z[2].hist2d(y_true[:,1], y_reco[:,1], bins=100,range=[[0,np.pi], [0,np.pi]])
    ax_z[2].set_title("Zenith truth/reco correlation")
    ax_z[2].set(xlabel=r"True", ylabel=r"ML reco")
    ax_z[2].plot([0, np.pi], [0,np.pi], 'w--')

    ax_z[3].hist2d(np.abs(y_reco[:,3]), reszeni, bins=100,\
                  range=[np.percentile(np.abs(y_reco[:,3]),[1,99]), np.percentile(reszeni,[1,99])])
    ax_z[3].set_title("ML Kappa correlation with zenith error")
    ax_z[3].set(xlabel=r"$\kappa$", ylabel=r"$\Delta \Theta$")
    for axi in ax_z:
        axi.legend()
    #Azimuth reconstruction
    
    ax_az=ax[2]

    ax_az[0].errorbar(xs, azimuth[:,1],yerr=[azimuth[:,0], azimuth[:,2]], fmt='k.',capsize=2,linewidth=1,ecolor='r',label=r'Median $\pm \sigma$')
    ax_az[0].set_title("Azimuth Performance")
    ax_az[0].plot(xs, azimuth[:,3], 'bo', label='68th')
    ax_az[0].set_ylabel(r"$\Delta \phi$")
    
    resazi=azi_res(y_true,y_reco)
    ax_az[1].hist(resazi, label = "ML reco - Truth", histtype = "step", bins = 50)
    ax_az[1].hist(y_reco[:, 2]%(2*np.pi), label = "ML reco", histtype = "step", bins = 50)
    ax_az[1].hist(y_true[:, 2], label = "Truth", histtype = "step", bins = 50)
    
    ax_az[1].set_title("Azimuth Perfomance")
    ax_az[1].set_ylabel(r"$\phi$")

    ax_az[2].hist2d(y_true[:,2], y_reco[:,2], bins=100,\
                  range=[[0,2*np.pi], [0,2*np.pi]])
    ax_az[2].set_title("Zenith truth/reco correlation")
    ax_az[2].set(xlabel=r"True", ylabel=r"ML reco")
    ax_az[2].plot([0, 2*np.pi], [0,2*np.pi], 'w--')
    
    
    ax_az[3].hist2d(np.abs(y_reco[:,4]), resazi, bins=100,\
                  range=[np.percentile(np.abs(y_reco[:,4]),[1,99]), np.percentile(resazi,[1,99])])
    ax_az[3].set_title("ML Kappa correlation with azimuth error")
    ax_az[3].set(xlabel=r"$\kappa$", ylabel=r"$\Delta \phi$")
    for axi in ax_az:
        axi.legend()
    fig.tight_layout()
    if save:
        plt.savefig(save_path)
    return fig, ax