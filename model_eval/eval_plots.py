import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.math import sin, cos, acos, abs, reduce_mean, subtract, square
import warnings
from scipy.stats import norm
from ExternalFunctions import nice_string_output, add_text_to_ax
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
    diffs = tf.minimum(abs(y_true[:, 2] - y_reco[:, 2]), abs(y_true[:, 2] - y_reco[:, 2])%(np.pi))
    return diffs.numpy()


def zeni_res(y_true, y_reco):
    diffs = tf.minimum(abs(y_true[:, 1] - y_reco[:, 1]), abs(y_true[:, 1] - y_reco[:, 1])%(np.pi))
    return diffs.numpy()    

def alpha_from_angle(y_true, y_reco):
    zep, zet, azp, azt = y_reco[:,1], y_true[:,1], y_reco[:,2], y_true[:,2]
    cosalpha=abs(sin(zep))*cos(azp)*sin(zet)*cos(azt)+abs(sin(zep))*sin(azp)*sin(zet)*sin(azt)+cos(zep)*cos(zet)
    cosalpha-=tf.math.sign(cosalpha) * eps
    alpha=acos(cosalpha)
    return alpha

def cos_angle(y_reco, y_true):
    zep, zet, azp, azt = y_reco[:,1], y_true[:,1], y_reco[:,2], y_true[:,2]
    # cosalpha=abs(sin(zep))*cos(azp)*sin(zet)*cos(azt)+abs(sin(zep))*sin(azp)*sin(zet)*sin(azt)+cos(zep)*cos(zet)
    cosalpha=abs(sin(zep))*abs(sin(zet))*cos(azp-azt)+cos(zep)*cos(zet) #check for double absolutes
    cosalpha-=tf.math.sign(cosalpha) * eps
    return cosalpha

def energy_angle_zeniazi(y_reco, y_true):
    
    energy_metric = tfp.stats.percentile(tf.math.abs(tf.subtract(y_true[:, 0], y_reco[:, 0])), [50-34, 50, 50+34]) 
    #for comparison
    classic_stat=tfp.stats.percentile(tf.subtract(y_true[:, 0], y_reco[:, 0]), [25, 75]) 
    w_energy=tf.subtract(classic_stat[1],classic_stat[0])/1.349

    alpha= acos(cos_angle(y_reco, y_true))
    angle_resi = 180 / np.pi * alpha #degrees
    angle_metric  = tfp.stats.percentile(angle_resi, [50-34,50,50+34, 68])


    return energy_metric, float(w_energy), angle_metric.numpy(), zeni_alpha(y_true, y_reco), azi_alpha(y_true, y_reco) 


def performance_vM2D(y_reco, y_true, metrics=energy_angle_zeniazi, save=False, save_path='', bins=10):
    '''Function to evaluate performance. Give reco values as E, zeni, azi, 1/sig_zeni^2, 1/sig_azi^2 
    '''
    energy = y_true[:, 0]
    counts, bins = np.histogram(energy, bins = bins)
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
    all_e, old_e, all_a, all_z, all_az = metrics(y_reco, y_true)
    title={'median +- sig': ''}
    summary={'E': f'{all_e[1]:.3f}, {all_e[0]:.3f}<->{all_e[2]:.3f}',
    'Omega': f'{all_a[1]:.3f}, {all_a[0]:.3f}<->{all_a[2]:.3f}',
    'Zeni': f'{all_z[1]:.3f}, {all_z[0]:.3f}<->{all_z[2]:.3f}',
    'Azi': f'{all_az[1]:.3f}, {all_az[0]:.3f}<->{all_az[2]:.3f}'}
    summary_old={'Old metrics:': ' ',
    'E': np.around(old_e,3),
    'Omega': f'{all_a[3]:.3f}',
    'Zeni': f'{all_z[3]:.3f}',
    'Azi': f'{all_az[3]:.3f}'}
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

#     text=nice_string_output(title, extra_spacing=2, decimals=3)
#     add_text_to_ax(0.05, 0.92, text, ax_top[3], fontsize=14)

#     text=nice_string_output(summary, extra_spacing=2, decimals=3)
#     add_text_to_ax(0.05, 0.85, text, ax_top[3], fontsize=14)

#     text=nice_string_output(summary_old, extra_spacing=2, decimals=3)
#     add_text_to_ax(0.05, 0.4, text, ax_top[3], fontsize=14)
#     for axi in ax_top:
#         axi.legend()
    print(summary_old, summary)
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
    ax_az[2].set_title("Azimuth truth/reco correlation")
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