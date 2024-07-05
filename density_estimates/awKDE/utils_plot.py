#Jam Sadiq
# A script for making plots for the paper

#import libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
from matplotlib import cm
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
import matplotlib.patches
from matplotlib.patches import Rectangle
import glob
import deepdish as dd

rcParams["text.usetex"] = True
rcParams["font.serif"] = "Computer Modern"
rcParams["font.family"] = "Serif"
rcParams["xtick.labelsize"]=18
rcParams["ytick.labelsize"]=18
rcParams["xtick.direction"]="in"
rcParams["ytick.direction"]="in"
rcParams["legend.fontsize"]=18
rcParams["axes.labelsize"]=18
rcParams["axes.grid"] = True
rcParams["grid.color"] = 'grey'
rcParams["grid.linewidth"] = 1.
#rcParams["grid.linestyle"] = ':'
rcParams["grid.alpha"] = 0.6

def normdata(dataV):
    normalized_data = (dataV - np.min(dataV)) / (np.max(dataV) - np.min(dataV))
    return normalized_data


dict_p = {'m1':'m_1', 'm2':'m_2', 'Xieff':'\chi_{eff}', 'chieff': '\chi_{eff}', 'DL':'D_L', 'logm1':'ln m_1', 'logm2': 'ln m_2', 'alpha':'\alpha'}
###########
def new2DKDE(XX, YY,  ZZ, LogMzvals, zvals, pdetvals, title='KDE', iterN=0, saveplot=False):
    contourlevels = np.logspace(-7, 0, 10)
    plt.figure(figsize=(8, 6))
    contourlevels = np.logspace(-7, 3, 10)[3:]
    # Plotting pcolormesh and contour
    #p = plt.pcolormesh(XX, YY, ZZ, cmap=plt.cm.get_cmap('Purples'),  norm=LogNorm(vmin=1e-5))
    CS = plt.contour(XX, YY, ZZ, colors='black', linestyles='dashed', linewidths=2, norm=LogNorm(), levels= np.logspace(np.log10(np.min(ZZ)), np.log10(np.max(ZZ)), 10))
    p = plt.scatter(LogMzvals, zvals, c=pdetvals, s=10, alpha=1.0, cmap=plt.cm.get_cmap('viridis'), norm=LogNorm())#vmin=1e-3))
    # Colorbar
    cbar = plt.colorbar(p)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(title, fontsize=18)  # Adjust the label text as neede
    # Axes and labels
    plt.tick_params(labelsize=15)
    plt.xlabel(r'$\log_{10}[Mz]$', fontsize=20)

    # Y-axis formatting
    scale_y = 1  # Adjust if needed, e.g., 1e3 for scaling
    ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / scale_y))
    plt.gca().yaxis.set_major_formatter(ticks_y)
    plt.ylabel(r"$z$", fontsize=20)

    # Title and layout
    plt.tight_layout()
    if saveplot ==True:
        plt.savefig("KDEIter{0}.png".format(iterN))
        plt.close()
    else:
        print("dont save")
        plt.show()
    return 0

from scipy.integrate import simpson
def  ThreePlots(XX, YY, ZZ,  TheoryMtot, Theory_z, logKDE=True,  iternumber=1, plot_name='median'):
    """
    Plots for paper only detected KDEas
    """
    # Integrate along the Mz-axis Now we compute 2D KDE using log10M not M
    kde_Mz = simpson(y=ZZ, x=YY, axis=0)
    # Integrate along the y-axis
    if logKDE==False:
        nonln_XX = np.exp(XX)/np.log(10)
        kde_Z = simpson(y=ZZ, x=nonln_XX, axis=1)
    else:
        kde_Z = simpson(y=ZZ, x=XX, axis=1)

    #kde_Mz = normdata(kde_Mz)
    #kde_Z = normdata(kde_Z)

    # Plot the original function
    fig, axs = plt.subplots(2, 2, figsize=(8, 8), gridspec_kw={'width_ratios': [1, 0.5], 'height_ratios': [0.5, 1]})
    #ax1 = fig.add_subplot(121, projection='3d')
    axs[1,0].contour(XX, YY, ZZ, cmap='viridis')
    axs[1,0].set_xlabel('Log10[Mz]')
    axs[1,0].set_ylabel('z')


    # Plot the integrated results
    #ax2 = fig.add_subplot(122)
    axs[0,0].plot(XX[0,:], kde_Mz, label='obs')
    #axs[0,0].plot(IntM, IntKDEM, 'k--', label='Intrinsic')
    axs[0,0].legend()
    #axs[0,0].set_xlabel('Log10[Mz]')
    axs[0,0].set_ylabel('p(Log10[Mz])')
    #axs[0,0].set_title('Integrated Results')

    #ax3 = fig.add_subplot(123)
    #axs[1,1].plot(z_eval, kde_Mz, label='Integral along y')
    #rotate plot
    axs[1,1].plot(kde_Z, YY[:, 0], label='obs')
    #axs[1,1].plot(IntZ, IntKDEz, 'k--', label='Intrinsic')
    #axs[1,1].plot(IntKDEz, IntZ, 'k--', label='Intrinsic')
    #axs[1,1].set_ylabel('z')
    axs[1,1].set_xlabel('p(z)')
    axs[1,1].legend()
    #axs[1,1].set_title('Integrated Results')
    axs[0, 1].axis('off')
    plt.tight_layout()

    plt.savefig(plot_name+"_IterativeKDE_plot_Iter{0}.png".format(iternumber))
    plt.close()
    #plt.show()
    weighthist = 1.0/100*np.ones(len(TheoryMtot))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.hist(np.log10(TheoryMtot), density=True, bins=20, weights=weighthist, histtype='step', label='detectedintrinsic', lw=2)
    ax1.plot(XX[0,:], kde_Mz, label='obs')
    ax1.semilogy()
    ax1.set_xlabel("Log10[Mz]", fontsize=18)
    #ax2.hist(z200, bins=10, histtype='step', label='observed', lw=2)
    ax2.hist(Theory_z, bins=50, histtype='step', density=True, weights=None, label='detected_intrinsic', lw=2)
    ax2.plot(YY[:, 0], kde_Z, label='obs')

    ax2.set_xlabel("z", fontsize=18)
    ax2.legend(fontsize=16)
    fig.suptitle('popIII model', fontsize=16)
    plt.tight_layout()
    plt.savefig(plot_name+"_withHistIterativeKDE_plot_Iter{0}.png".format(iternumber))
    #plt.show()
    plt.close()
    return  0





def Delnew2DKDE(XX, YY,  ZZ, title='KDE', iterN=0, saveplot=False):
    contourlevels = np.logspace(-5, 0, 15)
    plt.figure(figsize=(8, 6))
    contourlevels = np.logspace(-3, 2, 11)[3:]
    # Plotting pcolormesh and contour
    p = plt.pcolormesh(XX, YY, ZZ, cmap=plt.cm.get_cmap('Purples'),  norm=LogNorm(vmin=1e-7))
    CS = plt.contour(XX, YY, ZZ, colors='black', linestyles='dashed', linewidths=2, norm=LogNorm(vmin=1e-7))#, levels= contourlevels)
    
    # Colorbar
    cbar = plt.colorbar(p)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(title, fontsize=18)  # Adjust the label text as neede  
    # Axes and labels
    plt.tick_params(labelsize=15)
    plt.xlabel(r'$\log_{10}[Mz]$', fontsize=20)
    
    # Y-axis formatting
    scale_y = 1  # Adjust if needed, e.g., 1e3 for scaling
    ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / scale_y))
    plt.gca().yaxis.set_major_formatter(ticks_y)
    plt.ylabel(r"$z$", fontsize=20)
    
    # Title and layout
    plt.tight_layout()
    plt.suptitle("PopIII Model "+title)
    if saveplot==True:
        plt.savefig(title+'iter{0}.png'.format(iterN))
    else:
        print("notsaving")
    plt.close()
    #plt.show()
    return 0




##### Plots for Mock Data ###########################
def originaldataKDE_and_analyticfunc_plot(x_gridvalues, analyticfuncvalue, originalkdevalues, origalpha, origbw, m1_pow=0, pathplot='./', plot_analytic=False):
    """
    inputs:x_gridvalues, analyticfuncvalue, originalkdevalues, origalpha, origbw
    give plots with original dataKDE and analytic function with
    params of original mock data
    """
    fig = plt.figure(figsize=(8, 5))
    ax = plt.subplot(111)
    if plot_analytic==True:
        ax.plot(x_gridvalues, analyticfuncvalue, lw=2, ls='solid', color='k', label='Analytic')

    ax.plot(x_gridvalues, originalkdevalues, lw=2, ls='dashed', color='r', label=r'$orig KDE ,\alpha={0}, bw={1:.2f}$'.format(1, origbw))
    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
          ncol=2, fancybox=False)
    ax.set_ylim(ymin=1e-4)
    ax.semilogy()
    fig.savefig(pathplot+'OriginalmockdataKDE_Analyticfunc_pow_{0}.png'.format(m1_pow), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()
    return 0


def KDesPlot(x_gridvalues, true_y, errorkdeval, errorbw, kdelist, bwlist, pathplot='./', titlename=1, plot_analytic=False):
    """
    Inputs: x_gridvalues, true_y, errorkdeval, errorbw, kdelist, bwlist, pathplot
    we should use this when all iterations are done
    """
    fig = plt.figure(figsize=(9, 6))
    ax = plt.subplot(111)
    if plot_analytic ==True:
        ax.plot(x_gridvalues, true_y, lw=2, ls='dotted', color='cyan', label='True-dist')
    ax.plot(x_gridvalues, errorkdeval, lw=2, ls='dashed', color='b', label=r'$error,\alpha={0},bw={1:.2f}$'.format(1, errorbw))
    for l in range(len(kdelist)):
        plt.plot(x_gridvalues, kdelist[l], lw=1, ls='solid', label=r'$Iter{0} ,\alpha={1},bw={2:.2f}$'.format(l+1, 1, bwlist[l]))
    ax.set_xlabel(r"$m_1$")
    ax.set_ylabel(r"$p(m_1)$")
    # Put a legend to the right of the current axis
    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
          ncol=3, fancybox=False)
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.semilogy()
    fig.savefig(pathplot+'Errored_{0}Iterations_KDEs_step{1}.png'.format(len(kdelist), titlename), bbox_extra_artists=(lgd,), bbox_inches='tight')
    #plt.savefig(pathplot+'Errored_{0}Iterations_KDEs.png'.format(len(kdelist)))
    plt.close()
    return 0

def histogram_bwdata(datalist, dataname='bw',  pathplot='./', Iternumber=1):
    plt.figure(figsize=(8,6))
    plt.xlabel(dataname, fontsize=15)
    plt.hist(datalist, bins=20, color='red', fc='gray', histtype='step', alpha=0.8, density=True, label=dataname)
    plt.xlim(0.01, 1)
    plt.legend()
    plt.title("Iter{0}".format(Iternumber))
    plt.savefig(pathplot+dataname+"_histogramIter{0}.png".format(Iternumber), bbox_inches='tight')
    plt.close()
    return 0

def histogram_datalist(datalist, dataname='bw',  pathplot='./', Iternumber=1):
    """
    inputs: originalmean, shiftedmean, Iternumber, pathplot
    histogram to see shifted data alaonf with original data
    """
    plt.figure(figsize=(8,6))
    #plt.hist(datalist, bins=np.logspace(-2, 0, 25), color='red', fc='gray', histtype='step', alpha=0.8, density=True, label=dataname)
    plt.xlabel(dataname, fontsize=15)
    if dataname =='bw':
        plt.hist(datalist, bins=np.logspace(-1.5, -0.1, 15), color='red', fc='gray', histtype='step', alpha=0.8, density=True, label=dataname)
        minX, maxX = min(np.logspace(-2, 0, 25)) , max(np.logspace(-2, 0, 25))
        plt.xlim(0.01, 1.0)
        plt.semilogx()
    elif dataname =='alpha':
        plt.hist(datalist, bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], color='red', fc='gray', histtype='step', alpha=0.8, density=True, label=dataname)
        plt.xlim(0, 1)
    else:
        plt.xlim(min(datalist), max(datalist))

    plt.legend()
    plt.title("Iter{0}".format(Iternumber))
    plt.savefig(pathplot+dataname+"_histogramIter{0}.png".format(Iternumber), bbox_inches='tight')
    plt.close()
    return 0

def histogram_all_PEmockdata(samplelists_all, binvals=100, x_label='m1', pathplot='./'):
    """
    inputs
        samplelists: having a list of lists[arrays] mockPEsamples
        binvals: bydefault=100
        x_labels must be part of dict m1, m2 Xieff chieff DL
        pathplot: as label suggest give a pathwhere to save the plot

       get histogram combining all data
    """
    #list of lists data into a single row or column
    combinedata = np.concatenate(samplelists_all) 
    plt.figure(figsize=(8,6))
    plt.hist(combinedata, bins=binvals, color='red', fc='gray', histtype='step', alpha=0.8, density=True, label='mockdata')
    plt.xlabel(x_label, fontsize=18)
    plt.legend()
    plt.savefig(pathplot+x_label+'_combine_PEmockdata_histogram.png', bbox_inches='tight')
    plt.close()
    return 0


def histogram_original_shifted_data(originalmean, shiftedmean, Iternumber, pathplot='./'):
    """
    inputs: originalmean, shiftedmean, Iternumber, pathplot
    histogram to see shifted data alaonf with original data
    """
    plt.figure()
    plt.hist(originalmean, 30, color='red', fc='gray', histtype='step', alpha=0.8, density=True, label='sample')
    plt.hist(shiftedmean, 30, color='b', fc='gray', histtype='step', alpha=0.8, density=True, label='shifted-sample')
    plt.xlabel(r'$m_1$')
    plt.ylabel(r'$p(m_1)$')
    plt.legend()
    plt.savefig(pathplot+"errored_vs_shifted_data_histogramIter{0}.png".format(Iternumber), bbox_inches='tight')
    plt.close()
    return 0

def plot_scatter_originalmock_vs_error_data(originalmockdata, erroreddata, frac_gauss, mu_gauss, maxsigma_gauss, pathplot='./'):
    """
    inputs: originalmockdata, erroreddata, frac_gauss, mu_gauss, maxsigma_gauss, pathplot
    """
    plt.figure()
    plt.scatter(originalmockdata, erroreddata,color='red', label =r'$\alpha=0,fp={0:.2f},\mu={1:.2f},max \sigma={2:.2f}$'.format(frac_gauss, mu_gauss, maxsigma_gauss))
    plt.xlabel(r"$m_1$")
    plt.ylabel(r"$m_1 \pm \sigma{0}$".format(maxsigma_gauss))
    plt.legend()
    plt.savefig(pathplot+"scatter_original_vs_errored_data.png", bbox_inches='tight')
    plt.close()
    return 0


def erroreddataKDE_and_analyticfunc_plot(x_gridvalues, analyticfuncvalue, erroredkdevalues, origalpha, origbw, pathplot='./', plot_analytic=False):
    """
    inputs:x_gridvalues, analyticfuncvalue, originalkdevalues, alpha, bw
    give plots with error dataKDE and analytic function with
    params of original mock data
    """
    fig = plt.figure(figsize=(8, 5))
    ax = plt.subplot(111)
    if plot_analytic==True:
        ax.plot(x_gridvalues, analyticfuncvalue, lw=2, ls='solid', color='cyan', label='Analytic')
    ax.plot(x_gridvalues, erroredkdevalues, lw=2, ls='dashed', color='b', label=r'$erroredKDE ,\alpha={0}, bw={1:.2f}$'.format(1, origbw))
    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2, fancybox=False)
    ax.set_ylim(ymin=1e-4)
    ax.semilogy()
    fig.savefig(pathplot+'ErroredmockdataKDE_Analyticfunc.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()
    return 0

def averagekde_plot_mock(x_grid, analyticfuncvalue, kdelists, pathplot='./', titlename=1, plot_analytic=False, final=False):
    meankde = np.zeros_like(x_grid)
    for kde in kdelists:
        meankde += kde/len(kdelists)
    #5th, 50th and 95th percentile kde
    CI5 = np.percentile(kdelists, 5, axis=0)
    CI50 = np.percentile(kdelists, 50, axis=0)
    CI95 = np.percentile(kdelists, 95, axis=0)
    plt.figure(figsize=(8, 5))
    if plot_analytic==True:
        plt.plot(x_grid, analyticfuncvalue, 'k--', label='analytic')
    plt.plot(x_grid, meankde, 'r', label='median')
    plt.plot(x_grid, CI5, 'r-.', label='5th')
    plt.plot(x_grid, CI95, 'r-.', label='95th')
    plt.ylim(ymin = 1e-4)
    plt.xlim(2, 101)
    plt.semilogy()
    plt.legend()
    #plt.title('Iter{0}'.format(titlename))
    plt.savefig(pathplot+"averagekdeIter{0}.png".format(titlename), bbox_inches='tight')
    plt.close()
    if final ==True:
        np.savetxt('No_reweight_NoBots_x_analytic_kde50_5_95.txt', np.c_[x_grid, analyticfuncvalue, meankde, CI5, CI95])
    return meankde




#################################################################################
##################  GWTC3_dataPLOts #############################################
def averagekde_plot(x_grid, kdelists, pathplot='./', titlename=1, plot_analytic=False, plot_label='KDE'):
    meankde = np.zeros_like(x_grid)
    for kde in kdelists:
        meankde += kde/len(kdelists)
    #5th, 50th and 95th percentile kde
    CI5 = np.percentile(kdelists, 5, axis=0)
    CI50 = np.percentile(kdelists, 50, axis=0)
    CI95 = np.percentile(kdelists, 95, axis=0)
    np.savetxt(plot_label+"_1D_data_xgrid_median_5th_50th_95th_average_Iter{0}.txt".format(titlename), np.c_[x_grid, meankde, CI5, CI50, CI95])
    plt.figure(figsize=(8, 5))
    if plot_analytic==True:
        plt.plot(x_grid, analyticfuncvalue, 'k--', label='analytic')
    plt.plot(x_grid, meankde, 'r', label='median')
    plt.plot(x_grid, CI5, 'r-.', label='5th')
    plt.plot(x_grid, CI95, 'r-.', label='95th')
    plt.ylim(1e-4, 10.0)
    plt.xlim(0, 120)
    plt.semilogy()
    plt.legend()
    plt.xlabel(r'$m_1$', fontsize=15)
    if plot_label=='Rate':
        plt.ylabel(r'$\mathrm{d}\mathcal{R}/\mathrm{d}m_1[\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}\mathrm{M}_\odot^{-1}]$', fontsize=15)
    else:
        plt.ylabel(r'$p(m_1)$', fontsize=15)
    plt.title('Iter{0}'.format(titlename))
    plt.savefig(pathplot+plot_label+"1D_averagekdeIter{0}.png".format(titlename), bbox_inches='tight')
    plt.close()
    return meankde

def average2Dkde_plot(sample, m1vals, m2vals, XX, YY, kdelists, pathplot='./', titlename=1, plot_label='Rate', x_label='m1', y_label='m2', plot_error=False):
    sample1, sample2 = m1vals, m2vals
    CI5 = np.percentile(kdelists, 15.9, axis=0)
    CI50 = np.percentile(kdelists, 50, axis=0)
    CI95 = np.percentile(kdelists, 84.1, axis=0)
    
    #save the median data in hdf-file
    ZZ1 = CI95#.reshape(XX.shape)
    ZZ2 = CI5#.reshape(XX.shape)
    ZZM = CI50#.reshape(XX.shape)
    fig, axl = plt.subplots(1,1,figsize=(8,6))
    if y_label == 'm2':
        if plot_label=='Rate':
            #contourlevels = [5**i for i in range(-3, 4)] 
            contourlevels = np.logspace(-3, 2, 11)[3:]#np.array([1e-4, 3e-4,1e-3, 3e-3, 1e-2, 3e-2, 6e-2, 1e-1, 3e-1, 6e-1, 1, 10, 18])
        else:
            contourlevels =np.logspace(-5, 0.5, 8) #np.array([1e-5, 3e-4,1e-3, 3e-3, 1e-2, 3e-2, 6e-2, 1e-1, 3e-1, 6e-1, 1])
        #p = axl.pcolormesh(XX, YY, CI50, cmap=plt.cm.get_cmap('Purples'), vmin=1e-5, norm=LogNorm(),  label=r'$p(m_1, m_2)$')
        p = axl.pcolormesh(XX, YY, CI50, cmap=plt.cm.get_cmap('Purples'), vmin=1e-5, norm=LogNorm(),  label=r'$p(log(m_1), log(m_2))$')
        cbar = plt.colorbar(p, ax= axl)
        if plot_label =='KDE':
            cbar.set_label(r'$p(m_1, m_2)$',fontsize=18)
            #cbar.set_label(r'$p(ln \, m_1, \ln \, m_2)$',fontsize=18)
        CS = axl.contour(XX, YY, CI50, colors='black', levels=contourlevels ,vmin=1e-5,linestyles='dashed', linewidths=2, norm=LogNorm())
        axl.scatter(sample1, sample2,  marker="+", color="r", s=20)
        axl.scatter(sample2, sample1,  marker="+", color="r", s=20)
        sx = np.arange(3, 100, 1)
        ##uncomment
        axl.fill_between(sx, sx, 100, color='white', alpha=1, zorder=50)
        #axl.fill_between(np.arange(0, 100), np.arange(0, 100),100 , color='white',alpha=1,zorder=50)
        #axl.fill_between(np.arange(0, 50), np.arange(0, 50), 50 , color='white',alpha=1,zorder=100)
        #axl.set_ylim(2.3, 101)
        #axl.set_ylim(2.3, 101)
        axl.set_ylim(3, 101)
        axl.tick_params(axis="y",direction="in")
        axl.yaxis.tick_right()
        axl.yaxis.set_ticks_position('both')
        axl.set_ylabel(r'$m_2\,[M_\odot]$', fontsize=18)

    else:
        contourlevels = np.array([ 1e-5, 3e-4,1e-3, 3e-3, 1e-2, 3e-2, 6e-2, 1e-1, 3e-1, 6e-1, 1])
        p = axl.pcolormesh(XX, YY, CI50, cmap=plt.cm.get_cmap('Purples'), norm=LogNorm(),  vmin=1e-6, label=r'$p(m_1,'+dict_p[y_label] + ')$')
        CS = axl.contour(XX, YY, CI50, colors='black', levels=contourlevels, vmin=1e-5,  linestyles='dashed', linewidths=2, norm=LogNorm())
        axl.scatter(sample1, sample2,  marker="+", color="r", s=20)
        cbar = plt.colorbar(p, ax= axl)
        if plot_label =='KDE':
            cbar.set_label(r'$p(m_1, '+dict_p[y_label]+ ')$',fontsize=18)
    cbar.ax.tick_params(labelsize=20)
    axl.tick_params(labelsize=18)
    axl.set_xlabel(r'$m_1\,[M_\odot]$', fontsize=18)
    scale_y = 1#e3
    ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_y))
    axl.yaxis.set_major_formatter(ticks_y)
    axl.set_xlim(3, 100.1)
    #axl.set_ylim(-1, 1)  #for Xieff case
    axl.set_ylabel(r'$'+dict_p[y_label]+'$')
    axl.set_ylabel(r'$m_2\,[M_\odot]$', fontsize=18)
    axl.loglog()
    if plot_label =='KDE':
        #cbar.set_label(r'$p(m_1, '+dict_p[y_label]+ ')$',fontsize=18)
        cbar.set_label(r'$p(m_1, m_2)$',fontsize=18)
    else:
        if y_label == 'm2':
            cbar.set_label(r'$\mathrm{d^2}\mathcal{R}/\mathrm{d\,ln}\,m_1\mathrm{d\,ln}\,m_2[\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}\mathrm{ln\, M}_\odot^{-2}]$',fontsize=18)
            #cbar.set_label(r'$\mathrm{d}\mathcal{R}/\mathrm{d}m_1\mathrm{d}m_2[\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}\mathrm{M}_\odot^{-2}]$',fontsize=18)
        else:
            cbar.set_label(r'$\mathrm{d}\mathcal{R}/\mathrm{d}m_1\mathrm{d}'+dict_p[y_label]+'[\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}\mathrm{M}_\odot^{-1}]$',fontsize=18)
    axl.set_aspect('equal')
    fig.tight_layout()
    plt.savefig(pathplot+'m1_'+y_label+'_2D'+plot_label+'Iter{0}.png'.format(titlename), bbox_inches='tight')
    plt.close()

    errval = (ZZ1-ZZ2)/(2*ZZM)
    if plot_error==True:
        ig, axl = plt.subplots(1,1,figsize=(8,6))
        inr, maxr = np.min((ZZ1-ZZ2)/(2*ZZM)), np.max((ZZ1-ZZ2)/(2*ZZM))
        rint("min, max relative error = ", minr, maxr)
        inr, maxr = abs(minr), abs( maxr)
        contourlevels = np.array([0.2, 0.5, 0.9, 1.2, 1.5, 2, 3])
    #p = axl.pcolormesh(XX, YY, (ZZ1-ZZ2)/(2*ZZM), vmin=minr, cmap=plt.cm.get_cmap('Purples'), norm=LogNorm(), label=r'$\sigma_b(m_1, m_2)/p(m_1, m_2)$')
        p = axl.pcolormesh(XX, YY, (ZZ1-ZZ2)/(2*ZZM), cmap=plt.cm.get_cmap('Purples'), norm=LogNorm(), label=r'$\sigma_b(m_1, m_2)/(m_1, m_2)$')
        CS = axl.contour(XX, YY, (ZZ1-ZZ2)/(2*ZZM), levels=ncontourlevels, colors='black', linestyles='dashed', vmin=minr, linewidths=2, norm=LogNorm())
        axl.clabel(CS, inline=True, fontsize=14, fmt='%1.1f')
        axl.clabel(CS, inline=True, fontsize=14, fmt='%1.1f')
        axl.scatter(samplem1, samplem2, marker="+", color="r", s=20)
        cbar = plt.colorbar(p, ax= axl)
        cbar.ax.tick_params(labelsize=20)
        if plot_label=='KDE':
            cbar.set_label(r'$\sigma_b(m_1, m_2)/p(m_1, m_2)$',fontsize=18)
        else:
            cbar.set_label(r'$\sigma_b(m_1, m_2)/\mathrm{d}\mathcal{R}(m_1, m_2)$',fontsize=18)
        axl.tick_params(labelsize=15)
        axl.set_xlabel(r'$m_1\,[M_\odot]$')
    #axl.plot(0.001*np.loadtxt('get.txt'), label='dh')
        scale_y = 1#e3
        ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_y))
        axl.yaxis.set_major_formatter(ticks_y)
        axl.set_ylabel(r"$m_2 \, [M_\odot]$")
        axl.set_ylim(2, 102)
        axl.set_xlim(2, 102)
    #axl.fill_between(np.arange(0, 100), np.arange(0, 100),100 , color='white',alpha=1,zorder=50)
    #axl.fill_between(np.arange(0, 50), np.arange(0, 50), 50 , color='white',alpha=1,zorder=100)
        fig.tight_layout()
        plt.savefig(pathplot+"ErrorRatiokdeIter{0}.png".format(titlename), bbox_inches='tight')
        plt.close()
    return CI50


#def average_allkde_plot(x_grid, kdelists, pathplot='./', plot_analytic=False):
def average_allkde_plot(x_grid, analyticfuncvalue, kdelists, pathplot='./', plot_analytic=False):
    meankde = np.zeros_like(x_grid)
    for kde in kdelists:
        meankde += kde/len(kdelists)
    #5th, 50th and 95th percentile kde
    CI5 = np.percentile(kdelists, 5, axis=0)
    CI50 = np.percentile(kdelists, 50, axis=0)
    CI95 = np.percentile(kdelists, 95, axis=0)
    #np.savetxt('Finall100to1000_iteration_average_rate_xgrid_median_5th_50th_95th_.txt', np.c_[x_grid, meankde, CI5, CI50, CI95])
    plt.figure(figsize=(8, 5))
    if plot_analytic==True:
        plt.plot(x_grid, analyticfuncvalue, ls='solid', color='cyan', label='analytic')
    plt.plot(x_grid, meankde, 'r', linewidth = 2. , label='median')
    plt.plot(x_grid, CI5, 'r--', linewidth = 2., label='5th')
    plt.plot(x_grid, CI95, 'r--', linewidth = 2.,  label='95th')
    plt.ylim(ymin = 3e-4)
    plt.semilogy()
    plt.xlabel('$m_1$ [$\mathrm{M}_\odot$]', fontsize=14)
    plt.ylabel('$\mathrm{d}\mathcal{R}/\mathrm{d}m_1[\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}\mathrm{M}_\odot^{-1}]$', fontsize=14)
    plt.grid(ls=':',color='grey', alpha = 0.6)
    #plt.legend()
    #plt.title('combine_KDE')
    plt.savefig(pathplot+"combined_rate1D.png", bbox_inches='tight')
    #plt.savefig(pathplot+"combinedkde.png")
    plt.close()

def add_random_n_places(a, n):
    # Generate a float version
    out = a.astype(float)
    # Generate unique flattened indices along the size of a
    idx = np.random.choice(a.size, n, replace=False)

    # Assign into those places ramdom numbers in [-1,1)
    out.flat[idx] += np.random.uniform(low=-1, high=1, size=n)
    return out

def bandwidth_correlation(bwlist,  number_corr=100, error=0.02, param='bw', pathplot='./', log=True):
    """
    change the number for when we want before and after 
    Make a scatter plot with some random 
    error added to bwd 
    scatter point bw[i]  vs bw[i+1]
    and also plot correlation coefficient 
    """
#    listN = np.array(bwlist[:-1]) #from 0 to N-1 values of array
#    listNp1 = np.array(bwlist[1:]) #1 to Nth (last value of array)
#
#    listN = np.random.lognormal(np.log(listN), error)
#    listNp1 = np.random.lognormal(np.log(listNp1), error)
#    plt.figure(figsize=(8,5))
#    plt.scatter(listN , listNp1, marker= '+')
#    plt.xlabel(param+" N")
#    plt.ylabel(param+" N+1")
#    plt.semilogy()
#    plt.semilogx()
#    if error > 0.0:
#        plt.savefig(pathplot+param+"_correlation_scatter.png")
#    else:
#        plt.savefig(pathplot+param+"_correlation.png")
#    plt.close()
#
    #new correlation
    Cxy  =  []
    iternumber = []
    for i in range(int(len(bwlist)/2)):
        iternumber.append(i+1)
        if log==False:
            M = np.corrcoef(bwlist[i+1: ], bwlist[ :-i-1])
        else:
            M = np.corrcoef( np.log(bwlist[i+1: ]), np.log(bwlist[ :-i-1]))
        Cxy.append(M[0][1])
    print([Cxy[i] for i in range(10)])
    plt.figure(figsize=(8,4))
    plt.plot(iternumber[number_corr:], Cxy[number_corr:],'+-')
    plt.xlabel('Number of iterations')
    plt.ylabel(r'$C_{01}$', fontsize=14)
    plt.semilogx()
    plt.grid(True)
    plt.savefig(pathplot+param+"C01_iter_after_100.png", bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(8, 4))
    plt.plot(iternumber[:number_corr], Cxy[:number_corr],'+-')
    plt.xlabel('Number of iterations')
    plt.ylabel(r'$C_{01}$')
    #plt.semilogx()
    plt.grid(True)
    plt.savefig(pathplot+param+"C01_iter_before_100.png", bbox_inches='tight')
    plt.close()


    #corrcoefficient
    Cxy  =  []#[1.0]
    #for i in range(number_corr, int(len(bwlist)/2)):
    for i in range(int(len(bwlist)/2)):
        if log == True:
            M = np.corrcoef( np.log(bwlist[i+1: ]), np.log(bwlist[ :-i-1]))
        else:
            M = np.corrcoef(bwlist[i+1: ], bwlist[ :-i-1])
        Cxy.append(M[0][1])
    #print([Cxy[i] for i in range(10)])
    plt.figure(figsize=(8, 4))
    plt.plot(Cxy,'+-')
    plt.xlabel('Number of iterations')
    plt.xlim(xmin=1)
    plt.ylabel(r'$C_{xy}$')
    plt.semilogx()
    plt.grid(True)
    plt.savefig(pathplot+param+"CxyiterN.png", bbox_inches='tight')
    plt.close()
    return 0



def twoDKDEplot(sample, XX, YY, ZZ, pathplot='./', x_label='m1', y_label='m2'):
    sample1, sample2 = sample[:, 0], sample[:, 1]
    fig, axl = plt.subplots(1,1,figsize=(8,6))
    if y_label == 'm2':
        contourlevels = np.logspace(-5, 0.5, 10)#np.array([ 1e-5, 1e-4, 3e-4,1e-3, 3e-3, 1e-2, 3e-2, 6e-2, 1e-1, 3e-1, 6e-1, 1])
        p = axl.pcolormesh(XX, YY, ZZ, cmap=plt.cm.get_cmap('Purples'), vmin=1e-5, norm=LogNorm(),  label=r'$p(m_1, m_2)$')
        cbar = plt.colorbar(p, ax= axl)
        #cbar.set_label(r'$p(\ln\, m_1,\ln \, m_2)$',fontsize=18)
        cbar.set_label(r'$p(m_1, m_2)$', fontsize=18)
        CS = axl.contour(XX, YY, ZZ, colors='black', levels=contourlevels ,vmin=1e-5,linestyles='dashed', linewidths=2, norm=LogNorm())
        axl.scatter(sample1, sample2,  marker="+", color="r", s=20)
        axl.scatter(sample2, sample1,  marker="+", color="r", s=20)
        sx = np.arange(3, 100, 1)
        ### uncomment this
        axl.fill_between(sx, sx, 100, color='white', alpha=1, zorder=5)
        #axl.fill_between(np.arange(0, 100), np.arange(0, 100),100 , color='white',alpha=1,zorder=50)
        #axl.fill_between(np.arange(0, 50), np.arange(0, 50), 50 , color='white',alpha=1,zorder=100)
        axl.set_ylim(2.3, 101)
        #axl.set_ylim(2, 101)

        axl.tick_params(axis="y",direction="in")
        axl.yaxis.tick_right()
        axl.yaxis.set_ticks_position('both')

        axl.set_ylabel(r'$m_2\,[M_\odot]$', fontsize=18)

    else:
        p = axl.pcolormesh(XX, YY, ZZ, cmap=plt.cm.get_cmap('Purples'), norm=LogNorm(),  label=r'$p(m_1,'+dict_p[y_label] + ')$')
        CS = axl.contour(XX, YY, ZZ, colors='black', linestyles='dashed', linewidths=2, norm=LogNorm())
        axl.scatter(sample1, sample2,  marker="+", color="r", s=20)
        cbar = plt.colorbar(p, ax= axl)
        cbar.set_label(r'$p(m_1, '+dict_p[y_label]+ ')$',fontsize=18)
    cbar.ax.tick_params(labelsize=20)
    axl.tick_params(labelsize=18)
    axl.set_xlabel(r'$m_1\,[M_\odot]$', fontsize=18)
    scale_y = 1#e3
    ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_y))
    axl.yaxis.set_major_formatter(ticks_y)
    axl.set_xlim(3, 100.1)
    axl.set_aspect('equal')

    #axl.set_xlim(2, 101)
    #axl.set_ylabel(r'$'+dict_p[y_label]+'$')
    fig.tight_layout()

    #axl.loglog()
    plt.savefig(pathplot+'m1_'+y_label+'_2DKDE.png', bbox_inches='tight')
    plt.close()
    return 0



#### If we just have one iteration Rate 
def twoDRateplot(sample, XX, YY, ZZ, pathplot='./', x_label='m1', y_label='m2', paramtype='linear'):
    sample1, sample2 = sample[:, 0], sample[:, 1]
    fig, axl = plt.subplots(1,1,figsize=(8,6))
    if y_label == 'm2':
        contourlevels = np.logspace(-4, 1.2, 8)#np.array([1e-4, 3e-4,1e-3, 3e-3, 1e-2, 3e-2, 6e-2, 1e-1, 3e-1, 6e-1, 1, 10, 15, 20])
        p = axl.pcolormesh(XX, YY, ZZ, cmap=plt.cm.get_cmap('Purples'), vmin=1e-5, norm=LogNorm())
        cbar = plt.colorbar(p, ax= axl)
        cbar.set_label(r'$\mathrm{d^2}\mathcal{R}/\mathrm{d\,ln}\,m_1\mathrm{d\,ln}\,m_2[\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}\mathrm{ln\, M}_\odot^{-2}]$',fontsize=18)
        #cbar.set_label(r'$\mathrm{d}\mathcal{R}/\mathrm{d} m_1 \mathrm{d} m_2[\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}\mathrm{M}_\odot^{-2}]$',fontsize=18)

        #CS = axl.contour(XX, YY, ZZ, colors='black', levels=contourlevels ,vmin=1e-5,linestyles='dashed', linewidths=2, norm=LogNorm())
        CS = axl.contour(XX, YY, ZZ, colors='black',vmin=1e-5,linestyles='dashed', linewidths=2, norm=LogNorm())
        axl.scatter(sample1, sample2,  marker="+", color="r", s=20)
        axl.scatter(sample2, sample1,  marker="+", color="r", s=20)
        #axl.fill_between(np.arange(0, 100), np.arange(0, 100), 100 , color='white',alpha=1,zorder=50)
        #axl.fill_between(np.arange(0, 50), np.arange(0, 50), 50 , color='white',alpha=1,zorder=100)
        sx = np.arange(3, 100, 1)
        axl.fill_between(sx, sx, 100, color='white', alpha=1, zorder=50)
        #axl.fill_between(sx, 0, 100, color='white', alpha=1, zorder=50, where=sx>1)
        
        #axl.fill_between(np.arange(100), np.arange(100), 100, color='white' , alpha=1, zorder=50)
        axl.set_ylim(2.3, 101)
        axl.set_ylabel(r'$m_2\,[M_\odot]$', fontsize=18)

    else:
        p = axl.pcolormesh(XX, YY, ZZ, cmap=plt.cm.get_cmap('Purples'), norm=LogNorm(),  label=r'$p(m_1,'+dict_p[y_label] + ')$')
        CS = axl.contour(XX, YY, ZZ, colors='black', linestyles='dashed', linewidths=2, norm=LogNorm())
        axl.scatter(sample1, sample2,  marker="+", color="r", s=20)
        cbar = plt.colorbar(p, ax= axl)
        cbar.set_label(r'$\mathrm{d}\mathcal{R}/\mathrm{d}m_1\mathrm{d}'+dict_p[y_label]+'[\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}\mathrm{M}_\odot^{-1}]$',fontsize=18)
    
    cbar.ax.tick_params(labelsize=20)
    axl.tick_params(labelsize=18)
    axl.set_xlabel(r'$m_1\,[M_\odot]$', fontsize=18)
    scale_y = 1#e3
    ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_y))
    axl.yaxis.set_major_formatter(ticks_y)
    axl.set_xlim(3, 100.1)
    axl.set_aspect('equal')
    #axl.set_ylabel(r'$'+dict_p[y_label]+'$')
    fig.tight_layout()
    axl.loglog()
    plt.savefig(pathplot+'m1_'+y_label+'_2DRate.png', bbox_inches='tight')
    plt.close()
    return 0


import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

def combine_twoD1Dplot(XX, YY, ZZ, sample1, sample2, x_gridvalues, analyticfuncvalue1, analyticfuncvalue2, originalkdevalues1, originalkdevalues2, pathplot='./'):
    # Plot the conditional distributions
    fig = plt.figure(figsize=(7, 7))
    gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[2, 1])
    # gs.update(wspace=0., hspace=0.)

    # Plot surface on top left
    ax1 = plt.subplot(gs[0])
#   Plot 2D
    contourlevels = np.logspace(-4, 1.2, 8)#np.array([1e-4, 3e-4,1e-3, 3e-3, 1e-2, 3e-2, 6e-2, 1e-1, 3e-1, 6e-1, 1, 10, 15, 20])
    p = ax1.pcolormesh(XX, YY, ZZ, cmap=plt.cm.get_cmap('Purples'), vmin=1e-5, norm=LogNorm())

    CS = ax1.contour(XX, YY, ZZ, colors='black',vmin=1e-5,linestyles='dashed', linewidths=2, norm=LogNorm())
    ax1.scatter(sample1, sample2,  marker="+", color="r", s=20)
    ax1.scatter(sample2, sample1,  marker="+", color="r", s=20)
    ax1.yaxis.set_label_position('right')
    sx = np.arange(3, 100, 1)
    ax1.fill_between(sx, sx, 100, color='white', alpha=1, zorder=50)
    ax1.set_ylim(2.3, 101)
    ax1.set_ylabel(r'$m_2\,[M_\odot]$', fontsize=15)
    ax1.set_xlabel(r'$m_1\,[M_\odot]$', fontsize=15)
    scale_y = 1#e3
    ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_y))
    ax1.yaxis.set_major_formatter(ticks_y)
    ax1.set_xlim(3, 100.1)
    #axl.set_ylabel(r'$'+dict_p[y_label]+'$')
    fig.tight_layout()


    # Plot y
    ax2 = plt.subplot(gs[1])
    ax2.plot(analyticfuncvalue1, x_gridvalues,lw=2, ls='solid', color='cyan', label='True')

    ax2.plot(originalkdevalues1, x_gridvalues,lw=2, ls='dashed', color='b', label=r'$p(m_1)$')
    #ax2.plot(x_gridvalues, originalkdevalues1, lw=2, ls='dashed', color='b', label=r'$p(m_1)$')
    lgd = ax2.legend(ncol=2, fancybox=False)
    ax2.set_xlim(xmin=1e-4)
    ax2.semilogx()

    # Plot x
    ax3 = plt.subplot(gs[2])
    ax3.plot(x_gridvalues, analyticfuncvalue2, lw=2, ls='solid', color='cyan', label='True')

    ax3.plot(x_gridvalues, originalkdevalues2, lw=2, ls='dashed', color='b', label=r'$p(m_2)$')
    lgd = ax3.legend(ncol=2, fancybox=False)
    ax3.set_ylim(ymin=1e-4)
    ax3.semilogy()
    # Clear axis 4 and plot colarbar in its place
    ax4 = plt.subplot(gs[3])
    ax4.set_visible(False)
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('left', size='20%', pad=0.05)
    cbar = fig.colorbar(p, cax=cax)
    cbar.ax.set_ylabel('density: $p(m_1, m_2)$', fontsize=13)

    cbar.ax.tick_params(labelsize=20)
    ax4.tick_params(labelsize=15)
    plt.savefig(pathplot+'testplot.png', bbox_inches='tight')
    plt.close()


def average_rate_component(m1val, m2val, ratem1, ratem2, truem1, truem2 ,pathplot='./'):
    plt.figure(figsize=[7.2, 4.8])
    plt.semilogy(m1val, np.median(ratem1, axis=0), 'C0-', label='m')
    plt.semilogy(m1val, truem1, color='k', lw=2, ls='--', label='Analytic')
    plt.fill_between(m1val, np.percentile(ratem1, 95., axis=0) , np.percentile(ratem1, 5., axis=0), color='C0', alpha=0.2)
    #plt.semilogy(m2val, numpy.median(ratem2, axis=0) / m2val, 'C1-', label='Secondary mass')
    #plt.semilogy(m2val, truem2, color='magenta', lw=2, ls='-.',  label='True m2')
    #plt.fill_between(m2val, numpy.percentile(ratem2, 95., axis=0) / m2val, numpy.percentile(ratem2, 5., axis=0) / m2val, color='C1', alpha=0.2)
    plt.ylim(3.e-4, 10.)
    plt.xlim(3., 100.)
    plt.grid(True)
    plt.legend(loc='upper right', fontsize='large')
    plt.ylabel(r'$p(m_1)$', size='large')
    plt.xlabel(r' $m_1\,[M_\odot]$', size='large')
    plt.savefig(pathplot+'combined_OneDmass_average_with_true.png', bbox_inches='tight')
    plt.close()

def corre_tom(series, before_use_buf=5, bufsize=100, quantity='bandwidths', log=True, pathplot='./'):
    """
    series is array of values of bw of alpha
    chnage quantity with alpha for alpha correlation
    """
    #if quantity != 'bandwidths':
    #    log = False
    series_before = series[:before_use_buf]
    series_after = series[before_use_buf:]
    plt.plot(series_before, '+')
    #plt.plot(np.convolve(series_before, np.ones(5)/5., mode='valid'), '-')
    plt.plot(np.convolve(series_before, np.ones(10)/10., mode='valid'), '-.')
    if log:
        plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel(quantity.rstrip('s'))
    plt.savefig(pathplot+'before_'+quantity+'_series.png', bbox_inches='tight')
    plt.close()

    # autocorrelation
    if log:
        series_before = np.log(series_before)
        series_after = np.log(series_after)
    
    # pre-buffer
    acorr_before = []
    #iternumber = []
    for i in range(int(2. * len(series_before) / 3.)):  # lag between samples
        #iternumber.append(i+1)
        # correlate the series with i samples removed from the beginning and end resp.
        M = np.corrcoef(series_before[:-i-1], series_before[i+1:])
        acorr_before.append(M[0][1])
        #print([Cxy[i] for i in range(10)])
    plt.figure()
    plt.plot(np.arange(len(acorr_before))[1:], acorr_before[1:], '+-')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation of '+quantity)
    plt.semilogx()
    plt.grid()
    plt.savefig(pathplot+'before_'+quantity+'_autocorr.png', bbox_inches='tight')
    plt.close()
    
    # using buffer
    acorr_after = []
    #iternumber = []
    for i in range(int(2. * len(series_after) / 3.)):  # lag between samples
        #iternumber.append(i+1)
        # correlate the series with i samples removed from the beginning and end resp.
        M = np.corrcoef(series_after[:-i-1], series_after[i+1:])
        acorr_after.append(M[0][1])
        #print([Cxy[i] for i in range(10)])
    plt.figure()
    plt.plot(np.arange(len(acorr_after))[1:], acorr_after[1:], '+')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation of '+quantity)
    #plt.semilogx()
    plt.grid()
    plt.savefig(pathplot+'after_'+quantity+'_autocorr.png', bbox_inches='tight')
    plt.close()
    
