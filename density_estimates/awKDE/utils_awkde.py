"""
This is a module for functions related to awkde code
"""
from awkde import GaussianKDE
import numpy as np
import operator
import scipy.stats as st

######################## One dimensional case ########################
def kde_awkde(x, x_grid, alp=0.5, gl_bandwidth='silverman', ret_kde=False):
    """Kernel Density Estimation with awkde 
    inputs:
      x = training data: array-like, shape (n_samples, n_features)
      for one dimensional case use x[:, newaxis] 
      x_grid = testing data must be array-like, shape (n_samples, n_features)
      alp = smoothing factor for local bw
      gl_bandwidth = global bw for kde
      kwargs:
       ret_kde optional 
        if True kde will be output with estimated kde-values 
    """
    kde = GaussianKDE(glob_bw=gl_bandwidth, alpha=alp, diag_cov=True)
    kde.fit(x[:, np.newaxis])
    if isinstance(x_grid, (list, tuple, np.ndarray)) == False:
        y = kde.predict(x_grid)
    else:
        y = kde.predict(x_grid[:, np.newaxis])

    if ret_kde == True:
        return kde, y
    return y


def loocv_awkde(sample, bwchoice, alphachoice):
    """
    Use specific choice of alpha and bw 
    we use Leave one out cross validation
    on Awkde kde fit
    LOOCV:
    we train n-1 of the n sample leaving one
    in n iterations and compute kde fit on 
    n-1 samples. For each iteration we  use this kde 
    to predict the missed ith sample of ith iteration.
    We take log of this predicted value and
    sum these all values (N values if len Sample = N)
    We output this sum.
    fom  = log(predict(miss value)) is called 
    a figure of merit.
    """
    if bwchoice not in ['silverman', 'scott']:
        bwchoice = float(bwchoice) #bwchoice.astype(np.float) #for list
    fom = 0.0
    for i in range(len(sample)):
        leave_one_sample, miss_sample = np.delete(sample, i), sample[i]
        y = kde_awkde(leave_one_sample, miss_sample, alp=alphachoice, gl_bandwidth=bwchoice)
        #fom += np.log(y)  # for 1D case
        fom += np.log(np.mean(y))
    return fom


def get_optimized_bw_kde_using_loocv(sample, alphagrid, bwgrid, fomplot=False):
    """ 
    Given grid of alpha and bw choice it will get 
    figure of merit from loocv_awkde function for each choice
    return a dictionary of FOM:  FOM[(bw, alpha)]
    and  optimal param and FOM at these params:  optbw , optalpha, FOM[(optbw , optalpha)] 
    """
    FOM= {}
    for gbw in bwgrid:
        for alphavals in alphagrid:
            FOM[(gbw, alphavals)] = loocv_awkde(sample, gbw, alphavals)
    optval = max(FOM.items(), key=operator.itemgetter(1))[0]
    optbw, optalpha  = optval[0], optval[1]
    maxFOM = max(FOM) 

    return  FOM, optbw, optalpha, maxFOM


def get_kde(samplevalues, x_gridvalues, alphagrid, bwgrid, ret_kde=False):
    """
    inputs: samplevalues, x_gridvalues, alphagrid, bwgrid 
        make sure order of alpha and bwds
    return: kdeval, optbw, optalpha
      make sure of order of outputs
    """
    FOMdict, optbw, optalpha, maxFOM = get_optimized_bw_kde_using_loocv(samplevalues,
    alphagrid, bwgrid)
    print("bw = ", optbw)
    if ret_kde:
        kde, kdeval = kde_awkde(samplevalues, x_gridvalues, alp=optalpha, gl_bandwidth=optbw, ret_kde=ret_kde)
        return kde, kdeval, optbw, optalpha
    kdeval = kde_awkde(samplevalues, x_gridvalues, alp=optalpha, gl_bandwidth=optbw)
    return kdeval, optbw, optalpha

################################################################################
#######for Two(multi)D KDE
def N_dim_KDE_awkde(x, x_grid, alp=0.5, gl_bandwidth='silverman', ret_kde=False):
    """Kernel Density Estimation with awkde only with Gaussian Kernel

    ret_kde optional 
      if True kde will be output with estimated kde-values 
    """
    kde = GaussianKDE(glob_bw=gl_bandwidth, alpha=alp, diag_cov=True)
    kde.fit(x)
    y = kde.predict(x_grid)

    if ret_kde == True:
        return kde, y

    return y


def twoD_loocv_awkde(sample, bwchoice, alphachoice):
    """
    two dimsional case only changes are
    np.delete(sample, i) to np.delete(sample, i,0)
    np.log((y))  to  np.log(np.mean(y))
    """
    if bwchoice not in ['silverman', 'scott']:
        bwchoice = float(bwchoice) #bwchoice.astype(np.float) #for list
    fom = 0.0
    for i in range(len(sample)):
        leave_one_sample, miss_sample = np.delete(sample, i,0), sample[i]
        y = N_dim_KDE_awkde(leave_one_sample, miss_sample, alp=alphachoice, gl_bandwidth=bwchoice)
        fom += np.log(np.mean(y))
    return fom

def Ndim_get_optimized_bw_kde_using_loocv(sample, alphagrid, bwgrid, fomplot=False):
    """
    inputs:
      sample: [training set1, training set2]
      bwchoice: choices of bandwidth for kde
      alphachoice: choice of kde parameter [0.1, 1]
      kwargs:fomplot if True plot FOM as function of alpha and bandwidth
   output:
      returns optimized alpha, bandwidth, and maxFigureOfMerit value
      remember the order
      optional plot of fom as function of alpha bandwidth
    Explanation:
    Given grid of alpha and bw choice it will get
    figure of merit from loocv_awkde function for each choice
    return a dictionary of FOM:  FOM[(bw, alpha)]
    and  optimal param and FOM at these params:  optbw , optalpha, FOM[(optbw , optalpha)]
    """
    FOM= {}
    for gbw in bwgrid:
        for alphavals in alphagrid:
            FOM[(gbw, alphavals)] = twoD_loocv_awkde(sample, gbw, alphavals)
    optval = max(FOM.items(), key=operator.itemgetter(1))[0]
    optbw, optalpha  = optval[0], optval[1]
    maxFOM = FOM[(optbw, optalpha)]
    #print("optimizealpha, bw , FOM  = ", optalpha, optbw, maxFOM)
    if fomplot==True:
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)
        for bw in bwgrid:
            FOMlist = [FOM[(bw, al)] for al in alphagrid]
            if bw not in ['silverman', 'scott']:
                bw = float(bw) #bwchoice.astype(np.float) #for list
                ax.plot(alphagrid, FOMlist, label='{0:.3f}'.format(bw))
            else:
                ax.plot(alphagrid, FOMlist, label='{}'.format(bw))
        if optbw not in ['silverman', 'scott']:
            ax.plot(optalpha, maxFOM, 'ko', linewidth=10, label=r'$\alpha={0:.3f}, bw= {1:.3f}$'.format(optalpha, optbw))
        else:
            ax.plot(optalpha, maxFOM, 'ko', linewidth=10, label=r'$\alpha={0:.3f}, bw= {1}$'.format(optalpha, optbw))
        ax.set_xlabel(r'$\alpha$', fontsize=18)
        ax.set_ylabel(r'$FOM$', fontsize=18)
        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,1.25), ncol =6, fancybox=True, shadow=True, fontsize=8)
        #plt.ylim(maxFOM -5 , maxFOM +6)
        plt.savefig(opts.pathplot+"FOMfortwoDsourcecase.png", bbox_extra_artists=(lgd, ), bbox_inches='tight')
        plt.close()

    return optalpha, optbw, maxFOM

def get_Ndimkde(sample, xy_gridvalues, alphagrid, bwgrid, ret_kde=False, symmetry=False):
    """
    inputs: samplevalues, x_gridvalues, alphagrid, bwgrid 
    make sure order of alpha and bwds
    return: kdeval, optbw, optalpha
    make sure of order of outputs
    """
    if symmetry == True:
        swap_sample = np.vstack((sample[:, 1], sample[:,0])).T
        sample = np.vstack((sample, swap_sample))
    optalpha, optbw, maxFOM = Ndim_get_optimized_bw_kde_using_loocv(sample, alphagrid, bwgrid)
    print("alpha, bw = ", optalpha, optbw)
    #create grid valus for each reweight sample or use a fixed grid?
    if ret_kde:
        kdeobject, kdeval = N_dim_KDE_awkde(sample, xy_gridvalues, alp=optalpha, gl_bandwidth=optbw, ret_kde=True)
        return kdeobject, kdeval, optbw, optalpha
    kdeval = N_dim_KDE_awkde(sample, xy_gridvalues, alp=optalpha, gl_bandwidth=optbw)
    return kdeval, optbw, optalpha


##### Extra function if needed
def two_fold_crossvalidation(sample, bwchoice, alphachoice, n=5):
    """
    inputs:
      samples: training set
      bwchoice: choices of bwd for kde
      alphachoice: choice of kde parameter [0.1, 1]

    shuffle data into two equal samples
    train in half and test on other half
    in kde. 
    try this 5 times and take average 
    of those np.mean(combined 5 results)
    fom = np.log(average)
    """
    #randomly select data    
    random_data = np.random.choice(sample, len(sample))
    if bwchoice not in ['silverman', 'scott']:
        bwchoice = float(bwchoice)
    fomlist =[]
    for i in range(int(n)):
        #split data into two subset of equal size [or almost equal (n+1, n) parts]
        x_train, x_eval = np.array_split(random_data, 10)
        y = kde_awkde(x_train, x_eval, alp=alphachoice, gl_bandwidth=bwchoice)
        fomlist.append(np.log(y))
    return np.mean(fomlist)


def get_optimized_bw_kde_using_twofold(traindata, testdata, bwgrid, alphagrid):
    """ 
    Given grid of bw_choices 
    use two-fold cross_validation to get opt_bw
    figure of merit from two_fold_cv function for each choice
    return optimal param: optbw , optalpha
    """
    FOM= {}
    for gbw in bwgrid:
        for alphavals in alphagrid:
            FOM[(gbw, alphavals)] = two_fold_crossvalidation(traindata, gbw, alphavals)
    optval = max(FOM.items(), key=operator.itemgetter(1))[0]
    optbw, optalpha  = optval[0], optval[1]
    #get kde using optimized alpha and optimized bw
    kde_result = kde_awkde(traindata, testdata, alp=optalpha, gl_bandwidth=optbw)
    return optbw, optalpha, kde_result


def get_delta_from_KDE_variance(traindata, testdata, optbw, optalpha, err=False):
    """
    get global_sigma  from kde code and coeff
    from coeff of Kernel we will compute 95 percentile
    confidence interval
    there were changes in awkde codes
    #see https://github.com/JAMSADIQ/awkde/blob/master/awkde/awkde.py
    and
    #https://github.com/JAMSADIQ/awkde/blob/master/cpp/backend.cpp
    """
    kde = GaussianKDE(glob_bw=optbw, alpha=optalpha, diag_cov=True)
    kde.fit(traindata[:, np.newaxis])
    global_sigma_val = kde.get_global_sigma(testdata[:, np.newaxis])
    print("global_sigma_val  = ", global_sigma_val)
    #to get 95 percentile confidence interval
    if err == True:
        kde_result = kde_awkde(traindata, testdata, alp=optalpha, gl_bandwidth=optbw)
        coeff = kde.predict2(testdata[:, np.newaxis])
        sqcoeff = coeff**2
        cjmean = np.zeros(len(kde_result))
        cjsqmean =  np.zeros(len(kde_result))
        for k in range(len(kde_result)):
            cjmean[k] = sum(coeff[k])/len(coeff[k])
            cjsqmean[k] = sum(sqcoeff[k])/len(sqcoeff[k])
        sigmaarray = np.sqrt(cjsqmean - cjmean**2)
        PN = st.norm.ppf(.95) #1.28155156554466004
        Ndetection = len(traindata)
        error_estimate =  kde_result + (PN) * sigmaarray / np.sqrt(Ndetection)
        return global_sigma_val, error_estimate
    else:
        return global_sigma_val
