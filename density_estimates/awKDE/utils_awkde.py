"""
This is a module for functions related to awkde code
"""
from awkde import GaussianKDE
import numpy as np
import operator
import scipy.stats as st

################################################
def kde_awkde(x, x_grid, global_bandwidth='silverman', alpha=0.5, ret_kde=False):
    """
    Kernel Density Estimation with awkde.

    Parameters:
    x (array-like): Training data of shape (n_samples, n_features).
                    For the one-dimensional case, use x[:, np.newaxis].
    x_grid (array-like): Testing data, must be of shape (n_samples, n_features).
    global_bandwidth (str or float, optional): Global bandwidth for KDE. 
        Default is 'silverman'.
    alpha (float, optional): Smoothing factor for local bandwidth. 
        Default is 0.5.
    ret_kde (bool, optional): If True, the function will return the KDE object
        along with estimated KDE values.
    
    Returns:
    array or tuple: Estimated KDE values or a tuple (kde, y) if ret_kde is True.
    
    Raises:
    ValueError: If the input data does not have the required shape.
    """
    if len(x.shape) == 1:
        # Transform to a 2D array with shape (len, 1)
        x = x[:, np.newaxis]
    if len(x_grid.shape) == 1:
        x_grid = x_grid[:, np.newaxis]

    if len(x.shape) !=2:
        raise ValueError("data must have shape (n_samples, n_features).")
    
    kde = GaussianKDE(glob_bw=global_bandwidth, alpha=alpha, diag_cov=True)
    #fit the kde
    kde.fit(x)
    #evaluate the kde
    y = kde.predict(x_grid)

    if ret_kde == True:
        return kde, y
    return y


############# Cross validations log likelihood for figure of merit ################
def loocv_awkde(sample, bwchoice, alphachoice):
    """
    Perform Leave-One-Out Cross Validation (LOOCV) for Kernel Density Estimation (KDE) using awkde.

    This function calculates the figure of merit (FoM) as 
        the sum of the log likelihoods of the left-out samples.
    It uses Leave-One-Out cross validation to split the data, 
    trains the KDE on the training subset, and evaluates it on the left-out sample.

    Parameters:
    sample (array-like): The data samples to be used for KDE, of shape (n_samples, n_features).
    bwchoice (str or float): The choice of bandwidth for KDE. Accepts 'silverman', 'scott', or a float value.
    alphachoice (float): The smoothing factor for local bandwidth in KDE.

    Returns:
    float: The sum of the log likelihoods of the left-out samples (figure of merit).

    Raises:
    ValueError: If the bandwidth choice is neither 'silverman', 'scott', nor a valid float value.

    Example:
    >>> sample = np.random.normal(size=(100, 2))
    >>> bwchoice = 'silverman'
    >>> alphachoice = 0.5
    >>> fom = loocv_awkde(sample, bwchoice, alphachoice)
    >>> print(fom)
    """
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    if bwchoice not in ['silverman', 'scott']:
        bwchoice = float(bwchoice) #bwchoice.astype(np.float) #for list
    fom = 0.0
    for train_index, test_index in loo.split(sample):
        train_data, test_data = sample[train_index], sample[test_index]
        y = kde_awkde(train_data, test_data, global_bandwidth=bwchoice, alpha=alphachoice, ret_kde=False)
        fom += np.log(y)

    return fom

def kfold_cv_awkde(sample, bwchoice, alphachoice, n_splits=5):
    """
    Perform K-Fold Cross Validation for Kernel Density Estimation (KDE) using awkde.

    This function calculates the figure of merit (FoM) as
        the total sum of the sum oflog likelihoods of the left-out samples
        in k-fold split.
    It uses K-Fold cross validation to split the data, 
    trains the KDE on the training subset, and evaluates it on the testing samples.

    Parameters:
    sample (array-like): The data samples to be used for KDE, of shape (n_samples, n_features).
    bwchoice (str or float): The choice of bandwidth for KDE. Accepts 'silverman', 'scott', or a float value.
    alphachoice (float): The smoothing factor for local bandwidth in KDE.
    n_splits (int, optional): The number of splits for K-Fold cross validation. Default is 5.

    Returns:
    float: Total sum of sums of the log likelihoods of the left-out samples (figure of merit) in k-fold splits.

    Raises:
    ValueError: If the bandwidth choice is neither 'silverman', 'scott', nor a valid float value.

    Example:
    >>> sample = np.random.normal(size=(100, 2))
    >>> bwchoice = 'silverman'
    >>> alphachoice = 0.5
    >>> fom = kfold_cv_awkde(sample, bwchoice, alphachoice, n_splits=5)
    >>> print(fom)
    """
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fom = []
    if bwchoice not in ['silverman', 'scott']:
        bwchoice = float(bwchoice) 
    for train_index, test_index in kf.split(sample):
        train_data, test_data = sample[train_index], sample[test_index]
        nonlog_kde_val = kde_awkde(train_data, test_data, global_bandwidth=bwchoice, alpha=alphachoice, ret_kde=False)
        ### error if prob is 0 or negative raise error for it
        if np.any(nonlog_kde_val <= 0):
            raise ValueError("KDE estimate contains non-positive values, log will not work.")
        #contains_neg_or_zero = np.where(nonlog_kde_val <= 0) [0]
        #nonlog_kde_val[contains_neg_or_zero] = 1.0 #is this ok because log1=0?
        #print("zero in estimate = ", contains_neg_or_zero) # Output: True
        log_kde_val = np.log(nonlog_kde_val)
        fom.append(log_kde_val.sum())
    return sum(fom)


def optimize_parameters(sample, bandwidth_options, alpha_options, method='loo_cv', fom_plot_name=None):
    """
    return opt bandwidth, opt alpha and fom value
    """
    best_params = {'bandwidth': None, 'alpha': None}
    # Perform grid search
    fom_grid = {}
    for bandwidth in bandwidth_options:
        for alpha in alpha_options:
            if method == 'loo_cv':
                fom_grid[(bandwidth, alpha)] = loocv_awkde(sample, bandwidth, alpha)
            else:
                fom_grid[(bandwidth, alpha)] = kfold_cv_awkde(sample, bandwidth, alpha)
            print("bandwidth, alpha, score", bandwidth, alpha, fom_grid[(bandwidth, alpha)] )
    optval = max(fom_grid, key=lambda k: fom_grid[k])
    optbw, optalpha = optval[0], optval[1]
    best_score = fom_grid[(optbw, optalpha)]
    print("optbw, optalpha, fom_score", optbw, optalpha, best_score)
    best_params = {'bandwidth': optbw, 'alpha': optalpha}

    #plot
    if fom_plot_name is not None:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)
        for bw in bandwidth_options:
            fom_list = [fom_grid[(bw, al)] for al in alpha_options]
            if bw not in ['silverman', 'scott']:
                ax.plot(alpha_options, fom_list, label='{0:.3f}'.format(float(bw)))
            else:
                ax.plot(alpha_options, fom_list, label='{}'.format(bw))
        if optbw not in ['silverman', 'scott']:
            ax.plot(optalpha, best_score, 'ko', linewidth=10, label=r'$\alpha={0:.3f}, bw= {1:.3f}$'.format(optalpha, float(optbw)))
        else:
            ax.plot(optalpha, best_score, 'ko', linewidth=10, label=r'$\alpha={0:.3f}, bw= {1}$'.format(optalpha, optbw))
        ax.set_xlabel(r'$\alpha$', fontsize=18)
        ax.set_ylabel(r'$FOM$', fontsize=18)
        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=6, fancybox=True, shadow=True, fontsize=8)
        plt.tight_layout()
        plt.savefig(fom_plot_name+".png", bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close()

    return  optbw, optalpha, best_score



######################## Specific for 2D case (if m1-m2 can swap for symmetry) ################
def kde_twoD_with_do_optimize(sample, xy_gridvalues, bwgrid, alphagrid, ret_kde=False, symmetry=False, optimize_method='loocv'):
    """
    inputs: samplevalues, x_gridvalues, alphagrid, bwgrid 
    make sure order of alpha and bwds
    return: kdeval, optbw, optalpha
    make sure of order of outputs
    """
    if symmetry == True:
        swap_sample = np.vstack((sample[:, 1], sample[:,0])).T
        sample = np.vstack((sample, swap_sample))
    #do optimization over grid of values
    optbw, optalpha, maxFOM = optimize_parameters(sample, bwgrid, alphagrid, method=optimize_method)
    print("alpha, bw = ", optalpha, optbw)
    #create grid valus for each reweight sample or use a fixed grid?
    if ret_kde:
        kdeobject, kdeval = kde_awkde(sample, xy_gridvalues, global_bandwidth=optbw , alpha=optalpha, ret_kde=True)
        return kdeobject, kdeval, optbw, optalpha
    kdeval = kde_awkde(sample, xy_gridvalues, global_bandwidth=optbw , alpha=optalpha)
    return kdeval, optbw, optalpha





#################### Extra work for error estimate #############################
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
        kde_result = kde_awkde(traindata, testdata, global_bandwidth=optbw, alpha=optalpha)
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
