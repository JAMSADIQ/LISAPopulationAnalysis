"""
This is a module for functions related to awkde code
"""

import numpy as np
import operator
import scipy.stats as st  #we will use gaussian kde 
import matplotlib.pyplot as plt
import sklearn
################################################
def kde_scipy(x, x_grid, global_bandwidth='silverman', weights=None, ret_kde=False):
    """
    Kernel Density Estimation with scipy

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
    #kde = GaussianKDE(bandwidth=global_bandwidth, kernel='gaussian')
    #fit the kde
    if weights is not None:
        kde = st.gaussian_kde(x.T, bw_method=global_bandwidth , weights=weights)
        #kde.fit(x, sample_weight=weights)
    else:
        #kde.fit(x)
        kde = st.gaussian_kde(x.T, bw_method=global_bandwidth)
    #evaluate the kde
    #logy = kde.score_samples(x_grid)
    #y = np.exp(logy)
    y = kde(x_grid.T)
    if ret_kde == True:
        return kde, y
    return y


############# Cross validations log likelihood for figure of merit ################
def loocv_scipykde(sample, bwchoice, weight_vals):
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
        weight_train =  weight_vals[train_index]
        y = kde_scipy(train_data, test_data, global_bandwidth=bwchoice, weights=weight_train, ret_kde=False)
        if y == 0  or y <= 0:
            print(test_data, y)
        fom += np.log(y)

    return fom

def kfold_cv_scipykde(sample, bwchoice, weight_vals, n_splits=5):
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
    #if bwchoice not in ['silverman', 'scott']:
    #    bwchoice = float(bwchoice) 
    for train_index, test_index in kf.split(sample):
        train_data, test_data = sample[train_index], sample[test_index]
        weight_train = weight_vals[train_index]
        #print(test_data)
        nonlog_kde_val = kde_scipy(train_data, test_data, global_bandwidth=bwchoice, weights=weight_train, ret_kde=False)
        log_kde_val = np.log(nonlog_kde_val)
        fom.append(log_kde_val.sum())
    return sum(fom)


def optimize_parameters(sample, bandwidth_options, weight_vals, method='loo_cv', fom_plot_name=None):
    """
    return opt bandwidth, opt alpha and fom value
    """
    best_params = {'bandwidth': None, 'alpha': None}
    # Perform grid search
    fom_grid = {}
    print(method)
    for bandwidth in bandwidth_options:
        if method == 'loo_cv':
            fom_grid[(bandwidth)] = loocv_scipykde(sample, bandwidth, weight_vals)
        else:
            fom_grid[(bandwidth)] = kfold_cv_scipykde(sample, bandwidth, weight_vals)
            print("bandwidth, score", bandwidth, fom_grid[(bandwidth)] )
    optval = max(fom_grid, key=lambda k: fom_grid[k])
    optbw = optval
    best_score = fom_grid[(optbw)]
    #print("optbw, optalpha, fom_score", optbw, optalpha, best_score)
    best_params = {'bandwidth': optbw}

    #plot
    if fom_plot_name is not None:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)
        if bw not in ['silverman', 'scott']:
            ax.plot(bandwidth_options, fom_list)
        else:
            ax.plot(bandwidth_options, fom_list)
        ax.plot(optalpha, best_score, 'ko', linewidth=10, label=r'$ bw= {0:.3f}$'.format(float(optbw)))
        ax.set_xlabel(r'$bw$', fontsize=18)
        ax.set_ylabel(r'$FOM$', fontsize=18)
        handles, labels = ax.get_legend_handles_labels()
        plt.tight_layout()
        plt.savefig(fom_plot_name+".png", bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close()

    return  optbw, best_score



######################## Specific for 2D case (if m1-m2 can swap for symmetry) ################
def kde_twoD_with_do_optimize(sample, xy_gridvalues, bwgrid, weights=None, ret_kde=False, optimize_method='loo_cv'):
    """
    inputs: samplevalues, x_gridvalues, alphagrid, bwgrid 
    make sure order of alpha and bwds
    return: kdeval, optbw, optalpha
    make sure of order of outputs
    """
    print(optimize_method)
    optbw, maxFOM = optimize_parameters(sample, bwgrid, weights, method=optimize_method)
    optbw = float(optbw)
    #print("alpha, bw = ", optalpha, optbw)
    #create grid valus for each reweight sample or use a fixed grid?
    if ret_kde:
        kdeobject, kdeval = kde_scipy(sample, xy_gridvalues, global_bandwidth=optbw , weights=weights, ret_kde=True)
        return kdeobject, kdeval, optbw
    kdeval = kde_scipy(sample, xy_gridvalues, global_bandwidth=optbw , weights=weights)
    return kdeval, optbw

#############test the code with data#########################


##np.random.seed(42)
##mean = [0, 0]
##cov = [[1, 0.5], [0.5, 1]]  # diagonal covariance
##n_samples = 500
##
### Generate samples
##data = np.random.multivariate_normal(mean, cov, n_samples)
##
### Step 2: Create a pattern (e.g., higher density in a specific region)
### Let's create a denser region around (1,1)
##pattern_center = [1, 1]
##pattern_cov = [[0.1, 0], [0, 0.1]]
##n_pattern_samples = 100
##
### Generate pattern samples
##pattern_data = np.random.multivariate_normal(pattern_center, pattern_cov, n_pattern_samples)
##data_with_pattern = np.vstack([data, pattern_data])
##print(data_with_pattern.shape)
##
###quit()
### Step 3: Assign weights to the samples based on the pattern
##weights = np.ones(n_samples + n_pattern_samples)
##weights[-n_pattern_samples:] = 10
##
##
### Create grid for evaluation
##x_grid = np.linspace(-3, 3, 100)
##y_grid = np.linspace(-3, 3, 100)
##X, Y = np.meshgrid(x_grid, y_grid)
##xy_sample = np.vstack([X.ravel(), Y.ravel()]).T
##
##
##bwgrid = np.logspace(-1.5, 0, 10)
##density_weighted, bwopt = kde_twoD_with_do_optimize(data_with_pattern, xy_sample, weights, bwgrid, optimize_method='kfold_cv')
##density_weighted = density_weighted.reshape(X.shape)
##print(f"best bw = , {bwopt}")
##plt.contour(X, Y, density_weighted, levels=20, cmap='viridis')
##plt.scatter(data_with_pattern[:, 0], data_with_pattern[:, 1], c='blue', s=2, label='Data points')
##plt.scatter(pattern_data[:, 0], pattern_data[:, 1], c='red', s=10, label='Pattern points')
##
### Place title inside the plot
##plt.text(0.5, 0.95, 'KDE with weights', fontsize=20, ha='center')
##plt.legend(loc=4, fontsize=16)
##plt.show()
