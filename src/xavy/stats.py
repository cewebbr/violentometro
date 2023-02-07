import numpy as np
import pandas as pd
import scipy.stats as stats


def compute_correlation(a, b):
    """
    Return the Pearson correlation between 
    `a` (array-like) and `b` (array-like).
    """
    return np.corrcoef(a, b)[0, 1]


def bootstrap_correlation(series_a, series_b, n_trials=10000):
    """
    Return `n_trials` (int) correlations (array) 
    computed from `series_a` (Series) and `series_b` 
    (Series) when the second series is scrambled.
    """
    
    assert len(series_a) == len(series_b)
    n_instances = len(series_b)
    
    corrs = np.array([compute_correlation(series_a, series_b.sample(n_instances)) for _ in range(n_trials)])
    
    return corrs


def p_value(trials, threshold):
    """
    Compute the right-side p-value by counting the 
    fraction of random `trials` (numerical array-like) 
    that are greater than `threshold` (number).
    """
    pvalue = (trials > threshold).sum() / len(trials)
    return pvalue


def shuffle_data(data, random_state=None):
    """
    Shuffle the data while maintaining 
    the same index.
    
    Parameters
    ----------
    data : DataFrame or Series
        Data to be shuffled.
    random_state : int or None
        Seed for the number generator.
        Set to `None` for a random seed.
    
    Returns
    -------
    shuffled : DataFrame or Series
        Same as `data` but with shuffled
        entries, keeping the index order 
        exactly the same.
    """
    # Shuffle entries:
    shuffled = data.sample(len(data), random_state=random_state)
    # Keep the same index as before:
    shuffled.index = data.index
    
    return shuffled


def break_column_correlations(df, random_state=None):
    """
    Randomly shuffles the rows of the entries in each `df` column.
    
    Parameters
    ----------
    df : DataFrame
        Data with columns data to be shuffled.
    random_state : int or None
        Seed for the random number generator.
        
    Returns
    -------
    shuffled_df : DataFrame
        New DataFrame with the same index as `df` (and in the same 
        order) but with the entries in each column shuffled. Every
        column is shuffled independently.
    """
    
    shuffled_df = pd.DataFrame()
    
    for i, col in enumerate(df.columns):
        seed = None if random_state is None else random_state + i
        shuffled_df[col] = shuffle_data(df[col], random_state=seed)
        
    return shuffled_df


def frac_by_categories(df, dimensions=None):
    """
    Compute the fraction of rows in `df` (DataFrame) that falls 
    inside subgroups formed by categories under `dimensions` 
    (list of str, `df` column names). If `dimensions` is None,
    use all `df` columns.
    
    Returns a Series with `dimensions` as index.
    """
    
    if dimensions is None:
        dimensions = list(df.columns)
    
    n    = len(df)
    freq = df.groupby(dimensions).size() / n
    
    return freq


def resample_stats(df, stat_func, n_samples=4000):
    """
    Shuffle the entries in each column in `df` (DataFrame) and 
    compute a statistic `stat_func` (callable) over it `n_samples` 
    (int) times.
    
    Returns a statistics DataFrame with one column per reshuffled 
    data.
    """
    
    series_dict = dict()
    for i in range(n_samples):
        series = stat_func(break_column_correlations(df))
        series_dict[i] = series
    
    sampled_df = pd.DataFrame(series_dict)
        
    return sampled_df


def count_class_votes(X, N_classes=None):
    """
    Given the classification given by multiple 
    annotators, return the number of times each
    class was selected.
    
    Parameters
    ----------
    X : 2D array
        Each row is an instance that was classified;
        each column refers to a different annotator;
        The values are the indices of the classes.
    N_classes : int or None
        Number of classes available for classifying
        the instances. If None, assume all available
        classes are present and that they are 
        numbered from 0 to `Nclasses - 1`.
    
    Returns
    -------
    nvotes : 2D array
        Each row is an instance that was classified;
        each column refers to a different class;
        The values are the number of votes given by 
        the annotators for assigning to that class.
    """
    
    # If number of classes was not provided, 
    # assume they are numbered from 0 to Nclasses - 1:
    if N_classes is None:
        N_classes = X.max() + 1

    nvotes = np.apply_along_axis(np.bincount, 1, X, minlength=N_classes)
    
    return nvotes


def fleiss_kappa(X, N_classes=None, n_votes_per_class=True):
    """
    Compute Fleiss' Kappa from a data matrix.
    
    Parameters
    ----------
    X : 2D array
        * If `n_votes_per_class` is True (default):        
          Each row is an instance that was classified;
          each column refers to a different class;
          The values are the number of votes given by 
          the annotators for assigning to that class.
        * If `n_votes_per_class` is False:
          Each row is an instance that was classified;
          each column refers to a different annotator;
          The values are the indices of the classes.
    N_classes : int or None
        Only used if `n_votes_per_class` is False.
        Number of classes available for classifying
        the instances. If None, assume all available
        classes are present and that they are 
        numbered from 0 to `Nclasses - 1`.
    n_votes_per_class : bool
        Specifies the type of input `X` (see above).
    
    Returns
    -------
    kappa : float
        Fleiss' Kappa measure of agreement between 
        annotators.
    """
    # Convert annotators choices to class votes if necessary:
    if n_votes_per_class is False:
        #print('Converting...')
        X = count_class_votes(X, N_classes)
    
    # Count the number of annotators:
    n = X.sum(axis=1)
    assert len(set(n)) == 1, 'Expecting the same total number os votes (i.e. annotators) for all instances.'
    n = np.mean(n)

    # Compute the fraction of pairs of annotators that agree with each other:
    Pi = (X * (X -1)).sum(axis=1) / (n * (n - 1))
    P = Pi.mean() 

    # Compute the agreement chance:
    pj = X.mean(axis=0) / n
    assert np.isclose(pj.sum(), 1.0), 'Sum of pjs should be 1.0'
    Pe = (pj**2).sum()

    # Compute Fleiss' Kappa:
    kappa = (P - Pe) / (1 - Pe)
    
    return kappa


def triang_binom_posterior(prior_mode, n_trials, n_success, dp=0.005):
    """
    Compute the posterior for the success probability for a binomial 
    distribution given the number of trials, the observed number of 
    successes and a triangular prior.
    
    Parameters
    ----------
    prior_mode : float
        The mode of the triangular distribution that describes the 
        prior probability for the success probability of a single trial.
    n_trials : int
        Number of trials of the independent binary experiment.
    n_success : int
        Number of times the independent experiment returned a positive 
        (success) result.
    dp : float
        Interval between point used to compute the posterior.
        
    Returns
    -------
    p : 1D array
        Values of the success probability for a single trial of the 
        experiment for which the posterior was computed.
    post : 1D array
        The values at `p` of the posterior probability distribution.
    """
    
    p = np.arange(0, 1, dp)
    # Compute posterior:
    post = stats.triang.pdf(p, prior_mode) * stats.binom.pmf(n_success, n_trials, p)
    # Normalize posterior:
    post = post / (post.sum() * dp)

    return p, post


