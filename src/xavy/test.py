import numpy as np
import pandas as pd


def test_assert(assertion, message='Assertion failed', verbose=True):
    """
    Test if an assertion is True, printing
    a message if it is not.
    
    Parameters
    ----------
    assertion : bool
        The test made, which can be True
        (the expected result) or False.
    message : str
        Message to print out if `assertion`
        is False.
    verbose : bool
        Whether to print the `message` or 
        not.
    
    Returns
    -------
    result : bool
        Whether the assertion is True or
        False.
    """
    if not assertion:
        if verbose is True:
            print(message)
        return False
    
    return True


def direct_comparison(result, output, verbose=False, ith_test=''):
    """
    Check if `result` is equal to `output` directly:
    `result == output`. If `verbose` is True, print 
    this function's name.
    """
    if verbose is True:
        print('direct_comparison')

    test_assert(result == output, '{}: Function output "{}" does not match expected `output` "{}".'.format(ith_test, result, output))

    
def float_comparison(result, output, numerical=False, verbose=False, ith_test=''):
    """
    Check if `result` is equal to `output` directly:
    `result == output`. If `verbose` is True, print 
    this function's name.
    """
    if verbose is True:
        print('float_comparison')
    
    # Exact match:
    if not numerical: 
        test_assert(result == output, '{}: Function output "{}" does not match expected `output` "{}".'.format(ith_test, result, output))
    
    # Numerical comparison:
    else:
        test_assert(np.isclose(result, output), '{}: Function output "{}" is not close enough to expected `output` "{}".'.format(ith_test, result, output))


def listlike_comparison(result, output, numerical=False, verbose=False, ith_test=''):
    """
    Check if type and elements of `result` and 
    `output` are the same. 
    
    If `numerical` is True, use `np.isclose()` 
    to avoid round-off errors.
    
    If `verbose` is True, print this function's name.
    """
    if verbose is True:
        print('listlike_comparison')

    # Exact comparison:
    if not numerical:
        test_assert(result == output, '{}: Function output "{}" does not match expected `output` "{}".'.format(ith_test, result, output))

    # Numerical comparison:
    else:
        same_type = test_assert(type(result) == type(output), '{}: Function output type "{}" does not match `output` type "{}".'.format(ith_test, type(result), type(output)))
        if same_type is True:
            test_assert(np.isclose(np.array(result), np.array(output)).all(), '{}: Function output "{}" is not close enough to expected `output` "{}".'.format(ith_test, result, output))


def elwise_comparison(result, output, numerical=False, verbose=False, ith_test=''):
    """
    Check if type and elements of `result` and 
    `output` are the same, for array-like input. 
    
    If `numerical` is True, use `np.isclose()` 
    to avoid round-off errors.
    
    If `verbose` is True, print this function's name.
    """
    if verbose is True:
        print('elwise_comparison')
    
    # Check type:
    same_type = test_assert(type(result) == type(output), '{}: Function output type "{}" does not match `output` type "{}".'.format(ith_test, type(result), type(output)))
    
    # Check len:
    if same_type is True:
        same_len = test_assert(len(result) == len(output), '{}: Function output length "{}" does not match `output` length "{}".'.format(ith_test, len(result), len(output)))
        
        # Check index if any:
        if same_len is True:
            same_idx = True
            if type(output) in (pd.core.series.Series,):
                same_idx = test_assert((result.index == output.index).all(), '{}: Function output index "{}" do not match expected `output` index "{}".'.format(ith_test, result.index, output.index))
            
            # Check dtype, if any:
            if same_idx:
                same_dtype = True
                if type(output) in (pd.core.series.Series,):
                    same_dtype = test_assert(result.dtype == output.dtype, '{}: Function output dtype "{}" do not match expected `output` dtype "{}".'.format(ith_test, result.dtype, output.dtype))

                # Check values:
                if same_dtype:
                    if not numerical:
                        test_assert((result == output).all(), '{}: Function output "{}" does not match expected `output` "{}".'.format(ith_test, result, output))
                    else:
                        test_assert(np.isclose(result, output).all(), '{}: Function output "{}" is not close enough to expected `output` "{}".'.format(ith_test, result, output))
    

def dataframe_comparison(result, output, verbose=False, ith_test=''):
    """
    Check if type, columns, index and values of 
    `result` and `output` are the same, expecting
    two DataFrames as input. If `verbose` is True, 
    print this function's name.
    """
    
    if verbose is True:
        print('dataframe_comparison')
    
    # Check type:
    same_type = test_assert(type(result) == type(output), '{}: Function output type "{}" does not match `output` type "{}".'.format(ith_test, type(result), type(output)))
    
    # Check # cols:
    if same_type is True:
        same_ncols = test_assert(len(result.columns) == len(output.columns), 
                                 '{}: Function output number of columns "{}" does not match expected `output` number of columns "{}".'.format(ith_test, len(result.columns), len(output.columns)))
        # Check columns:
        if same_ncols is True:
            same_cols = test_assert((result.columns == output.columns).all(), '{}: Function output columns "{}" do not match expected `output` columns "{}".'.format(ith_test, result.columns, output.columns))
        
            # Check # rows:
            if same_cols is True:
                same_nrows = test_assert(len(result.index) == len(output.index), '{}: Function output number of rows "{}" does not match expected `output` number of rows "{}".'.format(ith_test, len(result.index), len(output.index)))
            
                # Check index:
                if same_nrows is True:
                    same_idx = test_assert((result.index == output.index).all(), '{}: Function output index "{}" do not match expected `output` index "{}".'.format(ith_test, result.index, output.index))

                    # Check values:
                    if same_idx is True:
                        (result == output) == ~(result.isnull() & output.isnull())
                        test_assert(((result == output) == ~(result.isnull() & output.isnull())).all().all(),
                                    '{}: Function output "{}" does not match expected `output` "{}".'.format(ith_test, result, output)) 

                        
def test_function(f, args=tuple(), output=None, kwargs=dict(), numerical=False, ith_test='', verbose=False):
    """
    Check if a function's output is the expected
    one.
    
    Parameters
    ----------
    f : callable
        Function to test.
    args : tuple
        Positional arguments for `f`.
    kwargs : dict
        Keyword arguments for `f`.
    output : anything
        Expected output of `f` given the inputs 
        above.
    numerical : bool
        If `numerical` is True, use `np.isclose()` 
        when possible to avoid round-off errors.
    ith_test : int or None
        The number of the test, to be printed out in
        case of failure.
    verbose : bool
        If True, print extra information about 
        the evaluation process besides the 
        failed tests.
    """
    
    # Get function's output:
    result = f(*args, **kwargs)
    
    # Direct comparison:
    if type(output) in (str, int, dict, bool):
        direct_comparison(result, output, verbose, ith_test)
    
    # Float comparison
    elif type(output) in (float,):
        float_comparison(result, output, numerical, verbose, ith_test)
    
    # Element-wise comparison:
    elif type(output) in (tuple, list, set):
        listlike_comparison(result, output, numerical, verbose, ith_test)

    # Built-in element-wise comparison:
    elif type(output) in (np.ndarray, pd.core.series.Series):
        elwise_comparison(result, output, numerical, verbose, ith_test)
    
    # DataFrame comparison:
    elif type(output) in (pd.core.frame.DataFrame,):
        dataframe_comparison(result, output, verbose, ith_test)
        
    else:
        print('Unknown output type.')

        
def multi_test_function(f, parameters, outputs, numerical=False, verbose=False):
    """
    Run multiple tests on function `f` by using
    the provided input parameters and comparing
    the function's output with an expected 
    output.
    
    Parameters
    ----------
    f : callable
        Function to be tested.
    parameters : list of tuples and dicts
        Each tuple or dict in the list are 
        the positional or keyword `f` 
        parameters, respectively, to be used as 
        input for a test.
    outputs : list or non-iterable
        The expected outputs from `f` given 
        the inputs `parameters`. It can be one 
        output for each input or a single 
        common output for all inputs.
    numerical : bool or list of bool
        For each test, specifies if the 
        comparison between each output and 
        expected output is to be made 
        tolerating float round-off errors or
        not.
    verbose : bool or list of bool
        Whether to print extra messages for 
        each test or not.
    """
    
    # Security checks:
    assert type(parameters) is list, '`parameters` must be a list of inputs.'
    
    # Standardize input:
    if type(outputs) is not list:
        outputs = len(parameters) * [outputs]
    if type(numerical) is bool:
        numerical = len(outputs) * [numerical]
    if type(verbose) is bool:
        verbose = len(outputs) * [verbose]
    
    # Security checks:
    assert len(parameters) == len(outputs), '`parameters` have len {} while `outputs` have len {}. They should be the same.'.format(len(parameters), len(outputs))
    assert len(parameters) == len(numerical), '`parameters` have len {} while `numerical` have len {}. They should be the same.'.format(len(parameters), len(numerical))
    assert len(parameters) == len(verbose), '`parameters` have len {} while `verbose` have len {}. They should be the same.'.format(len(parameters), len(verbose))
     
    # Loop over test cases:
    tests = range(1, len(outputs) + 1)
    for i, params, output, num, verb in zip(tests, parameters, outputs, numerical, verbose):
        
        # Positional input:
        if type(params) in (tuple,):
            test_function(f, params, output, numerical=num, verbose=verb, ith_test=i)
        
        # Keyword input:
        elif type(params) in (dict,):
            test_function(f, kwargs=params, output=output, numerical=num, verbose=verb, ith_test=i)
        
        else:
            raise Exception('Unknown `parameters` element type "{}".'.format(type()))
    
