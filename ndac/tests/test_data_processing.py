import numpy as np
import pandas as pd

import ndac


def test_quantile_classiffy():
    # example pandas series
    metric = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    sequence = pd.Series(['aa', 'bb', 'cc', 'dd', 'ee', 'ff', 'gg', 'hh', 'ii', 'jj', 'kk', 'll'])
    
    #call the function on the test series
    dataframe, hist = ndac.quantile_classify(metric, sequence)
    
    #verify that half of the data was removed
    np.testing.assert_almost_equal(len(dataframe), len(metric)/2, err_msg='quantiles not divided properly')
    
    return