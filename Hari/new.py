import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import ConvergenceWarning
import warnings


warnings.simplefilter('ignore', ConvergenceWarning)


data = pd.read_csv('stock_data.csv', index_col=0)
data.fillna(data.mean(), inplace=True)  

def apply_nmf(data, n_components, init='random'):
    try:
        model = NMF(n_components=n_components, init=init, solver='mu', max_iter=1000, random_state=42)
        W = model.fit_transform(data)
        H = model.components_
        reconstruction = np.dot(W, H)
        error = np.sqrt(mean_squared_error(data, reconstruction))  
    except ValueError:
        error = np.nan  
    return error


components = range(1, 50)  


errors_random = [apply_nmf(data, n, init='random') for n in components]
errors_nndsvd = [apply_nmf(data, n, init='nndsvd') for n in components]


plt.figure(figsize=(10, 6))
plt.scatter(components, errors_random, label='Random Initialization', marker='o')
plt.scatter(components, errors_nndsvd, label='NNDSVD Initialization', marker='o')
plt.title('Frobenius Norm Error vs. Number of Components')
plt.xlabel('Number of Components')
plt.ylabel('Frobenius Norm Error')
plt.legend()
plt.show()
