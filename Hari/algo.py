from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import NMF
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning



# Load the CSV file into a DataFrame
dataframe = pd.read_csv('stock_data.csv', index_col=0)
dataframe.fillna(dataframe.mean(), inplace=True)
matrix = dataframe.values
print(matrix.shape)

@ignore_warnings(category=ConvergenceWarning)
def nmfWrapper(A, n_components : int, W : np.ndarray = None, H : np.ndarray = None, lib : str = "sklearn", method="hals", maxIters=500, init="random", minErrorChange : float = 1e-4):
  """
  A wrapper around the different kinds of NMF to allow
  easy testing with both hand-implemented and SKLEARN
  based NMF.
  """

  if lib != "sklearn":
    raise ValueError("Only sklearn is supported for now")
  
  method = "cd" if method == "hals" else "mu"
  init = "random" if init == "random" else "nndsvd"


  returnDict = {}

  # A list containing frobenius norms
  # of every iteration 
  norms = []

  # Also contain times
  times = []
  startTime = datetime.utcnow()

  # Setting tolerance to something super high
  # so we can manually exit when we want to, and know
  # when NMF succeeded
  nmf = NMF(n_components = n_components, init = init, max_iter = 1, tol=100, solver = method)

  # Get initial set points for W and H
  W = nmf.fit_transform ( A )
  H = nmf.components_

  nmf = NMF(n_components = n_components, init = "custom", max_iter = 1, tol=100, solver = method)

  for i in range(maxIters):
    W = W * 0 + nmf.fit_transform ( A, W = W, H = H )
    H = H * 0 + nmf.components_

    currError = np.linalg.norm(A - W @ H, ord=2)
    norms.append(currError)

    times.append((datetime.utcnow() - startTime).total_seconds())

    # If errors have changed by less than
    # our tolerance, break
    if len(norms) >= 2 and abs ( norms[-1] - norms[-2] ) < minErrorChange:
      break

  print("DONE")

  return norms, W, H, times

# Set this to True if
# we want to do EDA and see
# performance on all n-components
RUN_EDA = False

if RUN_EDA:
  # Before doing anything else, let's visualize how final errors look like
  # for each method, as a function of compute
  MIN_COMPONENTS = 15
  MAX_COMPONENTS = 99
  
  nmf_norms_cd_randinit = [ nmfWrapper ( matrix, i, method="cd", init="random")[0][-1] for i in range(MIN_COMPONENTS, MAX_COMPONENTS, 10) ]
  print("REALLYDONE\n" * 100 )
  nmf_norms_cd_svdinit = [ nmfWrapper ( matrix, i, method="cd", init="nndsvd")[0][-1] for i in range(MIN_COMPONENTS, MAX_COMPONENTS, 10) ]
  print("REALLYDONE\n" * 100 )
  nmf_norms_mu_randinit = [ nmfWrapper ( matrix, i, method="mu", init="random")[0][-1] for i in range(MIN_COMPONENTS, MAX_COMPONENTS, 10) ]
  print("REALLYDONE\n" * 100 )
  nmf_norms_mu_svdinit = [ nmfWrapper ( matrix, i, method="mu", init="nndsvd")[0][-1] for i in range(MIN_COMPONENTS, MAX_COMPONENTS, 10) ]
  print("REALLYDONE\n" * 100 )
  
  # Plot all on the same plot, and dump to disk
  components = list(range(MIN_COMPONENTS, MAX_COMPONENTS, 10))
  
  plt.scatter(components, nmf_norms_cd_randinit, label="FAST HALS, Random Init")
  plt.scatter(components, nmf_norms_cd_svdinit, label="FAST HALS, SVD Init")
  plt.scatter(components, nmf_norms_mu_randinit, label="MU, Random Init")
  plt.scatter(components, nmf_norms_mu_svdinit, label="MU, SVD Init")
  plt.xlabel("Number of Components")
  plt.ylabel("Frobenius Norm after 500 iters")
  plt.title("Frobenius Norm vs. Number of Components, 500 iters max")
  plt.legend()
  plt.savefig("frobeniusNorms.png")

# Set this to True if you want to see performance results
# on n=125 components
RUN_FINAL_ANALYSIS = True
if RUN_FINAL_ANALYSIS:
  n_comp = 40

  # Get me convergence props, and
  # time to run for each method
  cd_randinit_norms, _, _, cd_randinit_times = nmfWrapper ( matrix, n_comp, method="cd", init="random", maxIters=500)
  cd_svdinit_norms, _, _, cd_svdinit_times = nmfWrapper ( matrix, n_comp, method="cd", init="nndsvd", maxIters=500)
  mu_randinit_norms, _, _, mu_randinit_times = nmfWrapper ( matrix, n_comp, method="mu", init="random", maxIters=500)
  mu_svdinit_norms, _, _, mu_svdinit_times = nmfWrapper ( matrix, n_comp, method="mu", init="nndsvd", maxIters=500)

  # Scatter time to get to each iteration on
  # one subplot, and frobenius norm on another
  fig, axs = plt.subplots(2)
  axs[0].plot(range(len(cd_randinit_times)), cd_randinit_times, label="FAST HALS, Random Init")
  axs[0].plot(range(len(cd_svdinit_times)), cd_svdinit_times, label="FAST HALS, SVD Init")
  axs[0].plot(range(len(mu_randinit_times)), mu_randinit_times, label="MU, Random Init")
  axs[0].plot(range(len(mu_svdinit_times)), mu_svdinit_times, label="MU, SVD Init")

  axs[0].set_xlabel("Iteration")
  axs[0].set_ylabel("Time (seconds)")

  axs[1].plot(range(len(cd_randinit_norms)), cd_randinit_norms, label="FAST HALS, Random Init")
  axs[1].plot(range(len(cd_svdinit_norms)), cd_svdinit_norms, label="FAST HALS, SVD Init")
  axs[1].plot(range(len(mu_randinit_norms)), mu_randinit_norms, label="MU, Random Init")
  axs[1].plot(range(len(mu_svdinit_norms)), mu_svdinit_norms, label="MU, SVD Init")

  axs[1].set_xlabel("Iteration")
  axs[1].set_ylabel("Frobenius Norm")

  plt.legend()
  plt.savefig("finalAnalysis.png")
  

# # Iterate over n_components, going from n=10 to n=200,
# # in increments of 20
# nmfFrobNorms = []
# nmfFrobNorms_SVDInit = []
# components = []
# for comp in range ( 10, 100, 10 ):
#   vaeMean = rescaleLatents ( vaeMean )
#   print(vaeMean.shape)
#   nmf = NMF( init = "random" , n_components = comp, max_iter=500)
#   W = nmf.fit_transform(vaeMean.cpu().numpy())
#   H = nmf.components_
#   recon = W @ H
#   # Undo changes to both
#   vaeMean = unscaleLatents ( vaeMean )
#   recon = unscaleLatents ( recon )
#   nmfFrobNorms.append ( torch.linalg.norm(torch.tensor(vaeMean).cpu() - torch.tensor(recon), ord=2) )
#   components.append ( comp )
# 
#   # Do the same, but with the other init method
#   vaeMean = rescaleLatents ( vaeMean )
#   nmf = NMF( init = "nndsvd" , n_components = comp, max_iter=500)
#   W = nmf.fit_transform(vaeMean.cpu().numpy())
#   H = nmf.components_
#   recon = W @ H
# 
#   vaeMean = unscaleLatents ( vaeMean )
#   recon = unscaleLatents ( recon )
#   nmfFrobNorms_SVDInit.append ( torch.linalg.norm(torch.tensor(vaeMean).cpu() - torch.tensor(recon), ord=2) )
# 
# plt.scatter(components, nmfFrobNorms, label="Random Init")
# plt.scatter(components, nmfFrobNorms_SVDInit, label="SVD Init")
# plt.legend()
# plt.xlabel("Number of NMF Components")
# plt.ylabel("Frobenius Norm")
# plt.title("Frobenius Norm vs. Number of NMF Components, 500 iters max")
# # plt.savefig("frobeniusNorm.png")