from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
from sklearn.decomposition import NMF
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

def rescaleLatents( latent ):
  """
  The latents aren't actually
  negative by default, so we rescale
  them so they are
  """
  return latent / 30 + 1

def unscaleLatents ( latent ):
  return ( latent - 1 ) * 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Init out vae
vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device)
vae.eval()

# Disable grad so that inference runs faster
torch.set_grad_enabled(False)

video = cv2.VideoCapture("PikePlace.mp4")

# Pull 1000 frames; we can average
# over these
frames = []
for i in range(1000):
  frames.append(video.read()[1])

frame = frames[0]
frame = torch.tensor(frame).float().permute(2, 0, 1).unsqueeze(0) / 255
frame = frame.to(device)

vaeMean = vae.encode(frame).latent_dist.mode()[0]

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

  A = A.numpy(force=True) if type(A) == torch.Tensor else A
  W = W.numpy(force=True) if type(W) == torch.Tensor else W
  H = H.numpy(force=True) if type(H) == torch.Tensor else H

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

vaeMean = vaeMean.reshape(-1, vaeMean.shape[-1])
print(f"VAE Mean shape: {vaeMean.shape}")

# Before doing anything else, let's visualize how final errors look like
# for each method, as a function of compute
MIN_COMPONENTS = 15
MAX_COMPONENTS = 200

nmf_norms_cd_randinit = [ nmfWrapper ( rescaleLatents(vaeMean), i, method="cd", init="random")[0][-1] for i in range(MIN_COMPONENTS, MAX_COMPONENTS, 20) ]
print("REALLYDONE\n" * 100 )
nmf_norms_cd_svdinit = [ nmfWrapper ( rescaleLatents(vaeMean), i, method="cd", init="nndsvd")[0][-1] for i in range(MIN_COMPONENTS, MAX_COMPONENTS, 20) ]
print("REALLYDONE\n" * 100 )
nmf_norms_mu_randinit = [ nmfWrapper ( rescaleLatents(vaeMean), i, method="mu", init="random")[0][-1] for i in range(MIN_COMPONENTS, MAX_COMPONENTS, 20) ]
print("REALLYDONE\n" * 100 )
nmf_norms_mu_svdinit = [ nmfWrapper ( rescaleLatents(vaeMean), i, method="mu", init="nndsvd")[0][-1] for i in range(MIN_COMPONENTS, MAX_COMPONENTS, 20) ]
print("REALLYDONE\n" * 100 )

# Plot all on the same plot, and dump to disk
components = list(range(MIN_COMPONENTS, MAX_COMPONENTS, 20))

plt.scatter(components, nmf_norms_cd_randinit, label="FAST HALS, Random Init")
plt.scatter(components, nmf_norms_cd_svdinit, label="FAST HALS, SVD Init")
plt.scatter(components, nmf_norms_mu_randinit, label="MU, Random Init")
plt.scatter(components, nmf_norms_mu_svdinit, label="MU, SVD Init")
plt.xlabel("Number of Components")
plt.ylabel("Frobenius Norm after 500 iters")
plt.title("Frobenius Norm vs. Number of Components, 500 iters max")
plt.legend()
plt.savefig("frobeniusNorms.png")

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
