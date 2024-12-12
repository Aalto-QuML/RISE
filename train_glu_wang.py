import torch
from simulators.oup import oup
import sbibm
from networks.summary_nets import OUPSummary, GL
from utils.get_nn_models import *
from inference.snpe.snpe_c import SNPE_C as SNPE
from inference.base import *
from utils.torchutils import *
from utils.metrics import RMSE
import pickle
import os
import argparse
import utils.metrics as metrics
import random
from sbibm.metrics import c2st


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def MMD_unweighted(x, y, lengthscale):
    """ Approximates the squared MMD between samples x_i ~ P and y_i ~ Q
    """

    m = x.shape[0]
    n = y.shape[0]

    z = torch.cat((x, y), dim=0)

    K = kernel_matrix(z, z, lengthscale)

    kxx = K[0:m, 0:m]
    kyy = K[m:(m + n), m:(m + n)]
    kxy = K[0:m, m:(m + n)]

    return (1 / m ** 2) * torch.sum(kxx) - (2 / (m * n)) * torch.sum(kxy) + (1 / n ** 2) * torch.sum(kyy)
    # return (1 / m ** 2) * torch.sum(kxx) - (2 / (m * n)) * torch.sum(kxy)

def median_heuristic(y):
    a = torch.cdist(y, y)**2
    return torch.sqrt(torch.median(a / 2))


def kernel_matrix(x, y, l):
    d = torch.cdist(x, y)**2

    kernel = torch.exp(-(1 / (2 * l ** 2)) * d)

    return kernel


def create_mask(data,degree):
        n_context = int(10 * degree)
        n_total = 10
        MASK = torch.ones((data.shape[0],data.shape[1]))
        for batch in range(data.shape[0]):
            ids = random.sample(range(n_total), n_context)
            for idx in ids:
                MASK[batch,idx] = 0
        return MASK

def sample_posteriors(posterior, obs, num):
    return posterior.sample((num,), x=obs, show_progress_bars=False)
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", type=float, default=0.1, help="degree of missingness")
    parser.add_argument("--type", type=str, default='mcar')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #set_seed(4)
    task = sbibm.get_task('gaussian_linear_uniform')
    prior = task.prior_dist
    simulator = task.get_simulator()
    post_samples_final = task.get_reference_posterior_samples(num_observation=1)
    post_samples_final = post_samples_final[:1000]

    prior = prior = Uniform(task.prior_params['low'].to(device),task.prior_params['high'].to(device))

    sum_net = GL(input_size=1, hidden_dim=4).to(device)
    neural_posterior = posterior_nn(
            model="maf",
            embedding_net=sum_net,
            hidden_features=20,
            num_transforms=3)

    inference = SNPE(prior=prior, density_estimator=neural_posterior,types ='glu',degree=0,missing=args.type, device='cuda')
    theta = torch.tensor(np.load("missing_data/glu_theta_1000.npy")).to(device)
    x = torch.tensor(np.load("missing_data/glu_x_1000.npy")).to(device)

    mask = create_mask(x,args.degree)
    #### Augmenting the binary indicator mask and data together
    x = torch.cat([x*mask.to(device),mask.to(device)],dim=-1)

    density_estimator = inference.append_simulations(theta, x.unsqueeze(1)).train(
            distance='none', x_obs=None, beta=0)
    torch.save(density_estimator, "test/density_estimator_glu.pkl")
    torch.save(sum_net, "test/sum_net_glu.pkl")
    posterior = inference.build_posterior(density_estimator)
    with open("test/posterior_glu.pkl", "wb") as handle:
            pickle.dump(posterior, handle)


    n_samples = 1000
    theta_gt = torch.tensor(np.load(f"missing_data/glu_theta_obs.npy"))
    obs_sample = torch.tensor(np.load(f"missing_data/glu_obs_zero_"+ str(int(args.degree*100))+".npy")).to(device)
    mask_sample = torch.tensor(np.load(f"missing_data/glu_obs_mask_"+ str(int(args.degree*100))+".npy")).to(device)

    fully_obs = torch.cat([obs_sample,mask_sample],dim=-1)
    
    lengthscale = median_heuristic(post_samples_final.cpu())
    n_sim = 100
    rmse_zero_npe = np.zeros(n_sim)
    mmd_zero_npe = np.zeros(n_sim)
    for i in range(0, n_sim):
            post_samples = sample_posteriors(posterior, fully_obs.unsqueeze(1), n_samples)
            rmse_zero_npe[i] = torch.sqrt(((post_samples.mean(dim=0).detach().cpu()-theta_gt.cpu())**2).mean()).item()
            mmd_zero_npe[i] = MMD_unweighted(post_samples.detach().cpu(), post_samples_final.cpu(), lengthscale)
    
    print(f" RMSE mean={np.mean(rmse_zero_npe)}, MMD mean={np.mean(mmd_zero_npe)}")