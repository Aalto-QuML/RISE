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
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", type=float, default=0.1, help="degree of missingness")
    parser.add_argument("--type", type=str, default='mcar')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #set_seed(4)
    task = sbibm.get_task('gaussian_linear_uniform')
    prior = task.get_prior()
    simulator = task.get_simulator()
    post_samples_final = task.get_reference_posterior_samples(num_observation=1)
    post_samples_final = post_samples_final[:1000]
    theta = prior(num_samples=1000)
    x = simulator(theta)

    prior =  Uniform(task.prior_params['low'].to(device),task.prior_params['high'].to(device))

    sum_net = GL(input_size=1, hidden_dim=4).to(device)
    neural_posterior = posterior_nn(
            model="maf",
            embedding_net=sum_net,
            hidden_features=20,
            num_transforms=3)

    inference = SNPE(prior=prior, density_estimator=neural_posterior,types ='glu',degree=args.degree,missing=args.type, device='cuda')

    density_estimator,missing_model = inference.append_simulations(theta, x.unsqueeze(1)).train(
            distance='none', x_obs=None, beta=0)
    torch.save(density_estimator, "test/density_estimator_glu.pkl")
    torch.save(sum_net, "test/sum_net_glu.pkl")
    posterior = inference.build_posterior(density_estimator)
    with open("test/posterior_glu.pkl", "wb") as handle:
            pickle.dump(posterior, handle)
