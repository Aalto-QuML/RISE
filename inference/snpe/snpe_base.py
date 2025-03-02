# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.
import time
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Union
from warnings import warn
from simulators.ricker import ricker
import torch
from torch import Tensor, nn, ones, optim
from torch.distributions import Distribution
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils import data
from torch.utils.tensorboard.writer import SummaryWriter

import utils
from utils.metrics import *
from inference.base import NeuralInference, check_if_proposal_has_default_x
from inference.posteriors import (
    DirectPosterior,
    MCMCPosterior,
    RejectionPosterior,
    VIPosterior,
)
from inference.posteriors.base_posterior import NeuralPosterior
from inference.potentials import posterior_estimator_based_potential
from utils import (
    RestrictedPrior,
    check_estimator_arg,
    handle_invalid_x,
    test_posterior_net_for_multi_d_x,
    validate_theta_and_x,
    warn_if_zscoring_changes_data,
    warn_on_invalid_x,
    warn_on_invalid_x_for_snpec_leakage,
    x_shape_from_simulation,
    corruption
)
from utils.sbiutils import ImproperEmpirical, mask_sims_from_prior
from model_function import CNN_LNP, MLP_LNP,CNN_LNP_MNAR,MLP_LNP_MNAR
from HH_helper_functions import *


class PosteriorEstimator(NeuralInference, ABC):
    def __init__(
        self,
        prior: Optional[Distribution] = None,
        density_estimator: Union[str, Callable] = "maf",
        types: str = "huxley",
        degree: int = 0.10,
        missing: str = "mcar",
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,
    ):
        """Base class for Sequential Neural Posterior Estimation methods.

        Args:
            density_estimator: If it is a string, use a pre-configured network of the
                provided type (one of nsf, maf, mdn, made). Alternatively, a function
                that builds a custom neural network can be provided. The function will
                be called with the first batch of simulations (theta, x), which can
                thus be used for shape inference and potentially for z-scoring. It
                needs to return a PyTorch `nn.Module` implementing the density
                estimator. The density estimator needs to provide the methods
                `.log_prob` and `.sample()`.

        See docstring of `NeuralInference` class for all other arguments.
        """

        super().__init__(
            prior=prior,
            device=device,
            logging_level=logging_level,
            summary_writer=summary_writer,
            show_progress_bars=show_progress_bars,
        )

        # As detailed in the docstring, `density_estimator` is either a string or
        # a callable. The function creating the neural network is attached to
        # `_build_neural_net`. It will be called in the first round and receive
        # thetas and xs as inputs, so that they can be used for shape inference and
        # potentially for z-scoring.
        self.missing = missing
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        check_estimator_arg(density_estimator)
        self.exp_type = types
        self.degree = degree
        if degree>0:
            if missing == 'mnar':
                if types == 'huxley' or types == 'oup' or types == 'ricker':
                    #self.missing_model = MLP_simple(10).to(device)
                    self.missing_model = CNN_LNP_MNAR(1).to(device)
                    #breakpoint()
                else:
                    self.missing_model = MLP_LNP_MNAR(10).to(device)
            else:
                if types == 'huxley' or types == 'oup' or types == 'ricker':
                    #self.missing_model = MLP_simple(10).to(device)
                    self.missing_model = CNN_LNP(1).to(device)
                    #breakpoint()
                else:
                    self.missing_model = MLP_LNP(10).to(device)

        if isinstance(density_estimator, str):
            self._build_neural_net = utils.posterior_nn(model=density_estimator)
        else:
            self._build_neural_net = density_estimator

        self._proposal_roundwise = []
        self.use_non_atomic_loss = False

    def append_simulations(
        self,
        theta: Tensor,
        x: Tensor,
        proposal: Optional[DirectPosterior] = None,
        data_device: Optional[str] = None,
    ) -> "PosteriorEstimator":
        r"""Store parameters and simulation outputs to use them for later training.

        Data are stored as entries in lists for each type of variable (parameter/data).

        Stores $\theta$, $x$, prior_masks (indicating if simulations are coming from the
        prior or not) and an index indicating which round the batch of simulations came
        from.

        Args:
            theta: Parameter sets.
            x: Simulation outputs.
            proposal: The distribution that the parameters $\theta$ were sampled from.
                Pass `None` if the parameters were sampled from the prior. If not
                `None`, it will trigger a different loss-function.
            data_device: Where to store the data, default is on the same device where
                the training is happening. If training a large dataset on a GPU with not
                much VRAM can set to 'cpu' to store data on system memory instead.

        Returns:
            NeuralInference object (returned so that this function is chainable).
        """

        is_valid_x, num_nans, num_infs = handle_invalid_x(x, True)  # Hardcode to True

        x = x[is_valid_x]
        theta = theta[is_valid_x]

        # Check for problematic z-scoring
        warn_if_zscoring_changes_data(x)
        warn_on_invalid_x(num_nans, num_infs, True)
        warn_on_invalid_x_for_snpec_leakage(
            num_nans, num_infs, True, type(self).__name__, self._round
        )

        if data_device is None:
            data_device = self._device

        theta, x = validate_theta_and_x(
            theta, x, data_device=data_device, training_device=self._device
        )
        self._check_proposal(proposal)

        if (
            proposal is None
            or proposal is self._prior
            or (
                isinstance(proposal, RestrictedPrior) and proposal._prior is self._prior
            )
        ):
            # The `_data_round_index` will later be used to infer if one should train
            # with MLE loss or with atomic loss (see, in `train()`:
            # self._round = max(self._data_round_index))
            self._data_round_index.append(0)
            prior_masks = mask_sims_from_prior(0, theta.size(0))
        else:
            if not self._data_round_index:
                # This catches a pretty specific case: if, in the first round, one
                # passes data that does not come from the prior.
                self._data_round_index.append(1)
            else:
                self._data_round_index.append(max(self._data_round_index) + 1)
            prior_masks = mask_sims_from_prior(1, theta.size(0))

        self._theta_roundwise.append(theta)
        self._x_roundwise.append(x)
        self._prior_masks.append(prior_masks)
        self._proposal_roundwise.append(proposal)

        if self._prior is None or isinstance(self._prior, ImproperEmpirical):
            if proposal is not None:
                raise ValueError(
                    "You had not passed a prior at initialization, but now you "
                    "passed a proposal. If you want to run multi-round SNPE, you have "
                    "to specify a prior (set the `.prior` argument or re-initialize "
                    "the object with a prior distribution). If the samples you passed "
                    "to `append_simulations()` were sampled from the prior, you can "
                    "run single-round inference with "
                    "`append_simulations(..., proposal=None)`."
                )
            theta_prior = self.get_simulations()[0].to(self._device)
            self._prior = ImproperEmpirical(
                theta_prior, ones(theta_prior.shape[0], device=self._device)
            )

        return self


    def train(
        self,
        training_batch_size: int = 50,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 50,
        max_num_epochs: int = 2**31 - 1,
        clip_max_norm: Optional[float] = 5.0,
        calibration_kernel: Optional[Callable] = None,
        resume_training: bool = False,
        force_first_round_loss: bool = False,
        discard_prior_samples: bool = False,
        retrain_from_scratch: bool = False,
        show_train_summary: bool = True,
        dataloader_kwargs: Optional[dict] = None,
        distance: str = "euclidean",
        beta: float = 1,
        x_obs: Tensor = None
    ) -> nn.Module:
        r"""Return density estimator that approximates the distribution $p(\theta|x)$.

        Args:
            training_batch_size: Training batch size.
            learning_rate: Learning rate for Adam optimizer.
            validation_fraction: The fraction of data to use for validation.
            stop_after_epochs: The number of epochs to wait for improvement on the
                validation set before terminating training.
            max_num_epochs: Maximum number of epochs to run. If reached, we stop
                training even when the validation loss is still decreasing. Otherwise,
                we train until validation loss increases (see also `stop_after_epochs`).
            clip_max_norm: Value at which to clip the total gradient norm in order to
                prevent exploding gradients. Use None for no clipping.
            calibration_kernel: A function to calibrate the loss with respect to the
                simulations `x`. See Lueckmann, Gonçalves et al., NeurIPS 2017.
            resume_training: Can be used in case training time is limited, e.g. on a
                cluster. If `True`, the split between train and validation set, the
                optimizer, the number of epochs, and the best validation log-prob will
                be restored from the last time `.train()` was called.
            force_first_round_loss: If `True`, train with maximum likelihood,
                i.e., potentially ignoring the correction for using a proposal
                distribution different from the prior.
            discard_prior_samples: Whether to discard samples simulated in round 1, i.e.
                from the prior. Training may be sped up by ignoring such less targeted
                samples.
            retrain_from_scratch: Whether to retrain the conditional density
                estimator for the posterior from scratch each round.
            show_train_summary: Whether to print the number of epochs and validation
                loss after the training.
            dataloader_kwargs: Additional or updated kwargs to be passed to the training
                and validation dataloaders (like, e.g., a collate_fn)
            distance: use which method to train corrupted method

        Returns:
            Density estimator that approximates the distribution $p(\theta|x)$.
        """
        # Load data from most recent round.
        self._round = max(self._data_round_index)
        

        if self._round == 0 and self._neural_net is not None:
            assert force_first_round_loss, (
                "You have already trained this neural network. After you had trained "
                "the network, you again appended simulations with `append_simulations"
                "(theta, x)`, but you did not provide a proposal. If the new "
                "simulations are sampled from the prior, you can set "
                "`.train(..., force_first_round_loss=True`). However, if the new "
                "simulations were not sampled from the prior, you should pass the "
                "proposal, i.e. `append_simulations(theta, x, proposal)`. If "
                "your samples are not sampled from the prior and you do not pass a "
                "proposal and you set `force_first_round_loss=True`, the result of "
                "SNPE will not be the true posterior. Instead, it will be the proposal "
                "posterior, which (usually) is more narrow than the true posterior."
            )

        # Calibration kernels proposed in Lueckmann, Gonçalves et al., 2017.
        if calibration_kernel is None:
            calibration_kernel = lambda x: ones([len(x)], device=self._device)

        # Starting index for the training set (1 = discard round-0 samples).
        start_idx = int(discard_prior_samples and self._round > 0)

        # For non-atomic loss, we can not reuse samples from previous rounds as of now.
        # SNPE-A can, by construction of the algorithm, only use samples from the last
        # round. SNPE-A is the only algorithm that has an attribute `_ran_final_round`,
        # so this is how we check for whether or not we are using SNPE-A.
        if self.use_non_atomic_loss or hasattr(self, "_ran_final_round"):
            start_idx = self._round

        # Set the proposal to the last proposal that was passed by the user. For
        # atomic SNPE, it does not matter what the proposal is. For non-atomic
        # SNPE, we only use the latest data that was passed, i.e. the one from the
        # last proposal.
        proposal = self._proposal_roundwise[-1]

        train_loader, val_loader = self.get_dataloaders(
            self.exp_type,
            self.degree,
            self.missing,
            start_idx,
            training_batch_size,
            validation_fraction,
            resume_training,
            dataloader_kwargs=dataloader_kwargs,
        )
        # First round or if retraining from scratch:
        # Call the `self._build_neural_net` with the rounds' thetas and xs as
        # arguments, which will build the neural network.
        # This is passed into NeuralPosterior, to create a neural posterior which
        # can `sample()` and `log_prob()`. The network is accessible via `.net`.
        if self._neural_net is None or retrain_from_scratch:

            # Get theta,x to initialize NN
            theta, x, _ = self.get_simulations(starting_round=start_idx)

            # Use only training data for building the neural net (z-scoring transforms)

            self._neural_net = self._build_neural_net(
                theta[self.train_indices].to("cpu"),
                x[self.train_indices, 0].to("cpu"),
            )
            self._x_shape = x_shape_from_simulation(x.to("cpu"))

            # test_posterior_net_for_multi_d_x(
            #     self._neural_net,
            #     theta.to("cpu"),
            #     x.to("cpu"),
            # )

            del theta, x

        # Move entire net to device for training.
        if self.degree>0:
            self._neural_net.to(self._device)
            self.missing_model.to(self.device)
        else:
            self._neural_net.to(self._device)

        if not resume_training:
            if self.degree>0:
                self.optimizer = optim.Adam(
                    list(self._neural_net.parameters()) + list(self.missing_model.parameters())  , lr=learning_rate
                )
            else:
                self.optimizer = optim.Adam(
                    list(self._neural_net.parameters())  , lr=learning_rate
                )
            self.epoch, self._val_log_prob = 0, float("-Inf")
        time_list = []
        while self.epoch <= max_num_epochs and not self._converged(
            self.epoch, stop_after_epochs
        ):

            # Train for a single epoch.
            self._neural_net.train()
            #self.missing_model.train()
            train_log_probs_sum = 0
            missing_loss_sum = 0
            train_loss_sum = 0
            epoch_start_time = time.time()
            t0 = time.time()
            for batch in train_loader:
                #breakpoint()
                self.optimizer.zero_grad()
                # Get batches on current device.
                theta_batch, x_batch, masks_batch,masks_data = (
                    batch[0].to(self._device),
                    batch[1].to(self._device),
                    batch[2].to(self._device),
                    batch[3].to(self._device),
                )

                if self.degree > 0:
                
                    if self.exp_type == 'huxley':
                        x_norm = (x_batch  - self.min_val)/(self.max_val - self.min_val)
                        x_new = x_norm[:,:,:1200].view(-1,1,1,30,40)
                        trgt =x_norm[:,:,:1200].view(-1,1,30,40)
                        masks_new = masks_data[:,:,:1200].view(-1,1,1,30,40)
                        if self.missing == 'mnar':
                            pred_mean,std,pred_dist,mask_pred = self.missing_model(x_new.squeeze(dim=1),masks_new.squeeze(dim=1))
                        else:
                            pred_mean,std,pred_dist = self.missing_model(x_new.squeeze(dim=1),masks_new.squeeze(dim=1))
                        pred_mean = torch.cat([pred_mean.view(-1,1,1200),x_norm[:,:,-1].unsqueeze(dim=-1)],dim=-1)

                        x_unnorm = pred_mean*(self.max_val - self.min_val) + self.min_val
                        x_new = masks_data*x_batch + (1-masks_data)*x_unnorm
                        #print("After transform missing model",x_new.shape)
                        summ_stat = []
                        I_inj, t_on, t_off, dt, t, A_soma = syn_current(dt=0.1)
                        t = np.arange(0, len(I_inj), 1) * dt
                        for sample in range(x_new.shape[0]):
                    #print(sample)
                            summary = calculate_summary_statistics_torch(x_new[sample].flatten(),t,dt)
                            summ_stat.append(summary)
                
                        summ_stat_arr = torch.stack(summ_stat)

                    else:
                        x_norm = (x_batch  - self.min_val)/(self.max_val - self.min_val)
                        trgt = x_norm
                        #pred_mean = self.missing_model(x_norm.squeeze(dim=1),masks_data.squeeze(dim=1))
                        if self.missing == 'mnar':
                            pred_mean,std,pred_dist, mask_pred = self.missing_model(x_norm.squeeze(1),masks_data.squeeze(1))
                        else:
                            pred_mean,std,pred_dist= self.missing_model(x_norm.squeeze(1),masks_data.squeeze(1))

                        pred_mean = pred_mean.unsqueeze(dim=1)
                        
                        x_unnorm = pred_mean*(self.max_val - self.min_val) + self.min_val
                        x_new = masks_data*x_batch + (1-masks_data)*x_unnorm
                        summ_stat_arr = x_new
                        #print(self.min_val.shape,self.max_val.shape,x_batch.shape,x_unnorm.shape)
                        #print(summ_stat_arr.shape)

                else:
                    if self.exp_type == 'huxley':
                        summ_stat = []
                        I_inj, t_on, t_off, dt, t, A_soma = syn_current(dt=0.1)
                        t = np.arange(0, len(I_inj), 1) * dt
                        for sample in range(x_batch.shape[0]):
                            summary = calculate_summary_statistics_torch(x_batch[sample].flatten(),t,dt)
                            summ_stat.append(summary)
                
                        summ_stat_arr = torch.stack(summ_stat)

                    else:
                        summ_stat_arr = x_batch

                train_losses, embedding_context, embedding_context_hidden = self._loss(
                    theta_batch,
                    summ_stat_arr,#[:, 0, :, :],
                    masks_batch,
                    proposal,
                    calibration_kernel,
                    force_first_round_loss=force_first_round_loss,
                )
                #breakpoint()

                if distance == "mmd":
                    theta, x, _ = self.get_simulations(starting_round=0)
                    theta_dim = theta[0].shape[0]

                    _, embedding_context_cont, embedding_context_cont_hidden = self._loss(
                        theta[0].reshape(-1, theta_dim),
                        x_obs,
                        masks_batch,
                        proposal,
                        calibration_kernel,
                        force_first_round_loss=True,
                    )

                    index_list = [int(i) for i in range(len(theta))]
                    random.shuffle(index_list)
                    theta = theta[index_list[:200]]
                    x = x[index_list[:200]]

                    _, embedding_context, _ = self._loss(
                        theta,
                        x,
                        masks_batch,
                        proposal,
                        calibration_kernel,
                        force_first_round_loss=True,
                    )

                    summary_loss = MMD_unweighted(embedding_context, embedding_context_cont, lengthscale=median_heuristic(embedding_context))

                    t_loss = torch.mean(train_losses)

                    train_loss = t_loss + beta * summary_loss

                elif distance == "euclidean":
                    theta, x, _ = self.get_simulations(starting_round=0)
                    theta_dim = theta[0].shape[0]

                    _, embedding_context_cont, embedding_context_cont_hidden = self._loss(
                        theta[0].reshape(-1, theta_dim),
                        x_obs,
                        masks_batch,
                        proposal,
                        calibration_kernel,
                        force_first_round_loss=True,
                    )

                    index_list = [int(i) for i in range(len(theta))]
                    random.shuffle(index_list)
                    theta = theta[index_list[:200]]
                    x = x[index_list[:200]]

                    _, embedding_context, _ = self._loss(
                        theta,
                        x,
                        masks_batch,
                        proposal,
                        calibration_kernel,
                        force_first_round_loss=True,
                    )

                    summary_loss = torch.mean(torch.cdist(embedding_context_cont, embedding_context, p=2.0))

                    t_loss = torch.mean(train_losses)

                    train_loss = t_loss + beta * summary_loss
                elif distance == "none":
                    #train_loss = torch.mean(train_losses) - 1e2*pred_dist.log_prob(trgt).mean() #huxley
                    if self.degree > 0:
                        if self.missing == 'mnar':
                            #print(mask_pred.shape,masks_data.shape)
                            #breakpoint()
                            if self.exp_type == 'huxley':
                                mask_loss = 100*nn.BCELoss()(mask_pred,masks_new.squeeze(1).float())
                                train_loss = torch.mean(train_losses)  - 1e2*pred_dist.log_prob(trgt).mean() + mask_loss
                            else:
                                mask_loss = nn.BCELoss()(mask_pred,masks_data.squeeze(1).float())
                                train_loss = torch.mean(train_losses)  - pred_dist.log_prob(trgt).mean() + mask_loss
                        else:
                            train_loss = torch.mean(train_losses)  - 1e2*pred_dist.log_prob(trgt).mean()
                    else:
                        train_loss = torch.mean(train_losses)
                    #print(' Loss for the batch : ',train_loss.item())
                    
                else:
                    raise NotImplementedError
                train_log_probs_sum -= train_losses.sum().item()
                #missing_loss_sum -= missing_loss.item()
                train_loss_sum -= train_loss.item()
                train_loss.backward()

                if clip_max_norm is not None:
                    clip_grad_norm_(
                        self._neural_net.parameters(), max_norm=clip_max_norm
                    )
                self.optimizer.step()
            epoch_end_time = time.time()
            #print(f"Time to train one epoch: {epoch_end_time-epoch_start_time}")
            self.epoch += 1

            train_log_prob_average = train_log_probs_sum / (
                len(train_loader) * train_loader.batch_size  # type: ignore
            )
            #missing_loss_avg = missing_loss_sum / (
            #    len(train_loader) * train_loader.batch_size  # type: ignore
            #)
            print(' Train Loss : ',-train_loss_sum / (
                len(train_loader) * train_loader.batch_size  # type: ignore
            ))
            self._summary["training_log_probs"].append(train_log_prob_average)

            # Calculate validation performance.
            if self.degree > 0:
                self._neural_net.eval()
                self.missing_model.eval()
            else:
                self._neural_net.eval()
            
            val_log_prob_sum = 0

            with torch.no_grad():
                for batch in val_loader:
                    theta_batch, x_batch, masks_batch,masks_data = (
                    batch[0].to(self._device),
                    batch[1].to(self._device),
                    batch[2].to(self._device),
                    batch[3].to(self._device),
                )
                    if self.degree > 0:
                        if self.exp_type == 'huxley':
                            x_norm = (x_batch  - self.min_val)/(self.max_val - self.min_val)
                            x_new = x_norm[:,:,:1200].view(-1,1,1,30,40)
                            #trgt = x_norm[:,:,:1200].view(-1,1,30,40)
                            masks_new = masks_data[:,:,:1200].view(-1,1,1,30,40)
                            #print(x_norm.shape,masks_data.shape)
                            #breakpoint()
                            if self.missing == 'mnar':
                                pred_mean,std,pred_dist,mask_pred = self.missing_model(x_new.squeeze(dim=1),masks_new.squeeze(dim=1))
                            else:
                                pred_mean,std,pred_dist = self.missing_model(x_new.squeeze(dim=1),masks_new.squeeze(dim=1))
                            #print(pred_mean.view(-1,1,1200).shape,x_norm[:,:,-1].shape)
                            pred_mean = torch.cat([pred_mean.view(-1,1,1200),x_norm[:,:,-1].unsqueeze(dim=-1)],dim=-1)
                            #masks_data = torch.cat([masks_new.view(-1,1,1200),x_norm[:,:,-1].unsqueeze(dim=-1)],dim=-1)
                            #print(pred_mean.shape)
                            #breakpoint()
                            x_unnorm = pred_mean*(self.max_val - self.min_val) + self.min_val
                            x_new = masks_data*x_batch + (1-masks_data)*x_unnorm
                            #print("After transform missing model",x_new.shape)
                            summ_stat = []
                            I_inj, t_on, t_off, dt, t, A_soma = syn_current(dt=0.1)
                            t = np.arange(0, len(I_inj), 1) * dt
                            for sample in range(x_new.shape[0]):
                        #print(sample)
                                summary = calculate_summary_statistics_torch(x_new[sample].flatten(),t,dt)
                                summ_stat.append(summary)
                    
                            summ_stat_arr = torch.stack(summ_stat)
                        
                        else:
                            x_norm = (x_batch  - self.min_val)/(self.max_val - self.min_val)
                            trgt = x_norm
                            if self.missing == 'mnar':
                                pred_mean,std,pred_dist, mask_pred = self.missing_model(x_norm.squeeze(1),masks_data.squeeze(1))
                            else:
                                pred_mean,std,pred_dist= self.missing_model(x_norm.squeeze(1),masks_data.squeeze(1))

                            pred_mean = pred_mean.unsqueeze(dim=1)
                            x_unnorm = pred_mean*(self.max_val - self.min_val) + self.min_val
                            x_new = masks_data*x_batch + (1-masks_data)*x_unnorm
                            summ_stat_arr = x_new

                    else:
                        if self.exp_type == 'huxley':
                            summ_stat = []
                            I_inj, t_on, t_off, dt, t, A_soma = syn_current(dt=0.1)
                            t = np.arange(0, len(I_inj), 1) * dt
                            for sample in range(x_batch.shape[0]):
                                summary = calculate_summary_statistics_torch(x_batch[sample].flatten(),t,dt)
                                summ_stat.append(summary)
                    
                            summ_stat_arr = torch.stack(summ_stat)

                        else:
                            summ_stat_arr = x_batch

                    #x_norm = (x_batch  - self.min_val)/(self.max_val - self.min_val)
                    #pred_mean,std,pred_dist = self.missing_model(x_norm.squeeze(dim=1),masks_data.squeeze(dim=1))

                    #pred_mean = pred_mean.unsqueeze(dim=1)
                    #x_unnorm = pred_mean*(self.max_val - self.min_val) + self.min_val
                    #x_new = masks_data*x_batch + (1-masks_data)*x_unnorm

                    val_losses, _, _ = self._loss(
                        theta_batch,
                        summ_stat_arr,
                        masks_batch,
                        proposal,
                        calibration_kernel,
                        force_first_round_loss=force_first_round_loss,
                    )
                    val_log_prob_sum -= val_losses.sum().item()

            # Take mean over all validation samples.
            self._val_log_prob = val_log_prob_sum / (
                len(val_loader) * val_loader.batch_size  # type: ignore
            )
            # Log validation log prob for every epoch.
            self._summary["validation_log_probs"].append(self._val_log_prob)
            self._summary["epoch_durations_sec"].append(time.time() - epoch_start_time)

            self._maybe_show_progress(self._show_progress_bars, self.epoch)

        self._report_convergence_at_end(self.epoch, stop_after_epochs, max_num_epochs)

        # Update summary.
        self._summary["epochs_trained"].append(self.epoch)
        self._summary["best_validation_log_prob"].append(self._best_val_log_prob)

        # Update tensorboard and summary dict.
        self._summarize(round_=self._round)

        # Update description for progress bar.
        if show_train_summary:
            print(self._describe_round(self._round, self._summary))

        # Avoid keeping the gradients in the resulting network, which can
        # cause memory leakage when benchmarking.
        if self.degree>0:
            self._neural_net.zero_grad(set_to_none=True)
            self.missing_model.zero_grad(set_to_none=True)
        else:
            self._neural_net.zero_grad(set_to_none=True)

        if self.degree > 0:
            return deepcopy(self._neural_net), deepcopy(self.missing_model)
        
        else:
            return deepcopy(self._neural_net)

    def build_posterior(
        self,
        density_estimator: Optional[nn.Module] = None,
        prior: Optional[Distribution] = None,
        sample_with: str = "rejection",
        mcmc_method: str = "slice_np",
        vi_method: str = "rKL",
        mcmc_parameters: Dict[str, Any] = {},
        vi_parameters: Dict[str, Any] = {},
        rejection_sampling_parameters: Dict[str, Any] = {},
    ) -> Union[MCMCPosterior, RejectionPosterior, VIPosterior, DirectPosterior]:
        r"""Build posterior from the neural density estimator.

        For SNPE, the posterior distribution that is returned here implements the
        following functionality over the raw neural density estimator:
        - correct the calculation of the log probability such that it compensates for
            the leakage.
        - reject samples that lie outside of the prior bounds.
        - alternatively, if leakage is very high (which can happen for multi-round
            SNPE), sample from the posterior with MCMC.

        Args:
            density_estimator: The density estimator that the posterior is based on.
                If `None`, use the latest neural density estimator that was trained.
            prior: Prior distribution.
            sample_with: Method to use for sampling from the posterior. Must be one of
                [`mcmc` | `rejection` | `vi`].
            mcmc_method: Method used for MCMC sampling, one of `slice_np`, `slice`,
                `hmc`, `nuts`. Currently defaults to `slice_np` for a custom numpy
                implementation of slice sampling; select `hmc`, `nuts` or `slice` for
                Pyro-based sampling.
            vi_method: Method used for VI, one of [`rKL`, `fKL`, `IW`, `alpha`]. Note
                some of the methods admit a `mode seeking` property (e.g. rKL) whereas
                some admit a `mass covering` one (e.g fKL).
            mcmc_parameters: Additional kwargs passed to `MCMCPosterior`.
            vi_parameters: Additional kwargs passed to `VIPosterior`.
            rejection_sampling_parameters: Additional kwargs passed to
                `RejectionPosterior` or `DirectPosterior`. By default,
                `DirectPosterior` is used. Only if `rejection_sampling_parameters`
                contains `proposal`, a `RejectionPosterior` is instantiated.

        Returns:
            Posterior $p(\theta|x)$  with `.sample()` and `.log_prob()` methods
            (the returned log-probability is unnormalized).
        """
        if prior is None:
            assert self._prior is not None, (
                "You did not pass a prior. You have to pass the prior either at "
                "initialization `inference = SNPE(prior)` or to "
                "`.build_posterior(prior=prior)`."
            )
            prior = self._prior
        else:
            utils.check_prior(prior)

        if density_estimator is None:
            posterior_estimator = self._neural_net
            # If internal net is used device is defined.
            device = self._device
        else:
            posterior_estimator = density_estimator
            # Otherwise, infer it from the device of the net parameters.
            device = next(density_estimator.parameters()).device.type

        potential_fn, theta_transform = posterior_estimator_based_potential(
            posterior_estimator=posterior_estimator, prior=prior, x_o=None
        )

        if sample_with == "rejection":
            if "proposal" in rejection_sampling_parameters.keys():
                self._posterior = RejectionPosterior(
                    potential_fn=potential_fn,
                    device=device,
                    x_shape=self._x_shape,
                    **rejection_sampling_parameters,
                )
            else:
                self._posterior = DirectPosterior(
                    posterior_estimator=posterior_estimator,
                    prior=prior,
                    x_shape=self._x_shape,
                    device=device,
                )
        elif sample_with == "mcmc":
            self._posterior = MCMCPosterior(
                potential_fn=potential_fn,
                theta_transform=theta_transform,
                proposal=prior,
                method=mcmc_method,
                device=device,
                x_shape=self._x_shape,
                **mcmc_parameters,
            )
        elif sample_with == "vi":
            self._posterior = VIPosterior(
                potential_fn=potential_fn,
                theta_transform=theta_transform,
                prior=prior,  # type: ignore
                vi_method=vi_method,
                device=device,
                x_shape=self._x_shape,
                **vi_parameters,
            )
        else:
            raise NotImplementedError

        # Store models at end of each round.
        self._model_bank.append(deepcopy(self._posterior))

        return deepcopy(self._posterior)

    @abstractmethod
    def _log_prob_proposal_posterior(
        self,
        theta: Tensor,
        x: Tensor,
        masks: Tensor,
        proposal: Optional[Any],
    ) -> Tensor:
        raise NotImplementedError

    def _loss(
        self,
        theta: Tensor,
        x: Tensor,
        masks: Tensor,
        proposal: Optional[Any],
        calibration_kernel: Callable,
        force_first_round_loss: bool = False,
    ) -> Tensor:
        """Return loss with proposal correction (`round_>0`) or without it (`round_=0`).

        The loss is the negative log prob. Irrespective of the round or SNPE method
        (A, B, or C), it can be weighted with a calibration kernel.

        Returns:
            Calibration kernel-weighted negative log prob.
            force_first_round_loss: If `True`, train with maximum likelihood,
                i.e., potentially ignoring the correction for using a proposal
                distribution different from the prior.
        """
        if self._round == 0 or force_first_round_loss:
            # Use posterior log prob (without proposal correction) for first round.
            log_prob, embedding_context, embedding_hidden = self._neural_net.log_prob(theta, x)
        else:
            log_prob, embedding_context, embedding_hidden = self._log_prob_proposal_posterior(theta, x, masks, proposal)

        return -(calibration_kernel(x) * log_prob), embedding_context, embedding_hidden

    def _check_proposal(self, proposal):
        """
        Check for validity of the provided proposal distribution.

        If the proposal is a `NeuralPosterior`, we check if the default_x is set.
        If the proposal is **not** a `NeuralPosterior`, we warn since it is likely that
        the user simply passed the prior, but this would still trigger atomic loss.
        """
        if proposal is not None:
            check_if_proposal_has_default_x(proposal)

            if isinstance(proposal, RestrictedPrior):
                if proposal._prior is not self._prior:
                    warn(
                        "The proposal you passed is a `RestrictedPrior`, but the "
                        "proposal distribution it uses is not the prior (it can be "
                        "accessed via `RestrictedPrior._prior`). We do not "
                        "recommend to mix the `RestrictedPrior` with multi-round "
                        "SNPE."
                    )
            elif (
                not isinstance(proposal, NeuralPosterior)
                and proposal is not self._prior
            ):
                warn(
                    "The proposal you passed is neither the prior nor a "
                    "`NeuralPosterior` object. If you are an expert user and did so "
                    "for research purposes, this is fine. If not, you might be doing "
                    "something wrong: feel free to create an issue on Github."
                )
        elif self._round > 0:
            raise ValueError(
                "A proposal was passed but no prior was passed at initialisation. When "
                "running multi-round inference, a prior needs to be specified upon "
                "initialisation. Potential fix: setting the `._prior` attribute or "
                "re-initialisation. If the samples passed to `append_simulations()` "
                "were sampled from the prior, single-round inference can be performed "
                "with `append_simulations(..., proprosal=None)`."
            )
