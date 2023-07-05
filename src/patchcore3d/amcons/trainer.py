import torch
import os
from pathlib import Path

from patchcore3d.amcons.amcons import AnomalyDetectorAMCons

torch.autograd.set_detect_anomaly(True)


class AnomalyDetectorTrainer:
    def __init__(self, dir_out, method, device, item=['flair'], zdim=32, dense=True, variational=False, n_blocks=5, lr=1*1e-4,
                 input_shape=(1, 224, 224), load_weigths=False, epochs_to_test=10, context=False, bayesian=False,
                 restoration=False, loss_reconstruction='bce', iteration=0, level_cams=-4,
                 alpha_kl=10, alpha_entropy=0., expansion_loss_penalty='log_barrier', p_activation_cam=0.2, t=25,
                 alpha_ae=10):

        # Init input variables
        self.dir_out = dir_out
        self.method = method
        self.device=device
        self.item = item
        self.zdim = zdim
        self.dense = dense
        self.variational = variational
        self.n_blocks = n_blocks
        self.input_shape = input_shape
        self.load_weights = load_weigths
        self.epochs_to_test = epochs_to_test
        self.context = context
        self.bayesian = bayesian
        self.restoration = restoration
        self.loss_reconstruction = loss_reconstruction
        self.lr = lr
        self.level_cams = level_cams
        self.expansion_loss_penalty = expansion_loss_penalty
        self.alpha_kl = alpha_kl
        self.alpha_entropy = alpha_entropy
        self.p_activation_cam = p_activation_cam
        self.t = t
        self.alpha_ae = alpha_ae

        # Prepare results folders
        self.dir_results = Path(dir_out) / item[0] / 'iteration_' / str(iteration)
        if not os.path.isdir(dir_out):
            os.makedirs(dir_out)
        if not os.path.isdir(Path(dir_out) / item[0]):
            os.makedirs(Path(dir_out) / item[0])
        if not os.path.isdir(self.dir_results):
            os.makedirs(self.dir_results)

        # Create trainer
        self.method = AnomalyDetectorAMCons(self.dir_results, device=self.device, item=self.item, zdim=self.zdim, lr=self.lr,
                                            input_shape=self.input_shape, epochs_to_test=self.epochs_to_test,
                                            load_weigths=self.load_weights, n_blocks=self.n_blocks,
                                            dense=self.dense, loss_reconstruction=self.loss_reconstruction,
                                            pre_training_epochs=0, level_cams=self.level_cams,
                                            alpha_entropy=self.alpha_entropy,
                                            alpha_kl=self.alpha_kl)

    def train(self, train_generator, epochs, dataset_test):
        self.method.train(train_generator, epochs, dataset_test)