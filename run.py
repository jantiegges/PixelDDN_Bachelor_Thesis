import numpy as np
import tqdm

import torch, argparse
import os
from torch.utils.tensorboard import SummaryWriter

from models import pixelDDN
from data import get_dataset
from utils import helper, figures
from utils.file_manager import pickle_save, pickle_load
from os import path


# save directory names
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_DIR, "./saved_data")
MODEL_DIR = os.path.join(THIS_DIR, "./saved_models")
EXP_DIR = os.path.join(THIS_DIR, "./saved_experiments")


class PixelDDNTrainer:
    """ class that contains all function for training the network """

    def __init__(self, ae_model, ddn_model, ae_config, ddn_config, train_config, data_config, args):
        """
        Params:
             ae_model (string): name of autoencoder model
             ddn_model (string): name of dynamic network model
             ae_config (dict): model parameters for autoencoder
             ddn_config (dict): model parameters for dynamic network
             train_config (dict): hyperparameters for training process
             data_config (dict): properties of dataset
             args: other arguments given
        """
        self.ae_model = ae_model
        self.ddn_model = ddn_model
        self.ae_params = ae_config
        self.ddn_params = ddn_config
        self.train_params = train_config
        self.data_params = data_config

        self.loss_type = self.train_params['loss_type']
        self.beta = self.train_params['beta']
        self.pred_steps = self.train_params['pred_steps']

        self.var = self.ae_params['variational']
        self.conv = self.ae_params['convolutional']

        # init variables for geco loss
        self.lambd = self.train_params['lambd']
        self.tol = self.train_params['tol']
        self.alpha = self.train_params['alpha']
        self.gc_ma = None

        # init PixelDDN network
        if ddn_model == "MLP":
            self.model = pixelDDN.PixelMLP(ae_model, ddn_model, ae_config, ddn_config, data_config, args.latent_dim)
        if ddn_model == "LNN":
            self.model = pixelDDN.PixelLNN(ae_model, ddn_model, ae_config, ddn_config, data_config, args.latent_dim)
        if ddn_model == "HNN":
            self.model = pixelDDN.PixelHNN(ae_model, ddn_model, ae_config, ddn_config, data_config, args.latent_dim)
        if ddn_model == "VIN":
            raise NotImplementedError

        # load saved model if path is specified
        if args.model_file != "None":
            PATH = "{}/{}".format(MODEL_DIR, args.model_file)
            self.model.load_state_dict(torch.load(PATH))
            self.model.eval()

        # load training data and data parameters or generate new one, if specified data path doesn't exist yet
        self.data, self.data_params = get_dataset(self.data_params, args.system, DATA_DIR, args.data_file, args.seed)

        # convert data to right format for convolutional autoencoder (channels first)
        self.trainset, self.testset = helper.convert_data(self.data)

        # load the data using the torch Dataloader
        self.trainloader = torch.utils.data.DataLoader(self.trainset, shuffle=True,
                                                  batch_size=self.train_params['batch_size'])

        self.testloader = torch.utils.data.DataLoader(self.trainset, shuffle=True,
                                                  batch_size=self.train_params['batch_size'])

        # specify path for tensorboard writer
        RUN_DIR = "{}/{}_{}".format(EXP_DIR, self.ddn_model, self.ae_model)
        self.writer = SummaryWriter(RUN_DIR)

        # init optimizer
        optim_params = [
            {
                'params': list(self.model.ae.parameters()),
                'lr': self.train_params["ae_lr"]#, 
                #'weight_decay': self.train_params["ae_wd"]
            },
            {
                'params': list(self.model.ddn.parameters()),
                'lr': self.train_params["ddn_lr"]#,
                #'weight_decay': self.train_params["ddn_wd"]
            },
        ]

        self.optimizer = torch.optim.Adam(optim_params)

        self.train_log = []
        self.test_log = []

    def compute_loss(self, pred, target):
        """ computes the loss for one training step
        Params:
            pred (dict): contains model output defined in model_output class
            target (Tensor) (N x time_steps x channels x H x W): contains target image(s)
        Returns:
            loss (float): loss of model
            optional:
                reconstruction_error (float): loss from reconstructing the images
                KLD (float): Kullback-Leibler divergence
        """
        
        # differentiate between variational and non-variational autoencoder
        if self.var:

            data_size = pred.reconstruction.flatten(1).size(1)

            # use mean reduction for loss
            if 'mean' in self.loss_type:
                reconstruction_error = torch.nn.functional.mse_loss(input=pred.reconstruction, target=target)
                kld_loss = torch.mean(-0.5 * torch.sum(1 + pred.z_logvar - pred.z_mean.pow(2) - pred.z_logvar.exp(), dim=1), dim=0)

                if self.loss_type == 'beta_mean':
                    # normalise kld
                    kld_loss = kld_loss / data_size
                    loss = reconstruction_error + (self.beta * kld_loss)
                elif self.loss_type == 'geco_mean':
                    # normalise kld
                    kld_loss = kld_loss / data_size
                    # compute geco contraint
                    geco_constraint = reconstruction_error - self.tol**2
                    loss = self.lambd * geco_constraint + kld_loss

                    # update geco constraint
                    with torch.no_grad():
                        if self.gc_ma is None:
                            self.gc_ma = geco_constraint
                        else:
                            self.gc_ma = self.alpha * self.gc_ma + (1 - self.alpha) * geco_constraint

                elif self.loss_type == 'normal_mean':
                    kld_loss = kld_loss / data_size
                    loss = reconstruction_error + kld_loss
                else:
                    raise NotImplementedError

            # use sum reduction for loss
            elif 'sum' in self.loss_type:
                reconstruction_error = torch.nn.functional.mse_loss(input=pred.reconstruction, target=target, reduction='sum')
                kld_loss = -0.5 * torch.sum(1 + pred.z_logvar - pred.z_mean.pow(2) - pred.z_logvar.exp())

                if self.loss_type == 'beta_sum':
                    # normalise
                    #kld_loss = kld_loss / data_size
                    loss = reconstruction_error + (self.beta * kld_loss)
                elif self.loss_type == 'geco_sum':
                    # compute geco contraint
                    geco_constraint = reconstruction_error - self.tol**2
                    loss = self.lambd * geco_constraint + kld_loss

                    # update geco constraint
                    with torch.no_grad():
                        if self.gc_ma is None:
                            self.gc_ma = geco_constraint
                        else:
                            self.gc_ma = self.alpha * self.gc_ma + (1 - self.alpha) * geco_constraint
                elif self.loss_type == 'normal_sum':
                    #kld_loss = kld_loss / data_size
                    loss = reconstruction_error + kld_loss
                else:
                    raise NotImplementedError

            return {'Total Loss': loss, 'Reconstruction Loss': reconstruction_error, 'KLD': kld_loss}

        else:
            loss = torch.nn.functional.mse_loss(input=pred.reconstruction, target=target)
            return {'Total Loss': loss}

    def fit(self):
        """ implements whole training process for the model """
        # set random seed
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        print(f"Training {self.ddn_model} model with {self.ae_model}")

        seq_len = self.data['settings']['seq_len']

        # use gpu with cuda support if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # use tqdm package for logging the process
        epoch_bar = tqdm.tqdm(total=self.train_params['epochs'], desc="Epoch", position=0)


        # training loop
        for epoch in range(self.train_params['epochs']):

            batch_bar = tqdm.tqdm(self.trainloader, disable=True)

            # loop through all batches
            for i, batch in enumerate(batch_bar):

                # zero out the gradients so that the parameter update is done correctly
                self.optimizer.zero_grad()
                # move to device
                batch = batch.to(device)

                # this is the input for the PixelDDN model
                input = batch[:, :seq_len, ...]

                # last image of the sequence is the target image to predict
                # alternatively one could also include the reconstruction
                # of the last input image: target = batch[:, -(self.pred_steps-1):, ...]
                target = batch[:, (seq_len-1):(seq_len + self.pred_steps), ...]

                # forward input through model
                prediction = self.model(input, self.pred_steps, variational=self.var, convolutional=self.conv)

                # compute losses
                losses = self.compute_loss(prediction, target)

                # perform one step of backward propagation
                loss = losses['Total Loss']
                loss.backward()

                # perform gradient clipping to avoid exploding gradients while training
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_params['max_norm'])

                # perform gradient descent
                self.optimizer.step()

                # every 100 steps update lambda and log loss
                if i % 100 == 99:

                    if 'geco' in self.loss_type and self.var:
                        # update lambda
                        self.lambd *= torch.exp(self.gc_ma.detach())
                        # clamp lambda to avoid infinite values
                        self.lambd = torch.clamp(self.lambd, 1e-10, 1e10)
                        # cast lambda back to float
                        self.lambd = self.lambd.item()

                    # normalise loss for logging for variational autoencoder with sum reduction
                    if 'sum' in self.loss_type:
                        normaliser = target.flatten().shape[0]
                        losses['Reconstruction Loss'] = losses['Reconstruction Loss'] / normaliser
                        losses['KLD'] = losses['KLD'] / target.shape[0]
                        losses['Total Loss'] = losses['KLD'] + losses['Reconstruction Loss']

                    # append train loss
                    self.train_log.append(losses)

                    self.writer.add_scalars("training loss", losses, epoch * len(self.trainloader) + i)

                    # plot first sequence of minibatch to tensorboard
                    plt_input = input[0]
                    plt_target = target[0,-self.pred_steps:,...]
                    plt_pred = prediction.reconstruction[0,-self.pred_steps:,...]

                    self.writer.add_figure("prediction vs. target",
                                           figures.plot_pred_vs_target(plt_input, plt_target, plt_pred, im_size),
                                           epoch * len(self.trainloader) + i)

                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'.format(
                        epoch+1, i * len(batch), len(self.trainloader.dataset),
                               100. * i / len(self.trainloader)), end='')

                    print(", ".join([f"{k}: {v:.3e}" for k, v in losses.items()]))

            # update epoch progress bar
            print("\n")
            epoch_bar.update(1)
            print("\n")

        # save the models learned parameters, it's config files and the training loss
        PATH = f"{MODEL_DIR}/{self.ddn_model}_{self.ae_model}/ae_config"
        pickle_save(self.ae_params, PATH)
        PATH = f"{MODEL_DIR}/{self.ddn_model}_{self.ae_model}/ddn_config"
        pickle_save(self.ddn_params, PATH)
        PATH = f"{MODEL_DIR}/{self.ddn_model}_{self.ae_model}/data_config"
        pickle_save(self.data_params, PATH)
        PATH = f"{MODEL_DIR}/{self.ddn_model}_{self.ae_model}/train_config"
        pickle_save(self.train_params, PATH)
        PATH = f"{MODEL_DIR}/{self.ddn_model}_{self.ae_model}/train_log"
        pickle_save(self.train_log, PATH)
        PATH = f"{MODEL_DIR}/{self.ddn_model}_{self.ae_model}/{self.ddn_model}_{self.ae_model}"
        torch.save(self.model.state_dict(), PATH)

        self.test(PATH)

    def test(self, PATH):
        """ tests a trained model
        Params:
            PATH: path of model to test
        """
        # load saved model parameters
        self.model.load_state_dict(torch.load(PATH))

        # set random seed
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        # print statement on console
        print(f"\nSuccesfully loaded {self.ddn_model} model with {self.ae_model} from {PATH}")
        print(f"\nTesting...")

        seq_len = self.data['settings']['seq_len']

        # use gpu with cuda support if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        batch_bar = tqdm.tqdm(self.testloader, disable=True)

        # loop through all batches
        for i, batch in enumerate(batch_bar):
            # move to device
            batch = batch.to(device)

            # this is the input for the PixelDDN model
            input = batch[:, :seq_len, ...]

            # last image of the sequence is the target image to predict
            # alternatively one could also include the reconstruction
            # of the last input image: target = batch[:, -(self.pred_steps-1):, ...]
            target = batch[:, (seq_len - 1):(seq_len + self.pred_steps), ...]

            # forward input through model
            prediction = self.model(input, self.pred_steps, variational=self.var, convolutional=self.conv)

            # compute losses
            losses = self.compute_loss(prediction, target)

            if i % 100 == 99:

                # normalise loss for logging for variational autoencoder with sum reduction
                if 'sum' in self.loss_type:
                    normaliser = target.flatten().shape[0]
                    losses['Reconstruction Loss'] = losses['Reconstruction Loss'] / normaliser
                    losses['KLD'] = losses['KLD'] / target.shape[0]
                    losses['Total Loss'] = losses['KLD'] + losses['Reconstruction Loss']

                self.test_log.append(losses)

                self.writer.add_scalars("test loss", losses, len(self.testloader) + i)

                # plot first sequence of minibatch to tensorboard
                plt_input = input[0]
                plt_target = target[0, -self.pred_steps:, ...]
                plt_pred = prediction.reconstruction[0, -self.pred_steps:, ...]
                self.writer.add_figure("prediction vs. target",
                                       figures.plot_pred_vs_target(plt_input, plt_target, plt_pred, im_size),
                                       len(self.testloader) + i)

                print('[{}/{} ({:.0f}%)]\t'.format(
                    i * len(batch), len(self.testloader.dataset),
                    100. * i / len(self.testloader)), end='')

                print(", ".join([f"{k}: {v:.3e}" for k, v in losses.items()]))

        # save the test losses for evaluation
        PATH = f"{MODEL_DIR}/{self.ddn_model}_{self.ae_model}/test_log"
        pickle_save(self.test_log, PATH)


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("--ae_model", default="CVAE", type=str, help="model type of autoencoder")
    parser.add_argument("--ddn_model", default="LNN", type=str, help="model type of dynamics network")
    parser.add_argument("--system", default="pendulum", type=str, help="dynamical system to train on")
    parser.add_argument("--latent_dim", default=2, type=int, help="dimension of the latent space")
    parser.add_argument("--data_file", default="not_given", type=str, help="file name of dataset")
    parser.add_argument("--model_file", default="None", type=str, help="file name of trained model to load")
    parser.add_argument("--train_flag", default=True, type=bool, help="bool whether model should be trained or not")
    parser.add_argument("--seed", default=32, type=int, help="random seed")
    parser.set_defaults(feature=True)
    return parser.parse_args()

if __name__ == "__main__":

    # read configuration files
    filepath = path.abspath(path.join(THIS_DIR, 'config/model_config.yaml'))
    model_config = helper.read_config(filepath)
    filepath = path.abspath(path.join(THIS_DIR, 'config/train_config.yaml'))
    train_config = helper.read_config(filepath)
    filepath = path.abspath(path.join(THIS_DIR, 'config/data_config.yaml'))
    data_config = helper.read_config(filepath)

    # Set network models to test
    args = get_args()
    ae_model = args.ae_model
    ddn_model = args.ddn_model

    # initialise autoencoder and dynamics network configurations
    ae_config = model_config['autoencoder'][ae_model]
    ddn_config = model_config['dynamics'][ddn_model]

    if args.data_file == "not_given":
        system = data_config['system']
        ep = data_config['episodes']
        ts = data_config['timesteps']
        im_size = data_config['im_size']
        seq_len = data_config['seq_len']
        channels = data_config['channels']
        latent_dim = data_config['latent_dim']

        # construct data_file
        args.data_file = f"{system}_{ep}_{ts}_{im_size}_{seq_len}_{channels}"

    trainer = PixelDDNTrainer(ae_model, ddn_model, ae_config, ddn_config, train_config, data_config, args)

    # trains and tests or just tests the model
    if args.train_flag:
        trainer.fit()
    else:
        trainer.test()