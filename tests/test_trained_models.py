import numpy as np
import scipy.integrate
import tqdm

solve_ivp = scipy.integrate.solve_ivp

import torch, argparse
import os
from torch.utils.tensorboard import SummaryWriter

from models import autoencoder
from data_tmp import get_dataset
from utils import helper, figures
from os import path
import math

# save directory names
THIS_DIR = path.dirname(__file__)
DATA_DIR = os.path.join(THIS_DIR, '..', "./saved_data")
MODEL_DIR = os.path.join(THIS_DIR, '..', "./saved_models")
EXP_DIR = os.path.join(THIS_DIR, '..', "./saved_experiments")

class AETrainer:
    """ class that contains all function for training the autoencoder """

    def __init__(self, ae_model, ae_config, train_config, data_config, args):
        """
        Params:
             ae_model:
             ae_config:
             train_config:
             data_config:
             args:
        """
        self.ae_model = ae_model
        self.ae_params = ae_config
        self.train_params = train_config
        self.data_params = data_config

        self.loss_type = self.train_params['loss_type']
        self.beta = self.train_params['beta']
        self.device = 'cpu'


        en_params = self.ae_params['encoder']
        de_params = self.ae_params['decoder']
        channels = data_config['channels']
        seq_len = data_config['seq_len']
        im_size = data_config['im_size']
        latent_dim = args.latent_dim
        activation = self.train_params['activation']

        # init PixelDDN network
        if ae_model == "LAE":
            self.model = autoencoder.LAE_Test(en_params, de_params, channels, seq_len, im_size, latent_dim, activation)
        if ae_model == "VAE":
            self.model = autoencoder.VAE_Test(en_params, de_params, channels, seq_len, im_size, latent_dim, activation)
        if ae_model == "CAE":
            self.model = autoencoder.CAE(en_params, de_params, channels, seq_len, im_size, latent_dim, activation)
        if ae_model == "CVAE":
            self.model = autoencoder.CVAE_Test(en_params, de_params, channels, seq_len, im_size, latent_dim, activation)

        # load saved model if path is specified
        if args.model_file != "None":
            PATH = "{}/{}".format(MODEL_DIR, args.model_file)
            self.model.load_state_dict(torch.load(PATH))
            self.model.eval()

        # load training data or generate new one, if specified data path doesn't exist yet
        self.data = get_dataset(self.data_params, args.system, DATA_DIR, args.data_file, args.seed)

        # convert data to right format for convolutional autoencoder (channels first)
        self.trainset, self.testset = helper.convert_data(self.data)

        # load the data using the torch Dataloader
        self.trainloader = torch.utils.data.DataLoader(self.trainset, shuffle=True,
                                                  batch_size=self.train_params['batch_size'])

        self.testloader = torch.utils.data.DataLoader(self.trainset, shuffle=True,
                                                  batch_size=self.train_params['batch_size'])

        # specify path for tensorboard writer
        RUN_DIR = "{}/{}".format(EXP_DIR, self.ae_model)
        self.writer = SummaryWriter(RUN_DIR)

        # TODO: what about weight decay
        self.optimizer = torch.optim.Adam(list(self.model.parameters()), self.train_params["ae_lr"])




    def compute_loss(self, pred, target, mu=None, logvar=None, z=None):
        """
        Params:
            prediction (Tensor (Nxtime_steps+1xchannelsxHxW):
            target (tensor :
        """
        # reconstruction of predicted image
        # TODO: might be smart to also include reconstruction of the initial state
        #pred_images = pred.reconstruction[1]
        reconstruction = pred

        # differentiate between variational and non-variational autoencoder
        if self.ae_params['variational']:

            #pred_mean = mu.flatten(1)
            pred_mean = mu
            #pred_logvar = logvar.flatten(1)
            pred_logvar = logvar
            std = torch.exp(logvar/2)

            # compute reconstruction error and Kullback_Leiber Divergence
            #reconstruction_error = torch.nn.functional.mse_loss(input=reconstruction, target=target, reduction='sum')
            reconstruction_error = torch.nn.functional.mse_loss(input=reconstruction, target=target)

            # this line works!!!!!
            kld_loss = torch.mean(-0.5 * torch.sum(1 + pred_logvar - pred_mean.pow(2) - pred_logvar.exp(), dim=1), dim=0)
            #kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            # normalise
            # TODO: why do one normalise?
            #kld_normaliser = reconstruction.flatten(1).size(1)
            #kld_loss = kld_loss / kld_normaliser

            if self.loss_type == 'beta':
                loss = reconstruction_error + self.beta * kld_loss
            elif self.loss_type == 'const_beta':
                # TODO: disentangled beta-vae loss
                # clamp the constraint to avoid infinite values
                #C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter,  1e-10, 1e10)
                #loss = reconstruction_error + self.gamma * (kld_loss - C).abs()
                raise NotImplementedError
            elif self.loss_type == 'normal_vae':
                loss = reconstruction_error + kld_loss
            else:
                raise NotImplementedError

            return {'Total Loss': loss, 'Reconstruction Loss': reconstruction_error, 'KLD': kld_loss}

            # weight = 1  # kwargs['M_N']  # Account for the minibatch samples from the dataset
            #
            # recons_loss = torch.nn.functional.mse_loss(reconstruction, target, reduction='sum')
            #
            # log_q_zx = self.log_density_gaussian(z, mu, logvar).sum(dim=1)
            #
            # zeros = torch.zeros_like(z)
            # log_p_z = self.log_density_gaussian(z, zeros, zeros).sum(dim=1)
            #
            # batch_size, latent_dim = z.shape
            # mat_log_q_z = self.log_density_gaussian(z.view(batch_size, 1, latent_dim),
            #                                         mu.view(1, batch_size, latent_dim),
            #                                         logvar.view(1, batch_size, latent_dim))
            #
            # # Reference
            # # [1] https://github.com/YannDubs/disentangling-vae/blob/535bbd2e9aeb5a200663a4f82f1d34e084c4ba8d/disvae/utils/math.py#L54
            # dataset_size = (1 / 10000) * batch_size  # dataset size
            # strat_weight = (dataset_size - batch_size + 1) / (dataset_size * (batch_size - 1))
            # importance_weights = torch.Tensor(batch_size, batch_size).fill_(1 / (batch_size - 1)).to(self.device)
            # importance_weights.view(-1)[::batch_size] = 1 / dataset_size
            # importance_weights.view(-1)[1::batch_size] = strat_weight
            # importance_weights[batch_size - 2, 0] = strat_weight
            # log_importance_weights = importance_weights.log()
            #
            # mat_log_q_z += log_importance_weights.view(batch_size, batch_size, 1)
            #
            # log_q_z = torch.logsumexp(mat_log_q_z.sum(2), dim=1, keepdim=False)
            # log_prod_q_z = torch.logsumexp(mat_log_q_z, dim=1, keepdim=False).sum(1)
            #
            # mi_loss = (log_q_zx - log_q_z).mean()
            # tc_loss = (log_q_z - log_prod_q_z).mean()
            # kld_loss = (log_prod_q_z - log_p_z).mean()
            #
            # # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
            #
            # self.alpha = 1.0
            # self.gamma = 1.0
            # self.anneal_steps = 10000
            # self.training = True
            #
            #
            # if self.training:
            #     self.num_iter += 1
            #     anneal_rate = min(0 + 1 * self.num_iter / self.anneal_steps, 1)
            # else:
            #     anneal_rate = 1.
            #
            # loss = recons_loss / batch_size + \
            #        self.alpha * mi_loss + \
            #        weight * (self.beta * tc_loss +
            #                  anneal_rate * self.gamma * kld_loss)
            #
            # return {'Total Loss': loss,
            #         'Reconstruction_Loss': recons_loss,
            #         'KLD': kld_loss,
            #         'TC_Loss': tc_loss,
            #         'MI_Loss': mi_loss}

        else:
            loss = torch.nn.functional.mse_loss(input=reconstruction, target=target)
            return {'Total Loss': loss}

    def fit(self):
        """ trains the model
        Params:
            args: arguments from parser
        """
        # set random seed
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        print(f"Training {self.ae_model}")

        seq_len = self.data['settings']['seq_len']

        # use gpu with cuda support if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # use tqdm package for logging the process
        epoch_bar = tqdm.tqdm(total=self.train_params['epochs'], desc="Epoch", position=0)

        train_loss = 0

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
                target = batch[:, seq_len-1, ...]
                # target = input
                # last image of the sequence is the target image to predict
                #target = batch[:, -2:, ...]
                # forward input to model

                if self.ae_params['convolutional']:
                    b, s, c, h, w = input.size()
                    input = input.reshape(b, s * c, h, w)
                    prediction, mu, logvar, z = self.model(input)
                    losses = self.compute_loss(prediction, target, mu, logvar, z)

                elif self.ae_params['variational']:
                    # concat along channel dimension
                    b, s, c, h, w = input.size()
                    input = input.reshape(b, s * c * h * w)
                    b, c, h, w = target.size()
                    target = target.reshape(b, c * h * w)
                    prediction, mu, logvar, z = self.model(input)

                    losses = self.compute_loss(prediction, target, mu, logvar, z)

                else:
                    # concat along channel dimension
                    b, s, c, h, w = input.size()
                    input = input.reshape(b, s * c * h * w)
                    b, c, h, w = target.size()
                    target = target.reshape(b, c * h * w)
                    prediction = self.model(input)
                    target = input[:, seq_len - 1]
                    #target = input
                    losses = self.compute_loss(prediction, target)

                # perform one step of backward propagation
                loss = losses['Total Loss']
                loss.backward()
                train_loss += loss.item()

                # perform gradient descent
                self.optimizer.step()

                # every 50 minibatches
                if i % 100 == 99:
                    self.writer.add_scalars("training loss", losses, epoch * len(self.trainloader) + i)

                    # plot first sequence of minibatch to tensorboard
                    plt_input = input[0]
                    plt_target = target[0]
                    plt_pred = prediction[0]
                    self.writer.add_figure("prediction vs. target",
                                           figures.plot_pred_vs_target(plt_input, plt_target, plt_pred),
                                           epoch * len(self.trainloader) + i)

                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'.format(
                        epoch+1, i * len(batch), len(self.trainloader.dataset),
                               100. * i / len(self.trainloader)), end='')

                    print(", ".join([f"{k}: {(v/len(batch)):.6f}" for k, v in losses.items()]))


            # epoch_bar.write(loss) write loss after every epoch
            epoch_bar.update(1)
            print("\n")

        # save the models learned parameters
        PATH = "{}/{}".format(MODEL_DIR, self.ae_model)
        # TODO: save autoencoder, ddn seperately?
        #torch.save(self.model.load_state_dict(), PATH)

        # TODO: test model --> save the prediction
        self.test()

    def test(self):
        # set random seed
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        print(f"Testing model {self.ae_model}")

        seq_len = self.data['settings']['seq_len']

        # use gpu with cuda support if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        batch_bar = tqdm.tqdm(self.testloader, disable=True)

        # loop through all batches
        for i, batch in enumerate(batch_bar):

            # zero out the gradients so that the parameter update is done correctly
            self.optimizer.zero_grad()
            # move to device
            batch = batch.to(device)

            # this is the input for the PixelDDN model
            input = batch[:, :seq_len, ...]
            target = batch[:, seq_len - 1, ...]
            # target = input
            # last image of the sequence is the target image to predict
            # target = batch[:, -2:, ...]
            # forward input to model
            if self.ae_params['convolutional']:
                b, s, c, h, w = input.size()
                input = input.reshape(b, s * c, h, w)
                prediction, mu, logvar, z = self.model(input)
                losses = self.compute_loss(prediction, target, mu, logvar, z)

            elif self.ae_params['variational']:
                # concat along channel dimension
                b, s, c, h, w = input.size()
                input = input.reshape(b, s * c * h * w)
                b, c, h, w = target.size()
                target = target.reshape(b, c * h * w)
                prediction, mu, logvar, z = self.model(input)

                losses = self.compute_loss(prediction, target, mu, logvar, z)

            else:
                # concat along channel dimension
                b, s, c, h, w = input.size()
                input = input.reshape(b, s * c * h * w)
                b, c, h, w = target.size()
                target = target.reshape(b, c * h * w)
                prediction = self.model(input)
                target = input[:, seq_len - 1]
                # target = input
                losses = self.compute_loss(prediction, target)

            # perform one step of backward propagation
            loss = losses['Total Loss']


            # every 50 minibatches
            if i % 100 == 99:
                self.writer.add_scalars("training loss", losses, len(self.testloader) + i)

                # plot first sequence of minibatch to tensorboard
                plt_input = input[0]
                plt_target = target[0]
                plt_pred = prediction[0]
                self.writer.add_figure("prediction vs. target",
                                       figures.plot_pred_vs_target(plt_input, plt_target, plt_pred),
                                       len(self.testloader) + i)

                print('Test: [{}/{} ({:.0f}%)]\t'.format(
                    i * len(batch), len(self.testloader.dataset),
                    100. * i / len(self.testloader)), end='')

                print(", ".join([f"{k}: {(v / len(batch)):.6f}" for k, v in losses.items()]))



def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("--ae_model", default="LAE", type=str, help="model type of autoencoder")
    parser.add_argument("--ddn_model", default="LNN", type=str, help="model type of dynamics network")
    parser.add_argument("--system", default="pendulum", type=str, help="dynamical system to train on")
    parser.add_argument("--latent_dim", default=2, type=int, help="dimension of the latent dimension")
    parser.add_argument("--data_file", default="pendulum_400_103_64_2_bw", type=str, help="file name of dataset")
    parser.add_argument("--model_file", default="None", type=str, help="file name of trained model to load")
    parser.add_argument("--train_flag", default=True, type=bool, help="bool whether model should be trained or not")
    #parser.add_argument("--learn_rate", default=1e-3, type=float, help="learning rate")
    #parser.add_argument("--weight_decay", default=1e-5, type=float, help="weight decay")
    parser.add_argument("--seed", default=32, type=int, help="random seed")
    parser.set_defaults(feature=True)
    return parser.parse_args()

if __name__ == "__main__":

    # read configuration files
    filepath = path.abspath(path.join(THIS_DIR, '..', 'config/model_config.yaml'))
    model_config = helper.read_config(filepath)
    filepath = path.abspath(path.join(THIS_DIR, '..', 'config/train_config.yaml'))
    train_config = helper.read_config(filepath)
    filepath = path.abspath(path.join(THIS_DIR, '..', 'config/data_config.yaml'))
    data_config = helper.read_config(filepath)

    # Set network models to test
    args = get_args()
    ae_model = args.ae_model
    ddn_model = args.ddn_model

    # initialise autoencoder and dynamics network configurations
    ae_config = model_config['autoencoder'][ae_model]
    ddn_config = model_config['dynamics'][ddn_model]

    channels = data_config['channels']
    seq_len = data_config['seq_len']
    im_size = data_config['im_size']
    latent_dim = args.latent_dim

    trainer = AETrainer(ae_model, ae_config, train_config, data_config, args)
    trainer.fit()