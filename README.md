# Physics-Informed Unsupervised Representation Learning for Pixel Observations of Dynamical Systems
Jan Tiegges

Most machine learning models today still lack a basic understanding of the environment around them, as they do not have any encoded physical knowledge. This prevents them from learning accurate representations of observed dynamical systems, especially in the context of visual observations without any domain-specific knowledge of the system. In a comprehensive literature review, this thesis first clarifies the state of the art in this young and dynamic field of research of physics-informed unsupervised representation learning. On this basis, a general model with encoded Lagrangian and Hamiltonian principles is constructed to discover the underlying governing equations of a dynamical system purely from image observations. The models are tested for their ability to encode and predict meaningful parameters and generate future images without domain-specific knowledge. A general framework consisting of an autoencoder and a dynamic network is proposed, where different types and versions are tested to discover the best autoencoder type as well as the differences that arise between Lagrangian, Hamiltonian and no priors. The results underline the advantages of models with encoded physical principles, whereby the Hamiltonian model proves to be the best. The analysis revealed the importance of symplectic integrators, the training method and other attributes, which leaves much room for further research.

## Project Structure

- **[Configuration Parameters](config/)**: Contains files for configurating the data set, the model parameters and the training hyperparemters.

- **[Models](models/)**: Contains the different implemented models for the autoencoder and the dynamic network.

- **[Utilities](utils/)**: Contains different classes used in the process.

- **[Saved Data](saved_data)**: Contains generated data sets.

- **[Saved Models](saved_models)**: Contains already trained models, ready to be loaded and tested.

- **[Saved Experiments](saved_experiments)**: Contains train and test logs of training process.

- **[Figures](figures)**: Contains figures created for the evaluation.

- **[Data](data.py/)**: Script for generating a data set based on different environments of the OpenAI Gym.

- **[Training/Testing](run.py)** Script for training and testing the models

- **[Evaluation](evaluate.ipynb)** Notebook for visualising the results of the models.

## Setup

pip install -r requirements.txt

## How to run the model
[run.py](run.py): Training as well as testing the models 
In order to train the model, run
```commandline
python run.py --ae_model <name_of_autoencoder> --ddn_model <name_of_dynamic_network>
```

Autoencoder options:
- Linear Autoencoder (LAE)
- Convolutional Autoencoder (CAE)
- Variational Autoencoder (VAE)
- Convolutional Variational Autoencoder (CVAE)

Dynamic Network options:
- Simple Fully-Connected Network (MLP)
- Lagrangian Neural Network (LNN)
- Hamitlonian Neural Network (HNN)

example:
```commandline
python run.py --ae_model CVAE --ddn_model LNN
```

All parameters to vary for the data set, the model structure and the training are to be specified in the corresponding configuration file before running the script.

```
optional arguments:
  -h, --help            shows help message
  
  --system <name_of_system>
                        Name of the dynamical system to train/test the network on
                        
  --latent_dim <size_of_latent_space>
                        Desired size of the latent space
                     
  --model_file <model_path>
                        Path to a specific trained model to be loaded
                        
  --data_file <data_path>
                        Path to a data set to be used for training                       
                        
  --train_flag True or False
                        Whether model should be trained or only tested
                        
  --seed <number>
                        Seed for generating dataset         
```

## References

The following resources have been used in constructing the code, which especially refers to the Hamiltonian model.

1. Peter Toth et al. “Hamiltonian generative networks”. In:arXiv preprint arXiv:1909.13789 (2019).

2. Sam Greydanus, Misko Dzamba and Jason Yosinski. “Hamiltonian neural net-works”. In:arXiv preprint arXiv:1906.01563 (2019).
   
3. https://www.tensorflow.org/tutorials/generative/cvae
   
4. https://github.com/AntixK/PyTorch-VAE/blob/master/models/beta_vae.py