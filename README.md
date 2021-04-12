# Physics-Informed Unsupervised Representation Learning for Pixel Observations of Dynamical Systems
Jan Tiegges

Abstract too follow...

- maybe put in reference file
- for the Pixel Hamiltonian Neural Network as comparison the papers HGN and HNN have been used for construction of the network

## Project Structure

- **[Configuration Parameters](config/)**: Contains files for configurating the data set, the model parameters and the training hyperparemters.

- **[Models](models/)**: Contains the different implemented models for the autoencoder and the dynamic network.

- **[Utilities](utils/)**: Contains different classes used in the process.

- **[Data Sets](saved_data)**: Contains generated data sets.

- **[Saved Models](saved_models)**: Contains already trained models, ready to be loaded and tested.

- **[Daved Experiments](saved_experiments)**: Contains train and test logs of training process.

- **[Figures](figures)**: Contains figures created for the evaluation.

- **[Data](data.py/)**: Script for generating a data set based on different environments of the OpenAI Gym.

- **[Training/Testing](run.py)** Script for training and testing the models

- **[Evaluation](evaluate.ipynb)** Notebook for visualising the results of the models.

## Setup

pip install -r requirements.txt`

## How to run the model
[run.py](run.py) Training as well as testing the models 
In order to train the model, run
```commandline
python run.py --ae_model <name_of_autoencoder> --ddn_model <name_of_dynamic_network>
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