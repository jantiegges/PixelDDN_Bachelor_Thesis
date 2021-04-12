import numpy as np
import gym
from PIL import Image
from utils.file_manager import pickle_load, pickle_save

# file for generating the dataset for training and testing

def get_dataset(data_params, system_name, save_dir, filename, seed):
    """ Returns a dateset build on an OpenAI Gym environment. Checks for saved one first,
    otherwise constructs one
    Params:
        data_params (dict): dictionary with all parameters for the data generation
        system_name (str): name of dynamical system
        save_dir (str): path where to save saved_data
    Return:
        saved_data (dict): dataset with train and test saved_data
    """
    if system_name == "pendulum":
        env_name = "Pendulum-v0"
    elif system_name == "double_pendulum":
        env_name = "Acrobat-v1"
    elif system_name == "cartpole":
        env_name = "CartPole-v1"
    else:
        raise Exception("Invalid Dynamical System Environment")

    # set path for saved_data
    path_data = "{}/{}.pkl".format(save_dir, filename)
    path_config = "{}/{}_config".format(save_dir, filename)

    # try to load saved dataset and create new one if it doesn't exist
    try:
        data = pickle_load(path_data)
        print("Successfully loaded saved_data from path {}".format(path_data))
        data_params = pickle_load(path_config)
    except:
        print("No existing dataset at path {}. New dataset will be constructed.".format(path_data))

        episodes = data_params['episodes']
        test_split = data_params['test_split']

        train_data, test_data, train_meta_data, test_meta_data, gym_settings = \
            create_dataset(data_params, env_name, seed, episodes, test_split)

        data = {'train_data': train_data, 'test_data': test_data,
                'train_meta_data': train_meta_data, 'test_meta_data': test_meta_data,
                'settings': gym_settings}

        pickle_save(data, path_data)
        pickle_save(data_params, path_config)

    return data, data_params


def create_dataset(data_params, env_name, seed, episodes, test_split):
    """ Creates a dateset with observations from OpenAI Gym environment
    Params:
        env_name (str): name of OpenAI environment
        seed (int): seed for reproducing results
        episodes (int): number of episodes to run the environment
        test_split (float): percentage number of the test split saved_data
    Returns:
        train_data [ep x ts x seq_len+1 x H x W x channels]: last image of sequence is target image
        test_data [ep x ts x seq_len+1 x H x W x channels]: last image of sequence is target image
        train_meta_data/test_meta_data(dict):
            'ccoords_in' [ep x ts x 2]: (q,p) of last input image
            'ccoords_out' [ep x ts x 2]: (q,p) of target image
            'gcoords_in' [ep x ts x 2]: (q,q_dot) of last input image
            'gcoords_out' [ep x ts x 2]: (q,q_dot) of target image
            'applied_force' [ep x ts x 1] : force applied to last input image
            'episodes': number of episodes
            'timesteps': number of timesteps
        settings (dict): gym_settings
    """

    timesteps = data_params['timesteps']
    im_size = data_params['im_size']
    seq_len = data_params['seq_len']
    pred_len = data_params['pred_len']
    channels = data_params['channels']

    # running OpenAI gym environment and save pixel observations, states and settings
    frames, canonical_coords, generalized_coords, forces, gym_settings = \
        run_system(data_params, env_name, seed, episodes, timesteps, im_size, seq_len, channels)

    pixel = []
    can_coords_in, can_coords_out = [], []
    gen_coords_in, gen_coords_out = [], []
    applied_force = []

    # set length of data
    data_len = timesteps - seq_len - pred_len
    for ep in range(episodes):
        pixel_tmp = []
        can_coords_in_tmp, can_coords_out_tmp = [], []
        gen_coords_in_tmp, gen_coords_out_tmp = [], []
        applied_force_tmp = []

        for i in range(data_len):
            # append sequences with length seq_len + pred_len
            # the last ones (pred_len) represents the target image(s) later
            #pixel_tmp.append(frames[ep][i:i+seq_len+1])
            pixel_tmp.append(frames[ep][i:i + seq_len + pred_len])
            # i+seq_len-1 corresponds to the last image before the target image(s)
            can_coords_in_tmp.append(canonical_coords[ep][i+seq_len-1])
            # i+seq_len: corresponfs to the target image(s)
            #can_coords_out_tmp.append(canonical_coords[ep][i+seq_len])
            can_coords_out_tmp.append(canonical_coords[ep][i + seq_len:])
            gen_coords_in_tmp.append(generalized_coords[ep][i+seq_len-1])
            #gen_coords_out_tmp.append(generalized_coords[ep][i+seq_len])
            gen_coords_out_tmp.append(generalized_coords[ep][i + seq_len:])
            applied_force_tmp.append(forces[ep][i+seq_len-1])

        pixel.append(pixel_tmp)
        can_coords_in.append(can_coords_in_tmp)
        can_coords_out.append(can_coords_out_tmp)
        gen_coords_in.append(gen_coords_in_tmp)
        gen_coords_out.append(gen_coords_out_tmp)
        applied_force.append(applied_force_tmp)

    pixel = np.asarray(pixel)
    can_coords_in = np.asarray(can_coords_in)
    can_coords_out = np.asarray(can_coords_out)
    gen_coords_in = np.asarray(gen_coords_in)
    gen_coords_out = np.asarray(gen_coords_out)
    applied_force = np.asarray(applied_force)

    # split in train and test saved_data
    test_idx = int(episodes * (1-test_split))

    train_data = pixel[:test_idx]
    train_ccoords_in, train_ccoords_out = can_coords_in[:test_idx], can_coords_out[:test_idx]
    train_gcoords_in, train_gcoords_out = gen_coords_in[:test_idx], gen_coords_out[:test_idx]
    train_applied_force = applied_force[:test_idx]

    test_data = pixel[test_idx:]
    test_ccoords_in, test_ccoords_out = can_coords_in[test_idx:], can_coords_out[test_idx:]
    test_gcoords_in, test_gcoords_out = gen_coords_in[test_idx:], gen_coords_out[test_idx:]
    test_applied_force = applied_force[test_idx:]

    # save lists in saved_data structure

    train_meta_data = {'ccoords_in': train_ccoords_in, 'ccoords_out': train_ccoords_out,
                       'gcoords_in': train_gcoords_in, 'gcoords_out': train_gcoords_out,
                       'applied_force': train_applied_force, 'episodes': test_idx,
                       'timesteps': timesteps-seq_len}

    test_meta_data = {'ccoords_in': test_ccoords_in, 'ccoords_out': test_ccoords_out,
                      'gcoords_in': test_gcoords_in, 'gcoords_out': test_gcoords_out,
                      'applied_force': test_applied_force, 'episodes': episodes - test_idx,
                      'timesteps': timesteps-seq_len}

    return train_data, test_data, train_meta_data, test_meta_data, gym_settings


def run_system(data_params, env_name, seed, episodes, timesteps, im_size, seq_len, channels):
    """ Runs OpenAI environment
    Params:
        env_name (str): name of OpenAI environment
        seed (int): seed for reproducing results
        timesteps (int): number of action steps in environment before terminating
        episodes (int): number of episodes to run the environment
        im_size (int): desired size of the input image
        seq_len (int): number of images in input sequence
        channels (int): number of channels of the input images (RGB: 3, BW: 1)
    Return:
        frames ([#frames, pixel]]: consecutive array of the pixel frames
        canonical_coords ([#frames, coords]): consecutive array of can. coords for each frame
        gym_settings (dict): current local symbol table
    """
    # updates and returns a dictionary of the current local arguments
    gym_settings = locals()

    # create environment
    print("Running OpenAI Environment {}".format(env_name))
    env = gym.make(env_name)
    env.seed(seed)
    env.reset()

    frames =[]
    canonical_coords = []
    generalized_coords = []
    forces = []

    # run environment
    for i_episode in range(episodes):
        state = env.reset()
        frames_tmp = []
        canonical_coords_tmp = []
        generalized_coords_tmp = []
        forces_tmp = []

        for step in range(timesteps):
            # save frames of observation
            frames_tmp.append(preproc(env.render('rgb_array'), im_size, channels, env_name))
            # TODO: for now no action on system, just let it swing, that's why 0 is given.
            #force = env.action_space.sample()
            force = [0.0]
            forces_tmp.append(force)
            # saves obs (we don't need the rest)
            state,_,_,_ = env.step(force)
            canonical_coords_tmp.append(get_canonical_coords(state, env_name))
            generalized_coords_tmp.append(get_generalized_coords(state, env_name))

        frames.append(frames_tmp)
        canonical_coords.append(canonical_coords_tmp)
        generalized_coords.append(generalized_coords_tmp)
        forces.append(forces_tmp)

    # finished running
    env.close()
    # convert lists to arrays (-1 stands for unknown dim and np figures it out)
    canonical_coords = np.asarray(canonical_coords)
    generalized_coords = np.asarray(generalized_coords)
    forces = np.asarray(forces)
    frames = np.asarray(frames)

    return frames, canonical_coords, generalized_coords, forces, gym_settings


def preproc(img, im_size, channels, env_name):
    """ turns gray, crops and resize the rgb pendulum observation
    Params:
        Img (array[int]): rgb array of pendulum observation image
        im_size (int): size of desired image size
        env_name (str): name of environment
    Return:
        Img (array (HxWxC)): pre processed image
    """
    # may need to rewrite crop operation depending on screen size
    # tested on MacBook Pro 2018 with resolution 1680x1050
    if env_name == "Pendulum-v0":
        # turn from rgb to grayscale and remove direction arrow
        if channels == 1:
            img = np.subtract(img[...,0], img[...,1])
        # crop image
        img = img[212:-212, 212:-212]
        # turn pixel value into value btw 0 and 1
        img = img / 255.0
        # resizes the image and save as Image object
        if channels == 1:
            img = Image.fromarray(img).resize((im_size, im_size))
        else:
            img = Image.fromarray(img, 'RGB').resize((im_size, im_size))
        img = np.asarray(img)
        # add bw color channel
        if channels == 1:
            img = img[..., np.newaxis]
        return img
    if env_name == "Acrobat-v1":
        raise NotImplementedError
    if env_name == "CartPole-v1":
        # turn from rgb to grayscale
        if channels == 1:
            img = np.subtract(img[...,0], img[...,1])
        # crop image
        # TODO: adjust cropping for cartpole
        #img = img[212:-212, 212:-212]
        # turn pixel value into value btw 0 and 1
        img = img / 255
        # resizes the image and save as Image object
        img = Image.fromarray(img).resize((im_size, im_size))
        img = np.asarray(img)
        # add bw color channel
        if channels == 1:
            img = img[..., np.newaxis]
        return img
    else:
        raise NotImplementedError

def get_generalized_coords(state, env_name):
    """ function for getting generalized coordinates from state coordinates
    Params:
        state (array[float]): state coordinates
        env_name (str): name of OpenAI environment
    Return:
        canonical_coords (array[float]): array with canonical coordinates (position and velocity)
    """
    if env_name == "Pendulum-v0":
        theta = np.arccos(state[0])
        theta_dot = state[2]

        q = theta
        q_dot = theta_dot

        generalized_coords = np.array([q, q_dot])
        return generalized_coords

    elif env_name == "Acrobat-v1":
        theta1 = np.arccos(state[0])
        theta2 = np.arccos(state[2])
        theta_dot1 = state[4]
        theta_dot2 = state[5]

        q = np.array([theta1, theta2])
        q_dot = np.array([theta_dot1, theta_dot2])

        generalized_coords = np.array([q, q_dot])
        return generalized_coords

    elif env_name == "CartPole-v1":
        x = state[0]
        x_dot = state[1]
        theta = state[2]
        theta_dot = state[3]

        q = np.array([x, theta])
        q_dot = np.array([x_dot, theta_dot])

        generalized_coords = np.array([q, q_dot])
        return generalized_coords

    else:
        raise NotImplementedError


def get_canonical_coords(state, env_name):
    """ function for getting canonical coordinates from state coordinates
    Params:
        state (array[float]): state coordinates
        env_name (str): name of OpenAI environment
    Return:
        canonical_coords (array[float]): array with canonical coordinates (position and momentum)
    """
    if env_name == "Pendulum-v0":
        # information from https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
        mass = 1.0
        theta = np.arccos(state[0])
        theta_dot = state[2]

        # compute momentum: p = m*v
        q = theta
        p = mass * theta_dot

        canonical_coords = np.array([q, p])
        return canonical_coords

    elif env_name == "Acrobat-v1":
        # information from https://github.com/openai/gym/blob/master/gym/envs/classic_control/acrobot.py
        masspole1 = 1.0
        masspole2 = 1.0
        theta1 = state[0]
        theta2 = state[1]
        theta_dot1 = state[2]
        theta_dot2 = state[3]

        # compute momentum: p = m*v
        q = np.array([theta1, theta2])
        p = np.array([(masspole1*theta_dot1), (masspole2*theta_dot2)])

        canonical_coords = np.array([q, p])
        return canonical_coords

    elif env_name == "CartPole-v1":
        # information from https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
        masscart = 1.0
        masspole = 0.1
        x = state[0]
        x_dot = state[1]
        theta = state[2]
        theta_dot = state[3]

        # compute momentum: p = m*v
        q = np.array([x, theta])
        p = np.array([(masscart*x_dot), (masspole*theta_dot)])

        canonical_coords = np.array([q, p])
        return canonical_coords
    else:
        raise NotImplementedError