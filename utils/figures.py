import numpy as np
import os
import imageio, shutil
from PIL import Image
import matplotlib.pyplot as plt

# This class is used for the generation of figures illustrating the model output

def make_gif(frames, save_dir, name, duration=1e-1, pixels=None):
    """Given a three dimensional array [frames, height, width], make
    a gif and save it
    Params:
        frames (list): contains all frames
        save_dir (string): path where figure is to be saved
        name (string): name of constructed figure
        duration (float): time step between frames
        pixels (list[int]): height and width of figure
    Returns:
        png_save_path (string): path where the figure has been saved
    """
    temp_dir = './_temp'
    os.mkdir(temp_dir) if not os.path.exists(temp_dir) else None
    images = []
    for i in range(len(frames)):
        im = np.asarray((frames[i].clip(-.5,.5) + .5)*255)
        im = im.astype(np.uint8)
        im = Image.fromarray(im, mode="L")
        if pixels is not None:
            im = im.resize(size=pixels)
        images.append(im)
        # save temporary images to be later sticked together
        im.save(temp_dir + "/f_{:04d}.png".format(i))

    save_path = '{}/{}.gif'.format(save_dir, name)
    png_save_path = '{}.png'.format(save_path)
    imageio.mimsave(save_path, images, duration=duration)
    os.rename(save_path, png_save_path)

    shutil.rmtree(temp_dir) # remove all the images
    return png_save_path

def plot_pred_vs_target(input, target, pred, im_size):
    """ function for plotting input, target and predicted images for comparison during training
    Params:
        input (Tensor): sequence of input images
        target (Tensor): sequence of target images
        pred (Tensor) sequence of predicted images
        im_size (int): height/width of output images
    Returns:
        fig: plot of all images with labels
    """

    seq_len = input.shape[0]
    pred_len = pred.shape[0]
    fig = plt.figure(figsize=(12,12))
    ax = []
    idx = 1

    # plot input
    for i in range(seq_len):
        ax.append(fig.add_subplot(3, pred_len, idx))
        ax[-1].set_title(f"input img{i}")
        img = input[i].detach().numpy().reshape(-1,im_size)
        plt.imshow(img)
        idx += 1
    
    idx = pred_len + 1

    # plot target
    for i in range(pred_len):
        ax.append(fig.add_subplot(3, pred_len, idx))
        ax[-1].set_title(f"target img{i}")
        img = target[i].detach().numpy().reshape(-1,im_size)
        plt.imshow(img)
        idx += 1

    # plot prediction
    for i in range(pred_len):
        ax.append(fig.add_subplot(3, pred_len, idx))
        ax[-1].set_title(f"pred img{i}")
        img = pred[i].detach().numpy().reshape(-1,im_size)
        plt.imshow(img)
        idx += 1

    return fig