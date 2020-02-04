import os

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, ImageFolder
from torchvision.utils import save_image
import numpy as np
from tensorboardX import SummaryWriter
import random
from numpy.random import rand
import cv2
from Masks import Masks
from cv2 import imread
import glob
from itertools import repeat

from autoencoder_residual_relu import autoencoder
from discriminator import Discriminator
import pytorch_ssim
from img_operations import min_max_normalization, compose_2x2_image, compose_2x1_image, compose_3x2_image, compose_4x2_image, diff, toGray, img_transform
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
from functools import reduce
import time
from math import log10

from data_ops import load_data_mul
from model_runs import model_runs, dataset_folders_train


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
print("Start00")




num_epochs = 200
batch_size = 16
learning_rate = 2e-3  #*batch_size#1e-3

validation_rate = 0.05
#GAN = True
#validate_defects = False
#inpaint = True
#=======Saving Parameters=========
resume_training = False
model_save_path = './model_saves/'

model_file_ending = "model.pb"
training_file_ending = 'training.tar'



#==============Losses=================
criterion = nn.BCELoss()
ssim_loss_big = pytorch_ssim.SSIM(window_size=31)
ssim_loss_test = pytorch_ssim.SSIM(window_size=11, size_average=False)
ssim_loss_small = pytorch_ssim.SSIM(window_size=11)

def feed_forward(model, data, ssim_loss_function, masks = None):


    img_raw = data[0]
    img = Variable(img_raw).cuda()
    if model.inpaint:
        masks = Variable(masks).cuda()
        masked_img = img * (1 - masks)
        input_img = masked_img  # torch.cat((masked_img, masks), dim=1)
    else:
        input_img = img
    # ===================forward=====================
    output = model(input_img)
    ssim_loss = 1 - ssim_loss_function(output, img)
    MSE_loss = nn.MSELoss()(output, img)

    if model.inpaint:
        data = {"img": img, "masks": masks, "output": output}
    else:
        data = {"img": img, "output": output}
    loss = {"mse": MSE_loss, "ssim": ssim_loss}

    return data, loss


def average(lst):
    return reduce(lambda a, b: a + b, lst) / len(lst)

def validate(model, validation, writer, index, name):

    ssim_losses = []
    MSE_losses = []

    if model.inpaint:
        data_enumerator = enumerate(zip(validation["data"], validation["masks"]))
    else:
        data_enumerator = enumerate(validation["data"])

    for i, data in data_enumerator:
        if model.inpaint:
            (data, masks) = data
            data, loss = feed_forward(model, data, ssim_loss_small, masks)
        else:
            data, loss = feed_forward(model, data, ssim_loss_small)
        ssim_losses.append(loss["ssim"].item())
        MSE_losses.append(loss["mse"].item())

    log_images(data, writer, name+" Validation ", index, model.inpaint)

    avg_ssim, avg_MSE = average(ssim_losses), average(MSE_losses)

    writer.add_scalar(name + '_loss/ssim_loss', avg_ssim, index)
    writer.add_scalar(name + '_loss/MSE_loss', avg_MSE, index)
    writer.add_scalar(name + '_loss/ssim_loss_log', log10(avg_ssim), index)
    writer.add_scalar(name + '_loss/MSE_loss_log', log10(avg_MSE), index)

    return avg_ssim, avg_MSE



def log_images(data, writer, name, index, inpaint):
    img_raw = data["img"]
    output = data["output"]
    if inpaint:
        masks = data["masks"]
        masked_img = img_raw * (1 - masks)
        masked_output = (output * masks + masked_img)

    diff_whole = diff(img_raw, output)

    ssim_whole = 1 - ssim_loss_test(img_raw, output)
    ssim_whole=toGray(ssim_whole)
    if inpaint:
        diff_masked = masks * diff_whole
        ssim_masked = masks * ssim_whole

        log_images = compose_4x2_image(img_raw, masked_img, diff_whole, ssim_whole, masked_output, output, diff_masked, ssim_masked)
        log_image_array = compose_2x1_image(log_images[0], log_images[1])
    else:
        log_images = compose_2x2_image(img_raw, diff_whole, output, ssim_whole)
        log_image_array = compose_2x1_image(log_images[0], log_images[1])

    writer.add_image(name,
                     log_image_array, index)


def log_validation(validation_ssim, validation_MSE, epoch, epochs):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.subplots_adjust(hspace=0.5)
    ax1.plot(range(len(validation_ssim)), validation_ssim, 'g')
    ax1.set_title("SSIM Loss")
    ax2.plot(range(len(validation_MSE)), validation_MSE, 'r')
    ax2.set_title("MSE Loss")
    writer.add_figure("Loss", fig)



class circle_mem:
    def __init__(self, items, batch_size):
        self.items = items
        self.len = len(items)
        self.batch_size = batch_size

    def get(self, index):
        return self.items[index % self.len]

    def get_batch(self, index):
        index1 = index % self.len
        index2 = (index + batch_size) % self.len
        if (index1 > index2):
            return self.items[index1:] + self.items[:index2]
        else:
            return self.items[index1:index2]


def split_dataset(dataset, validation_split_ratio, params=None, val_params=None):
    dataset_len = len(dataset)
    indices = list(range(dataset_len))

    # Randomly splitting indices:
    val_len = int(np.floor(validation_split_ratio * dataset_len))
    validation_idx = np.random.choice(indices, size=val_len, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    # Contiguous split
    # train_idx, validation_idx = indices[split:], indices[:split]

    ## Defining the samplers for each phase based on the random indices:
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    train_loader = torch.utils.data.DataLoader(dataset, sampler=train_sampler, **params)
    validation_loader = torch.utils.data.DataLoader(dataset, sampler=validation_sampler, **val_params)
    data_loaders = {"train": train_loader, "val": validation_loader}
    data_lengths = {"train": len(train_idx), "val": val_len}
    return data_loaders, data_lengths


def load_data(root, validation_rate, params, val_params=None, shuffle=True):

    dataset = ImageFolder(root, transform=img_transform, is_valid_file=None)  # , loader=imread)

    data_loaders, data_lengths = split_dataset(dataset, validation_rate, params, val_params)



    train_mask_dataset = Masks(next(iter(data_loaders["train"]))[0], 40, length=data_lengths["train"])
    train_mask_loader = DataLoader(train_mask_dataset, shuffle=True, **params)

    val_mask_dataset = Masks(next(iter(data_loaders["val"]))[0], 40, length=data_lengths["val"])
    val_mask_loader = DataLoader(val_mask_dataset, shuffle=True, **val_params)

    train = {"data": data_loaders["train"], "masks": train_mask_loader, "length": data_lengths["train"]}
    val = {"data": data_loaders["val"], "masks": val_mask_loader, "length": data_lengths["train"]}

    return train, val

def save_model(model, optimizer, model_path, training_path, epoch):
    torch.save(model, model_path)

    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    torch.save(state,  training_path)



def train_discriminator(model, criterion, data, is_real_data, random_noise = 0, retain_graph=False):
    output = model(data)
    size = output.size()[0]
    if is_real_data:
        target = torch.zeros(size).cuda()
        target += torch.rand(size).cuda()*random_noise
    else:
        target = torch.ones(size).cuda()
        target -= torch.rand(size).cuda()*random_noise

    error = criterion(output, target)
    error.backward(retain_graph=retain_graph)
    return error.item()

def train_generator(generator, discriminator, criterion, img_raw, masks=None):
    if generator.inpaint:
        masks = masks.cuda()
        input_img = img_raw * (1 - masks)
    else:
        input_img = img_raw
    # ===================forward=====================
    output = generator(input_img)
    size = output.size()[0]
    decision = discriminator(output)
    error = criterion(decision, torch.zeros(size).cuda())
    error.backward()
    return error.item()

def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def add_noise(img, weight=0):
    noise1 = torch.rand(img.shape).cuda()*weight
    noise2 = torch.rand(img.shape).cuda()*weight
    noise1 = (1-img)*noise1
    noise2 = img * noise2
    noise = noise1-noise2
    return img+noise

params = {'batch_size': batch_size,
          'num_workers': 6}

val_params = {'batch_size': 64,
          'num_workers': 6}



def train(root, inpaint, GAN, name, model_path, training_path, learning_rate, batch_size):
    print("Loading train and validation data")
    train, validation = load_data(root, validation_rate, params, val_params)
    print("Loading done")
    #==============Init Model==========
    model = autoencoder(3, inpaint).cuda()
    a_optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5)
    start_epoch = 0
    index = 0
    #============Load pretrained model if True===========
    if resume_training:
        checkpoint = torch.load(generator_training_path)
        model.load_state_dict(checkpoint['state_dict'])
        a_optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        index = epoch * train["length"] // (log_interval * batch_size)

    #============GAN==========
    if GAN:
        discriminator = Discriminator(3).cuda()
        gan_learning_rate = learning_rate / 10
        d_optimizer = torch.optim.Adam(
            discriminator.parameters(), lr=gan_learning_rate, weight_decay=1e-5)
        g_optimizer = torch.optim.Adam(
            model.parameters(), lr=gan_learning_rate, weight_decay=1e-5)

    log_interval = 64
    epochs = range(start_epoch, num_epochs)

    writer = SummaryWriter()
    writer.add_scalar('hyperparamters/learning rate', learning_rate, 0)
    for epoch in epochs:
        epoch_time = time.time()
        print("epoch nr {}".format(epoch))
        if inpaint:
            data_enumerator = enumerate(zip(train["data"], train["masks"]))
        else:
            data_enumerator = enumerate(train["data"])

        t0 = time.time()
        gan_condition = GAN and epoch >= 0
        for i, data in data_enumerator:
            if inpaint:
                (data, masks) = data
                data, loss = feed_forward(model, data, ssim_loss_small, masks)
            else:
                data, loss = feed_forward(model, data, ssim_loss_small)
            MSE_loss = loss["mse"]
            ssim_loss = loss["ssim"]
            img = data["img"]
            output = data["output"]
            # ===================backward====================
            a_optimizer.zero_grad()
            ssim_loss.backward()
            a_optimizer.step()
            fool_loss=0
            if gan_condition:
                if epoch < 100:
                    random_label_noise = 0.4
                elif epoch < 150:
                    random_label_noise = 0.2
                else:
                    random_label_noise = 0
                if epoch < 50:
                    random_img_noise = 0.5
                elif epoch < 100:
                    random_img_noise = 0.2
                else:
                    random_img_noise = 0


                d_optimizer.zero_grad()
                d_error_real = train_discriminator(discriminator, criterion, add_noise(img, random_img_noise), True, random_noise=random_label_noise, retain_graph=False)

                d_error_generated = train_discriminator(discriminator, criterion, add_noise(output, random_img_noise).detach(), False, random_noise=random_label_noise, retain_graph=True)
                d_optimizer.step()

                g_optimizer.zero_grad()
                if inpaint:
                    fool_loss = train_generator(model, discriminator, criterion, img, masks)
                else:
                    fool_loss = train_generator(model, discriminator, criterion, img)
                g_optimizer.step()


            # ====================Logging=====================
            if i  % log_interval == 0:
                print('log iteration {}, loss:{:.4f}, MSE_loss:{:.4f}, {:.4f} seconds per batch, {:.4f} seconds per log'.format(index, ssim_loss, MSE_loss, (time.time() - t0)/log_interval, (time.time() - t0)))
                
                if gan_condition:
                    print('discriminator error real {}, discriminator error gen:{:.4f}, generator error:{:.4f}'.format(d_error_real, d_error_generated, fool_loss))
                t0 = time.time()
                validate(model, validation, writer, index, name)
                if gan_condition:
                    writer.add_scalar('GAN/g_loss', fool_loss, index)
                    writer.add_scalar('GAN/d_loss_real', d_error_real, index)
                    writer.add_scalar('GAN/d_loss_fake', d_error_generated, index)
                    writer.add_image('GAN_image/real_noise', add_noise(img, random_img_noise)[0], index)
                    writer.add_image('GAN_image/real', img[0], index)
                    writer.add_image('GAN_image/fake_noise', add_noise(output, random_img_noise)[0], index)
                    writer.add_image('GAN_image/fake', output[0], index)
                if index%40==0 and index!=0:
                    print("Scaling learning rate", 0.56)
                    learning_rate = learning_rate * 0.56
                    adjust_learning_rate(a_optimizer, learning_rate)
                    if gan_condition:
                        gan_learning_rate = gan_learning_rate * 0.56
                        adjust_learning_rate(d_optimizer, gan_learning_rate)
                        adjust_learning_rate(g_optimizer, gan_learning_rate)
                    writer.add_scalar('hyperparamters/learning rate', learning_rate, index)


                index += 1

        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}, MSE_loss:{:.4f}, Time per epoch:{:.4f}'
              .format(epoch + 1, num_epochs, ssim_loss, MSE_loss, time.time() - epoch_time))

        save_model(model, a_optimizer, model_path, training_path, epoch)
    writer.close()

def run_all_models(dataset_folders, runs):
    for run in runs:
        root = dataset_folders[run["object"]]
        inpaint = run["inpaint"]
        GAN = run["GAN"]
        name = run["object"]
        if GAN: name += "_GAN"
        if inpaint: name += "_inpaint"

        model_path = model_save_path +"/inference/" + name + "_" + model_file_ending
        training_path = model_save_path + name + "_" + training_file_ending
        train(root, inpaint, GAN, name, model_path, training_path, learning_rate, batch_size)

def test_train(dataset_folders, runs):
    run = runs[0]
    root = dataset_folders[run["object"]]
    inpaint = run["inpaint"]
    GAN = run["GAN"]
    name = run["object"]

    model_path = model_save_path + name + "_" + model_file_ending
    training_path = model_save_path + name + "_" + training_file_ending
    train(root, inpaint, GAN, name, model_path, training_path, learning_rate, batch_size)


#test_train(dataset_folders_train, model_runs)
run_all_models(dataset_folders_train, model_runs)