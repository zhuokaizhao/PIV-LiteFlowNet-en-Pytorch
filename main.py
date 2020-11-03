# main functions of PIV-LiteFlowNet-en
# Author: Zhuokai Zhao

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import png
import math
import glob
import time
import copy
import torch
import timeit
import random
import imageio
import argparse
import itertools
import subprocess
import numpy as np
import pyvista as pv
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageFont, ImageDraw

import data
import model

# preferably use the non-display gpu for training
# os.environ['CUDA_VISIBLE_DEVICES']='0, 1'
# os.environ['CUDA_VISIBLE_DEVICES']='1'
# preferably use the display gpu for testing
os.environ['CUDA_VISIBLE_DEVICES']='2'

print('\n\nPython VERSION:', sys.version)
print('PyTorch VERSION:', torch.__version__)
from subprocess import call
# call(["nvcc", "--version"]) does not work
# ! nvcc --version
print('CUDNN VERSION:', torch.backends.cudnn.version())
print('Number CUDA Devices:', torch.cuda.device_count())
print('Devices')
call(['nvidia-smi', '--format=csv', '--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free'])
print('Active CUDA Device: GPU', torch.cuda.current_device())

print ('Available devices ', torch.cuda.device_count())
print ('Current cuda device ', torch.cuda.current_device())

# perform some system checks
def check_system():
    if sys.version_info.minor < 4 or sys.version_info.minor > 7:
        raise Exception('Python 3.4 - 3.7 is required')

    if int(tf.__version__.split('.')[0]) < 2:
        raise Exception('TensorFlow 2 is required')

    if not int(str('').join(torch.__version__.split('.')[0:2])) >= 13:
        raise Exception('At least PyTorch version 1.3.0 is needed')


# Print iterations progress
def print_progress_bar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)


def main():

	# input arguments
    parser = argparse.ArgumentParser()
    # mode (data, train, or test mode)
    parser.add_argument('--mode', required=True, action='store', nargs=1, dest='mode')
    # input training dataset director
    parser.add_argument('--train-dir', action='store', nargs=1, dest='train_dir')
    # input validation dataset ditectory
    parser.add_argument('--val-dir', action='store', nargs=1, dest='val_dir')
    # input testing dataset ditectory
    parser.add_argument('--test-dir', action='store', nargs=1, dest='test_dir')
    # epoch size
    parser.add_argument('-e', '--num-epoch', action='store', nargs=1, dest='num_epoch')
    # batch size
    parser.add_argument('-b', '--batch-size', action='store', nargs=1, dest='batch_size')
    # loss function
    parser.add_argument('-l', '--loss', action='store', nargs=1, dest='loss')
    # checkpoint path for continuing training
    parser.add_argument('-c', action='store', nargs=1, dest='checkpoint_path')
    # input or output model directory
    parser.add_argument('-m', '--model', action='store', nargs=1, dest='model_dir')
    # output directory (tfrecord in 'data' mode, figure in 'training' mode)
    parser.add_argument('-o', '--output-dir', action='store', nargs=1, dest='output_dir')
    # verbosity
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', default=False)
    args = parser.parse_args()

    # check the system and directory
    check_system()
    # check_directory(args)

    mode = args.mode[0]
    verbose = args.verbose
    overwrite = args.overwrite

    if mode == 'train':
        if torch.cuda.device_count() > 1:
            print('\n', torch.cuda.device_count(), 'GPUs available')
            device = torch.device('cuda')
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if verbose:
            print(f'\nmode: {mode}')

        # train and val directory
        train_dir = args.train_dir[0]
        val_dir = args.val_dir[0]

        # checkpoint path to load the model from
        if args.checkpoint_path != None:
            checkpoint_path = args.checkpoint_path[0]
        else:
            checkpoint_path = None
        # directory to save the model to
        model_dir = args.model_dir[0]
        # loss graph directory
        figs_dir = args.output_dir[0]
        # train-related parameters
        num_epoch = int(args.num_epoch[0])
        batch_size = int(args.batch_size[0])
        target_dim = 2
        if args.time_span != None:
            time_span = int(args.time_span[0])
        else:
            time_span = None
        loss = args.loss[0]

        # make sure the model_dir is valid
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
            print(f"model_dir {model_dir} did not exist, but has been created")

        # load the data
        print(f'\nLoading datasets')
        # Read data
        train_img1_name_list, train_img2_name_list, train_gt_name_list = data.read_all(train_dir)
        val_img1_name_list, val_img2_name_list, val_gt_name_list = data.read_all(val_dir)
        # construct dataset
        train_data, train_labels = data.construct_dataset(train_img1_name_list,
                                                            train_img2_name_list,
                                                            train_gt_name_list)

        val_data, val_labels = data.construct_dataset(val_img1_name_list,
                                                        val_img2_name_list,
                                                        val_gt_name_list)

        num_channels = train_data.shape[1] // 2

        if verbose:
            print(f'\nGPU usage: {device}')
            print(f'input training data dir: {train_dir}')
            print(f'input validation data dir: {val_dir}')
            print(f'input checkpoint path: {checkpoint_path}')
            print(f'output model dir: {model_dir}')
            print(f'output figures dir: {figs_dir}')
            print(f'epoch size: {num_epoch}')
            print(f'batch size: {batch_size}')
            print(f'loss function: {loss}')
            print(f'number of image channel: {num_channels}')
            print(f'train_data has shape: {train_data.shape}')
            print(f'train_labels has shape: {train_labels.shape}')
            print(f'val_data has shape: {val_data.shape}')
            print(f'val_labels has shape: {val_labels.shape}')


        # model
        piv_lfn_en = model.PIV_LiteFlowNet_en()

        # load checkpoint info if existing
        starting_epoch = 0
        if checkpoint_path != None:
            checkpoint = torch.load(checkpoint_path)
            piv_lfn_en.load_state_dict(checkpoint['state_dict'])
            starting_epoch = checkpoint['epoch']

        if torch.cuda.device_count() > 1:
            print('\nUsing', torch.cuda.device_count(), 'GPUs')
            lmsi_model = torch.nn.DataParallel(lmsi_model)

        lmsi_model.to(device)
        # define optimizer
        optimizer = torch.optim.Adam(lmsi_model.parameters(), lr=1e-4)

        if checkpoint_path != None:
            optimizer.load_state_dict(checkpoint['optimizer'])

        # training for a number of epochs
        train_start_time = time.time()
        # train/val losses for all the epochs
        all_epoch_train_losses = []
        all_epoch_val_losses = []
        for i in range(starting_epoch, starting_epoch+num_epoch):
            print(f'\n Starting epoch {i+1}/{starting_epoch+num_epoch}')
            epoch_start_time = time.time()

            # train/val losses for all the batches
            all_batch_train_losses = []
            all_batch_val_losses = []

            # define loss
            if loss == 'MSE' or loss == 'RMSE':
                loss_module = torch.nn.MSELoss()
            elif loss == 'MAE':
                loss_module = torch.nn.L1Loss()
            else:
                raise Exception(f'Unrecognized loss function: {loss}')

            # assume training data has shape (755, 2, 256, 256)
            # which corresponds to each image pair
            # have a data loader that select the image pair
            for phase in ['train', 'val']:
                # dataloader randomly loads a batch_size number of image pairs
                if phase == 'train':
                    data = torch.utils.data.TensorDataset(train_data, train_labels)
                    dataloader = torch.utils.data.DataLoader(data,
                                                            batch_size=batch_size,
                                                            shuffle=True,
                                                            num_workers=4)
                    # number for batches used to plot progress bar
                    num_batch = int(np.ceil(len(train_data) / batch_size))
                elif phase == 'val':
                    data = torch.utils.data.TensorDataset(val_data, val_labels)
                    dataloader = torch.utils.data.DataLoader(data,
                                                            batch_size=batch_size,
                                                            shuffle=True,
                                                            num_workers=4)
                    # number for batches used to plot progress bar
                    num_batch = int(np.ceil(len(val_data) / batch_size))

                for j, (batch_data, batch_labels) in enumerate(dataloader):
                    # has shape (batch_size, 260, 1, 128, 128)
                    batch_start_time = time.time()
                    # send data to GPU
                    batch_data = batch_data.to(device)
                    batch_labels = batch_labels.to(device)

                    if phase == 'train':
                        # train/validate
                        cur_label_pred = lmsi_model(batch_data)

                        # compute loss
                        train_loss = loss_module(cur_label_pred, batch_labels)
                        if loss == 'RMSE':
                            train_loss = torch.sqrt(train_loss)

                        # Before the backward pass, use the optimizer object to zero all of the
                        # gradients for the variables it will update
                        optimizer.zero_grad()

                        # Backward pass: compute gradient of the loss with respect to model parameters
                        train_loss.backward()

                        # update to the model parameters
                        optimizer.step()

                        # save the loss
                        all_batch_train_losses.append(train_loss.detach().item())

                        # batch end time
                        batch_end_time = time.time()
                        batch_time_cost = batch_end_time - batch_start_time

                        # show mini-batch progress
                        print_progress_bar(iteration=j+1,
                                            total=num_batch,
                                            prefix=f'Batch {j+1}/{num_batch},',
                                            suffix='%s loss: %.3f, time: %.2f' % (phase+' '+loss, all_batch_train_losses[-1], batch_time_cost),
                                            length=50)

                    elif phase == 'val':
                        lmsi_model.eval()

                        with torch.no_grad():

                            # train/validate
                            cur_label_pred = lmsi_model(batch_data)

                            # compute loss
                            val_loss = loss_module(cur_label_pred, batch_labels)
                            if loss == 'RMSE':
                                val_loss = torch.sqrt(val_loss)

                            # save the loss
                            all_batch_val_losses.append(val_loss.detach().item())

                        # batch end time
                        batch_end_time = time.time()
                        batch_time_cost = batch_end_time - batch_start_time

                        # show mini-batch progress
                        print_progress_bar(iteration=j+1,
                                            total=num_batch,
                                            prefix=f'Batch {j+1}/{num_batch},',
                                            suffix='%s loss: %.3f, time: %.2f' % (phase+' '+loss, all_batch_val_losses[-1], batch_time_cost),
                                            length=50)

                print('\n')

        epoch_end_time = time.time()
        batch_avg_train_loss = np.mean(all_batch_train_losses)
        batch_avg_val_loss = np.mean(all_batch_val_losses)
        all_epoch_train_losses.append(batch_avg_train_loss)
        all_epoch_val_losses.append(batch_avg_val_loss)
        print('\nEpoch %d completed in %.3f seconds, avg train loss: %.3f, avg val loss: %.3f'
                    % ((i+1), (epoch_end_time-epoch_start_time), all_epoch_train_losses[-1], all_epoch_val_losses[-1]))

    train_end_time = time.time()
    print('\nTraining completed in %.3f seconds' % (train_end_time-train_start_time))

    # save loss graph
    if checkpoint_path != None:
        prev_train_losses = checkpoint['train_loss']
        prev_val_losses = checkpoint['val_loss']
        all_epoch_train_losses = prev_train_losses + all_epoch_train_losses
        all_epoch_val_losses = prev_val_losses + all_epoch_val_losses

    plt.plot(all_epoch_train_losses, label='Train')
    plt.plot(all_epoch_val_losses, label='Validation')
    plt.title(f'Training and validation loss on PIV-LiteFlowNet-en model')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    loss_path = os.path.join(figs_dir, f'piv_lfn_en_loss_{starting_epoch+num_epoch}.png')
    plt.savefig(loss_path)
    print(f'\nLoss graph has been saved to {loss_path}')

    # save model as a checkpoint so further training could be resumed
    model_path = os.path.join(model_dir, f'PIV-LiteFlowNet-en_{starting_epoch+num_epoch}.pt')
    # if trained on multiple GPU's, store model.module.state_dict()
    if torch.cuda.device_count() > 1:
        model_checkpoint = {
                                'epoch': starting_epoch+num_epoch,
                                'state_dict': lmsi_model.module.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'train_loss': all_epoch_train_losses,
                                'val_loss': all_epoch_val_losses
                            }
    else:
        model_checkpoint = {
                                'epoch': starting_epoch+num_epoch,
                                'state_dict': lmsi_model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'train_loss': all_epoch_train_losses,
                                'val_loss': all_epoch_val_losses
                            }

    torch.save(model_checkpoint, model_path)
    print(f'\nTrained model/checkpoint has been saved to {model_path}\n')



if __name__ == "__main__":
    main()