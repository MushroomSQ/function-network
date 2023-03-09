from collections import OrderedDict
from tqdm import tqdm
from dataset import get_dataloader
from config import get_config
from util.utils import cycle
from agent import get_agent
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # create experiment config
    config = get_config('pqnet')('train')

    # create network and training agent
    tr_agent = get_agent(config)

    # load from checkpoint if provided
    if config.cont:
        tr_agent.load_ckpt(config.ckpt)

    # create dataloader
    train_loader = get_dataloader('train', config)
    val_loader = get_dataloader('val', config)
#    val_loader = cycle(val_loader)

    # start training
    clock = tr_agent.clock

    # save and load
    train_loss = []
    train_path = config.log_dir + "/losses.txt"
    if os.path.exists(train_path):
        with open(train_path, 'r') as f:
            for line in f:
                train_loss.append(float(line.strip()))
    test_loss = []
    test_path = config.log_dir + "/test_loss.txt"
    if os.path.exists(test_path):
        with open(test_path, 'r') as f:
            for line in f:
                test_loss.append(float(line.strip()))
    correct_all = []
    correct_path = config.log_dir + "/correct.txt"
    if os.path.exists(correct_path):
        with open(correct_path, 'r') as f:
            for line in f:
                correct_all.append(float(line.strip()))

    for e in range(clock.epoch, config.nr_epochs):
        # begin iteration
        total_losses = torch.tensor(0).float().cuda()
        test_total = torch.tensor(0).float().cuda()
        pbar = tqdm(train_loader)
        for b, data in enumerate(pbar):
            # train step
            outputs, losses = tr_agent.train_func(data)

            for k, v in losses.items():
                total_losses += v

            pbar.set_description("EPOCH[{}][{}]".format(e, b))
            pbar.set_postfix(OrderedDict({k: v.item() for k, v in losses.items()}))

            clock.tick()

        # save losses
        total_losses = float(total_losses.cpu()/len(pbar))
        losses_path = config.log_dir + "/losses.txt"
        with open(losses_path, 'a') as f:
            f.write(str(total_losses))
            f.write("\n")
        train_loss.append(total_losses)

        # update lr by scheduler
        tr_agent.update_learning_rate()

        if config.module == 'part_ae':
            with torch.no_grad():
                corrects = []
                pbar = tqdm(val_loader)
                for b,data in enumerate(pbar):
                    outputs, losses = tr_agent.val_func(data)
                    for k, v in losses.items():
                        test_total += v
                    pbar.set_description("EPOCH[{}][{}]".format(e, b))
                    pbar.set_postfix(OrderedDict({k: v.item() for k, v in losses.items()}))
                    for j in range(len(outputs)): 
                        output = outputs[j].detach().cpu().numpy()
                        dis_values = np.zeros_like(output,dtype=np.uint8)
                        dis_values[np.where(output > 0.5)] = 1
                        target_sdf = data['values'][j].numpy()
                        correct = sum(dis_values == target_sdf)/len(target_sdf)
                        corrects.append(correct)
                test_total = float(test_total.cpu()/len(pbar))
                test_loss.append(test_total)
                test_loss_path = config.log_dir + "/test_loss.txt"
                with open(test_loss_path, 'a') as f:
                    f.write(str(test_total))
                    f.write("\n")
                correct_path = config.log_dir + "/correct.txt"
                with open(correct_path,'a') as f:
                    f.write(str(np.mean(corrects)))
                    f.write("\n")
                correct_all.append(np.mean(corrects))
                print(np.mean(corrects))

        if config.module == 'graph':
            with torch.no_grad():
                distances = []
                pbar = tqdm(val_loader)
                for b,data in enumerate(pbar):
                    outputs, losses = tr_agent.val_func(data)
                    for k, v in losses.items():
                        test_total += v
                    pbar.set_description("EPOCH[{}][{}]".format(e, b))
                    pbar.set_postfix(OrderedDict({k: v.item() for k, v in losses.items()}))
                    z_voxel, node_vec = outputs[0].detach().cpu().numpy(), outputs[1].detach().cpu().numpy()
                    distance = np.mean(np.sum(np.square(z_voxel - node_vec),axis=1))
                    distances.append(distance)
                test_total = float(test_total.cpu()/len(pbar))
                test_loss.append(test_total)
                test_loss_path = config.log_dir + "/test_loss.txt"
                with open(test_loss_path, 'a') as f:
                    f.write(str(test_total))
                    f.write("\n")
                distance_path = config.log_dir + "/distance.txt"
                with open(distance_path, 'a') as f:
                    f.write(str(np.mean(distances)))
                    f.write("\n")
                correct_all.append(np.mean(distance))
                print("distance: {}".format(np.mean(distances)))

        clock.tock()
        if clock.epoch % config.save_frequency == 0:
            tr_agent.save_ckpt()
            # save fig
            save_loss(train_loss, test_loss, config.log_dir, "loss")
            if config.module == 'part_ae':
                save_fig(correct_all, config.log_dir, "correct")
            if config.module == 'graph':
                save_fig(correct_all, config.log_dir, "distance")
        tr_agent.save_ckpt('latest')

        # if e < 10:
        #     train_loss = []
        #     test_loss = []

def save_fig(data, save_path, name):
    x = [i for i in range(len(data))]
    y = [data[i] for i in range(len(data))]
    plt.figure()
    plt.plot(x, y)
    plt.xlabel('epoch')
    plt.ylabel(name)
    save_path = save_path + "/{}.png".format(name)
    plt.savefig(save_path)

def save_loss(train_loss, test_loss, save_path, name):
    plt.figure()
    x = range(len(train_loss))
    plt.plot(x, train_loss, marker='.', ms=1, label='train_loss')
    plt.plot(x, test_loss, marker='.', ms=1, label='test_loss')
    plt.xlabel('epoch')  # X轴标签
    plt.ylabel("loss")  # Y轴标签
    save_path = save_path + "/{}.png".format(name)
    plt.savefig(save_path)

if __name__ == '__main__':
    main()
