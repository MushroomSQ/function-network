import argparse
import logging
import os
import datetime
from pathlib import Path
from graph.pointnet2.ModelNetDataLoader import ModelNetDataLoader
import torch
import importlib
from graph.graph_extraction import Center_Node
from tqdm import tqdm
import graph.provider as provider
import numpy as np
import torch.nn.functional as F
import json
import shutil


torch.backends.cudnn.enabled=False

def parse_args():
    parser = argparse.ArgumentParser('Models')
    parser.add_argument('--model', type=str, default='graph.model_util', help='model name [default: pointnet2 + graph model]')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 5]')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training [default: 200]')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='learning rate in training [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='1', help='specify gpu device [default: 0]')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--category', type=str, default='easyscene', help='category to train')
    return parser.parse_args()

def test(model, loader,epoch,experiment_dir):
    list_pre = []
    list_tru = []
    for k, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        points = points.transpose(2, 1)
        points = points.cuda()
        target = target.cuda()
        pred, _ = model(points)
        pred_choice = pred.data.max(1)[1]
        list_pre.append(pred_choice.int())
        list_tru.append(target[:, 0, 0].int())
    account = 0
    for i in range(len(list_tru)):
        if list_pre[i] == list_tru[i]:
            account += 1
    with open(f"{experiment_dir}/predict.txt", 'a') as f:
        f.write("epoch:{}".format(epoch))
        f.write('\n')
        f.write(str(list_pre) + '\n')
        f.write('\n')
        f.write('-------------------------------------------------------------------------')
        f.write('\n')
        f.write('\n')
    acc = account/len(list_tru)
    return acc

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./proj_log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath(args.category)
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('pretrain_pointnet')
    experiment_dir.mkdir(exist_ok=True)

    if args.log_dir is not None:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
        experiment_dir.mkdir(exist_ok=True)

    log_dir = experiment_dir.joinpath('logs/')
    model_dir = experiment_dir.joinpath('model/')
    if os.path.exists(log_dir):
        response = input('Experiment log/model already exists, overwrite to retrain? (y/n) ')
        if response != 'y':
            exit()
        shutil.rmtree(log_dir)
        shutil.rmtree(model_dir)
    log_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    configs = json.load(open("./config.json", "r"))
    DATA_PATH = configs['dataset']

    TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, category=args.category, npoint=args.num_point, split='train',
                                       normal_channel=args.normal)
    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, category=args.category, npoint=args.num_point, split='test',
                                      normal_channel=args.normal)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=4)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=False, num_workers=4)


    '''MODEL LOADING'''

    MODEL = importlib.import_module(args.model)

    cuda = True
    if cuda:
        classifier = MODEL.Pointnet2(Center_Node.number,args.normal).cuda()
    else:
        classifier = MODEL.Pointnet2(Center_Node.number,args.normal)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checskpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    global_epoch = 0
    global_step = 0
    accuracy = 0

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        total_loss = torch.zeros(1)
        if cuda:
            total_loss = total_loss.cuda()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            # print("--------------------------")

            points,graph = data

            # points = points.data.numpy()
            # points = provider.random_point_dropout(points)
            # points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            # points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            # points = torch.Tensor(points)

            points = points.transpose(2, 1)
            if cuda:
                points = points.cuda()
                graph = graph.cuda()
            optimizer.zero_grad()
    #
            classifier = classifier.train()
            logits_center_node,trans_feat = classifier(points)
            target = graph[:, 0, 0].int()
            loss = F.cross_entropy(logits_center_node, target.long())

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            global_step += 1

        log_string("Losses:{}".format(total_loss))
        scheduler.step()
        torch.save(classifier.state_dict(), f"{model_dir}/latest.pth")

        with torch.no_grad():
            acc = test(classifier.eval(), testDataLoader,epoch+1,experiment_dir)
            log_string(f"acc:{acc}")
            if acc > accuracy:
                accuracy = acc
                torch.save(classifier.state_dict(), f"{model_dir}/best.pth")
                log_string(f"save:{epoch+1}")
            if (epoch+1) % 30 == 0:
                torch.save(classifier.state_dict(), f"{model_dir}/{epoch+1}.pth")

        global_epoch += 1

    logger.info('End of training...')

if __name__ == '__main__':
    args = parse_args()
    main(args)