import argparse
import logging
import os
import datetime
from pathlib import Path
from graph.pointnet2.ModelNetDataLoader import ModelNetDataLoader
import torch
import importlib
import shutil
from tqdm import tqdm
import graph.provider as provider
import numpy as np
from graph.geometry_helpers import Obj_Interaction,Center_Node,Surround_Node
import pickle
import json

torch.backends.cudnn.enabled=False

def parse_args():
    parser = argparse.ArgumentParser('Models')
    parser.add_argument('--model', type=str, default='graph.model', help='model name [default: pointnet2 + graph model]')
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
    parser.add_argument('--load_pretrain', action='store_true', default=False, help='pretrain pointnet2 load [default: False]')

    return parser.parse_args()

def test(model, loader,epoch,experiment_dir):
    list_pre = []
    accuracy = []
    boxes = []
    for k, data in tqdm(enumerate(loader), total=len(loader)):
        list_truth = []
        list_predict = []
        acc = 0
        points, target = data
        points = points.transpose(2, 1)
        points = points.cuda()

        interact_list,predict_edge_list,pred,box,success = model.predict(points)
        if success == 1:
            accuracy.append(0)
            continue
        boxes.append(np.array(box))

        node_num = target[0, 0, 1].int()
        start_truth = Center_Node.category[target[0,0,0].int()]
        for i in range(1,node_num):
            end_truth = Surround_Node.category[target[0,i,1].int()]
            edge_type_truth = Obj_Interaction.function[target[0,i,0].int()]
            edge_truth = "{} --- {} --- {}".format(start_truth, edge_type_truth, end_truth)
            list_truth.append(edge_truth)
        for i in range(1,node_num):
            end_truth = Surround_Node.category[target[0,i,1].int()]
            state_true = int(target[0,i,3].cpu())
            drt_true = int(target[0,i,4].cpu())
            dst_true = int(target[0,i,5].cpu())
            v_drt_true = int(target[0,i,6].cpu())
            edge_truth = "{} --- {}-{}-{}-{} --- {}".format(start_truth, state_true, drt_true, dst_true, v_drt_true, end_truth)
            list_truth.append(edge_truth)

        num_truth = len(list_truth)

        for i in range(len(interact_list)):
            start_predict, end_predict, edge_type_predict = interact_list[i].get()
            edge_predict = "{} --- {} --- {}".format(start_predict, edge_type_predict, end_predict)
            list_predict.append(edge_predict)
        for i in range(len(predict_edge_list)):
            start_predict, end_predict, state, drt, dst, v_drt = predict_edge_list[i].get()
            edge_predict = "{} --- {}-{}-{}-{} --- {}".format(start_predict, state, drt, dst, v_drt, end_predict)
            list_predict.append(edge_predict)
        list_pre.append(list_predict)
        for i in range(len(list_predict)):
            if list_predict[i] in list_truth:
                acc +=1
                list_truth.remove(list_predict[i])

        acc = acc/num_truth
        accuracy.append(acc)

    acc = np.mean(accuracy)
    with open(f"{experiment_dir}/predict.txt", 'a') as f:
        f.write("epoch:{}".format(epoch))
        f.write('\n')
        for i in range(len(list_pre)):
            f.write(str(list_pre[i]) + '\n')
        f.write('\n')
        f.write('-------------------------------------------------------------------------')
        f.write('\n')
        f.write('\n')
    boxes = np.array(boxes)
    return acc,boxes

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
    category_dir = experiment_dir
    experiment_dir = experiment_dir.joinpath('classification')
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


    '''graph config'''
    shuffle_nodes = False
    shuffle_edges = False
    cuda = True
    config = {
        "shuffle_nodes": shuffle_nodes,
        "shuffle_edges": shuffle_edges,
        "hidden_size": 384,
        "initializing_layer": "3",
        "propagation_layer": "3",
        "aggregation_layer": "3",
        "choose_node_graph_vector": True,
        "node_and_type_together": True,
        "init_with_graph_representation": True,
        "everything_together": True,
        "include_one_hot": False,
        "cuda": cuda,
        "rounds_of_propagation_dict": {"ita": 3,"edge":3},
    }
    with open(f"{experiment_dir}/graph_config.pkl", 'wb') as f:
        pickle.dump(config, f, pickle.HIGHEST_PROTOCOL)
    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    # shutil.copy('%s.py' % args.model, str(experiment_dir))
    # shutil.copy('pointnet2\\pointnet_util.py', str(experiment_dir))

    for (key, value) in config.items():
        setattr(MODEL.GraphNetConfig, key, value)
    MODEL.GraphNetConfig.compute_derived_attributes()
    print(MODEL.GraphNetConfig.__dict__)

    if cuda:
        classifier = MODEL.GraphNet(normal_channel=args.normal).cuda()
    else:
        classifier = MODEL.GraphNet(normal_channel=args.normal)

    '''load pretrain model''' 
    load_pointnet2 = args.load_pretrain

    if load_pointnet2 == True:
        start_epoch = 0
        pretrain_checkpoint = torch.load(os.path.join(category_dir, 'pretrain_pointnet', 'model', 'best.pth'))
        classifier.pointnet2.load_state_dict(pretrain_checkpoint)
        log_string('Use pretrain pointnet2 model')
    else:
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, classifier.parameters()),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, classifier.parameters()), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    global_epoch = 0
    global_step = 0
    accuracy = 0

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        total_losses = {}
        total_losses["acn"] = torch.zeros(1)  # add node
        total_losses["an"] = torch.zeros(1)  # add node
        total_losses["ita"] = torch.zeros(1)
        total_losses["edge"] = torch.zeros(1)  # choose node
        total_losses["box"] = torch.zeros(1)
        if cuda:
            for key in total_losses.keys():
                total_losses[key] = total_losses[key].cuda()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):

            points,graph = data

            ## data argumentation
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

            classifier = classifier.train()
            losses = classifier.train_step(points,graph)

            loss = sum(losses.values())

            loss.backward()

            optimizer.step()

            b = classifier.batch_size
            n = classifier.node_size
            for key in losses.keys():
                if key =="acn":
                    total_losses[key] += losses[key]
                else:
                    total_losses[key] += (losses[key] * b) / n

            global_step += 1
        log_string("Losses:")
        for (key, value) in total_losses.items():
            log_string(f"    {key}: {(value/len(trainDataLoader))[0].data.cpu().numpy()}")

        scheduler.step()

        torch.save(classifier.state_dict(), f"{model_dir}/latest.pth")

        with torch.no_grad():
            acc,box = test(classifier.eval(), testDataLoader,epoch+1,experiment_dir)
            log_string(f"acc:{acc}")

            if acc > accuracy:
                accuracy = acc
                torch.save(classifier.state_dict(), f"{model_dir}/best.pth")
                # torch.save(classifier.state_dict(), "proj_log/easyscene_graph/best.pth")
                # np.save(f"{model_dir}/box.npy",np.array(box))
                log_string(f"save:{epoch+1}")
            if (epoch+1) % 30 == 0:
                torch.save(classifier.state_dict(), f"{model_dir}/{epoch+1}.pth")

        global_epoch += 1

    logger.info('End of training...')

if __name__ == '__main__':
    args = parse_args()
    main(args)