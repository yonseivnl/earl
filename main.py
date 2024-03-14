import os
import argparse
import copy
import torch
import torch.nn
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from utils import plot_tsne, get_distance_matrix
from ignite.utils import setup_logger, convert_tensor
import datasets
from collections import defaultdict
import models
import warnings
import random
import pickle5 as pickle
from utils import HardTransform
import torchvision.transforms as T


from datasets import FeatureMemory
import time

warnings.filterwarnings(action='ignore')

def parse_args():
    parser = argparse.ArgumentParser(description="Online Continual Learning")

    ### Project Configuration Arguments
    parser.add_argument("--logdir", type=str, required=True)

    ### Dataset Arguments
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--datadir", type=str, default="dataset")
    parser.add_argument("--stream", type=str, default="disjoint", choices=["disjoint", "continuous", "blurry"])
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--samples_per_task", type=int, default=10000, help="explicit task boundary for twf")

    ### Architecture Arguments
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--arch", type=str, default="resnet18")
    parser.add_argument("--neck", type=str, default="default", choices=["default", "larger"])

    ### Online Continual Learning Arguments
    parser.add_argument("--model", type=str, default="simple")
    parser.add_argument("--method", type=str, default="memory_only", choices=["er", "der", "ocs", "scr", "oddl", "remind", "mir", "memory_only", "etf", "ewc", "memo", "ncfscil", "bic", "xder", "er_ace", "cls_er", "cloc"])
    parser.add_argument("--memory_size", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_iters", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--test_batch_size", type=int, default=500)
    parser.add_argument("--test_freq", type=int, default=100)

    # Residual Memorization
    parser.add_argument("--residual_addition", action="store_true")
    parser.add_argument("--residual_num", type=int, default=50)
    parser.add_argument("--knn_top_k", type=int, default=25)

    # EMA Teacher Distillation
    parser.add_argument("--ema_coeff", type=float, default=0.999)
    parser.add_argument("--distill_coeff", type=float, default=0.1)
    parser.add_argument("--use_adaptive", action="store_true")

    # Hard Augmentation
    parser.add_argument("--pseudo_batch_size", type=int, default=1)
    parser.add_argument("--use_pseudo_stream", action="store_true")
    parser.add_argument("--aug_ver", type=str, default="ver3")
    parser.add_argument("--pickle_store", action="store_true")
    parser.add_argument("--pseudo_cls_interval", type=int, default=3)
    
    # Base Initialization
    parser.add_argument("--after_baseinit", action="store_true")
    parser.add_argument("--return_idx", action="store_true")
    parser.add_argument("--baseinit_samples", type=int, default=30000, help="# samples for base initialization")
    parser.add_argument("--feat_mem_size", type=int, default=3000, help="memory size for features")
    parser.add_argument("--spatial_feat_dim", type=int, default=8, help="size of features")
    parser.add_argument("--num_codebooks", type=int, default=32, help="# of codebooks")
    parser.add_argument("--codebook_size", type=int, default=256, help="codebook size")
    
    parser.add_argument("--domain_incre_eval", action="store_true")
    
    parser.add_argument("--strong_hard_augmentation", action="store_true")
    args, remaining_args = parser.parse_known_args()

    return args


def main(args):
    device = torch.device("cuda:0")
    pickle_prefix = args.logdir.split('/')[1]


    val_loader = None
    # Setup
    args.logdir = os.path.join(args.logdir, f"seed_{args.random_seed}")
    os.makedirs(args.logdir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=args.logdir)
    logger = setup_logger(name="main", filepath=os.path.join(args.logdir, "log.txt"))
    logger.info(" ".join(os.sys.argv))
    
    baseinit_based_method = ["ncfscil", "remind"]
    featuremem_based_method = ["ncfscil", "remind"]
    if args.method in featuremem_based_method:
        args.return_idx = True
        feature_memory = FeatureMemory(args.feat_mem_size, args.num_classes, device)
    
    dataset = datasets.get_dataset(name=args.dataset, root=args.datadir, split="train", stream=args.stream, method=args.method, random_seed=args.random_seed, num_classes=args.num_classes, return_idx=args.return_idx, domain_incre=args.domain_incre_eval)
    stream = datasets.construct_stream(dataset=dataset)
    
    reservoir_based_method = ["er", "ewc", "scr", "der", "mir", "xder", "er_ace", "cls_er", "oddl", "cloc"]
    er_based_method = ["ocs", "memo", "remind", "ncfscil"] # class balanced
    memory_only_based_method = ["memory_only", "etf"]

    if args.dataset == "cifar10":
        args.test_batch_size = 500

    elif args.dataset == "cifar100":
        args.test_batch_size = 100
    elif args.dataset == "clear10":
        args.test_batch_size = 50
    elif args.dataset == "clear100":
        args.test_batch_size = 50

    if args.num_iters >= 1:
        print("LARGER THAN 1", end=" ")
        # if integer
        if args.num_iters % 1 != 0:
            print("AND FLOAT")
            float_num_iters = args.num_iters / (int(args.num_iters) + 1) # 2.25 -> 0.75
            args.num_iters = int(args.num_iters) + 1 # 2.25 -> 3
            # 3 * 0.75 = 2.25
        else:
            print("AND INTEGER")
            float_num_iters = 1
            args.num_iters = int(args.num_iters)
    else:
        float_num_iters = args.num_iters
        args.num_iters = 1
        
    if args.dataset == "clear10":
        task_boundaries = [300, 900, 1800, 3000, 4500, 6400, 8400, 10800, 13500]
    elif args.dataset == "clear100":
        # task_boundaries = [1000, 3000 ,6000,10000,15000,21000,27998,35998,44998,54989]
        task_boundaries = [900,2700,5400,9000,13500,18900,25198,32398,40498,49489]
    task_num = 0
    # if args.dataset in ["clear10", "clear100"] and args.method=="etf":
    #     batch_sampler = datasets.ClearBatchSampler(
    #         stream=stream,
    #         memory_size=args.memory_size,
    #         batch_size=args.batch_size,
    #         num_iterations=args.num_iters,
    #     )
    #     waiting_size = 1
        
    if args.method in er_based_method:
        batch_sampler = datasets.ERBatchSampler(
            stream=stream,
            memory_size=args.memory_size,
            batch_size=args.batch_size,
            num_iterations=args.num_iters
        )
        waiting_size = args.batch_size // 2

    elif args.method in memory_only_based_method:
        batch_sampler = datasets.MemoryBatchSampler(
            stream=stream,
            memory_size=args.memory_size,
            batch_size=args.batch_size,
            num_iterations=args.num_iters,
        )
        waiting_size = 1
        
    elif args.method in reservoir_based_method:
        batch_sampler = datasets.ReservoirBatchSampler(
            stream=stream,
            memory_size=args.memory_size,
            batch_size=args.batch_size,
            num_iterations=args.num_iters,
            method=args.method
        )
        if args.method in ["der", "xder"]:
            waiting_size = args.batch_size // 3
        else:
            waiting_size = args.batch_size // 2

    elif args.method == "bic":
        batch_sampler = datasets.ValidationBatchSampler(
            stream=stream,
            memory_size=args.memory_size,
            batch_size=args.batch_size,
            num_iterations=args.num_iters,
            num_tasks = 5,
        )

        waiting_size = args.batch_size // 2

    def collate_function(batches):
        if batches == []:
            return torch.Tensor([0]), torch.Tensor([0])
        
        images = torch.stack([batch[0] for batch in batches], dim=0)
        labels = torch.tensor([batch[1] for batch in batches], dtype=torch.long)
        return images, labels
    
    if args.method == "bic":
        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            collate_fn=collate_function
        )
    else:
        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
        )
        
    hard_augmentation = HardTransform(dataset.transform.base_transform)

    # Test Dataset
    domain_incre = args.domain_incre_eval
    if domain_incre:
        test_dataset = datasets.DomainIncrementalDataset(datasets.get_dataset(name=args.dataset, root=args.datadir, split="test", class_to_label=dataset.kls_to_label, domain_incre=domain_incre, domain_to_cls=dataset.domain_to_cls))
    else:
        test_dataset = datasets.IncrementalDataset(datasets.get_dataset(name=args.dataset, root=args.datadir, split="test", class_to_label=dataset.kls_to_label, domain_incre=domain_incre))
    

    # Model
    model = models.get_model(args.method, args.arch, args.neck, image_size=dataset.image_size, num_classes=dataset.num_classes, residual_addition=args.residual_addition, residual_num=args.residual_num, dataset=dataset, num_iters=args.num_iters, lr=args.lr, knn_top_k=args.knn_top_k,
                             feat_dim=args.spatial_feat_dim, num_codebooks=args.num_codebooks, codebook_size=args.codebook_size, dataset_name=args.dataset, device=device, train_num=len(train_dataloader))
    model = model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Scaler for Mixed Precision
    scaler = torch.cuda.amp.GradScaler()
    
    rt_batch = []
    rt_images = []
    rt_labels = []
    acc_list = []
    acc_list2 = []
    train_acc_list = []
    cka_mat_dict = defaultdict(list)
    current_num_iters = 0

    for iteration, batch in enumerate(train_dataloader, start=1):

        test_batch = [batch[-1], batch[1]]
        batch = batch[:-1]

        if args.domain_incre_eval:
            if args.return_idx:
                img, label, idx, domain = batch
                batch = (img, label, idx)
                unq_domain = list(map(int, domain))
            else:
                img, label, domain = batch
                batch = (img, label)
                unq_domain = list(map(int, domain))
        if args.method in memory_only_based_method:
            batch = [batch[0][:-1], batch[1][:-1]]
        current_start_index = (iteration -1) // (args.num_iters * waiting_size)
        current_stream = stream[current_start_index * waiting_size : min((current_start_index + 1) * waiting_size, len(stream))]
        current_labels = [label for _, label in current_stream]
        new_class_observe = test_dataset.observe(list(set(current_labels)))
        if args.domain_incre_eval:
            test_dataset.observe_domain(unq_domain)

        if new_class_observe:
            if args.domain_incre_eval:
                model.observe_novel_class(len(test_dataset.observed_labels))
                optimizer = optim.Adam(model.parameters(), lr=args.lr)
            else:
                model.observe_novel_class(len(test_dataset.observed_labels))
                optimizer = optim.Adam(model.parameters(), lr=args.lr)

        current_num_iters += float_num_iters
        
        if current_num_iters >= 1:
            # Define `before_model_update`` if you want to do something before model update at each iteration
            if getattr(model, "before_model_update_", None) is not None:
                model.before_model_update_(iter_num=iteration, num_iter=args.num_iters, float_iter=float_num_iters, batch_sampler=batch_sampler)

            model.train()
            optimizer.zero_grad()
            batch = convert_tensor(batch, device=device, non_blocking=True)

            # Pseudo Stream Generation
            pseudo_loader = None
            pseudo_cls_interval = min([args.pseudo_cls_interval, args.num_classes - len(test_dataset.observed_labels)])

            if iteration != 1 and args.use_pseudo_stream: # and len(test_dataset.observed_labels) > 1:
                pseudo_classes = np.random.choice(np.arange(len(test_dataset.observed_labels)), size=min(4, len(test_dataset.observed_labels)), replace=False)
                pseudo_batch_size = args.batch_size / len(test_dataset.observed_labels)
                if pseudo_batch_size < 1:
                    do_pseudo_batch = np.random.rand(1) < pseudo_batch_size * 2
                    pseudo_indices = sum([np.random.choice(v, size=min(1, len(v)), replace=False).tolist() for cls, v in batch_sampler.memory.items() if (cls in pseudo_classes and do_pseudo_batch)], start=[])
                else:
                    pseudo_batch_size = round(args.batch_size / len(test_dataset.observed_labels))
                    pseudo_indices = sum([np.random.choice(v, size=min(pseudo_batch_size, len(v)), replace=False).tolist() for cls, v in batch_sampler.memory.items() if (cls in pseudo_classes)], start=[])
                
                if len(pseudo_indices) > 0:
                    pseudo_dataset = datasets.get_memory(dataset, pseudo_indices, hard_augmentation)
                    pseudo_loader = torch.utils.data.DataLoader(
                        pseudo_dataset,
                        batch_size=len(pseudo_indices),
                        num_workers=1,
                    )
            
            memory_dataset_train_transform = datasets.get_memory(dataset, batch_sampler.memory_buffers[0], dataset.transform.base_transform, test_dataset.dataset.transform)
            memory_loader_train_transform = torch.utils.data.DataLoader(
                memory_dataset_train_transform,
                batch_size=len(batch_sampler.memory_buffers[0]),
                num_workers=2,
            )

            with torch.cuda.amp.autocast(True):
                if args.method in featuremem_based_method:
                    if args.method in baseinit_based_method and not args.after_baseinit:
                        loss = model(batch, pseudo_loader=pseudo_loader, pseudo_cls_interval=pseudo_cls_interval, device=device, observed_classes=torch.Tensor(list(test_dataset.observed_labels)).cuda(), memory=memory_dataset_train_transform)
                    else:
                        x, y, idxs = batch
                        if len(feature_memory)>0:
                            mem_features, mem_labels = feature_memory.retrieve_feature(args.method, len(y[waiting_size:]))
                        all_features, loss = model(batch, mem_features=mem_features, mem_labels=mem_labels, observed_classes=torch.Tensor(list(test_dataset.observed_labels)).cuda())
                        feature_memory.update_feature(all_features[:waiting_size], y[:waiting_size].cpu().numpy(), idxs[:waiting_size].cpu().numpy())

                else:
                    loss = model(batch, last_iteration=len(train_dataloader)==iteration, pseudo_loader=pseudo_loader, pseudo_cls_interval=pseudo_cls_interval, device=device, optimizer=optimizer, observed_classes=torch.Tensor(list(test_dataset.observed_labels)).cuda(), memory=memory_dataset_train_transform, batch_sampler=train_dataloader.batch_sampler, iteartion=iteration)

            if (args.method in reservoir_based_method or args.method in er_based_method or (args.dataset in ["clear10", "clear100"] and args.method=="etf")) and \
                    iteration % (args.num_iters * batch_sampler.temp_batch_size) == 0:
                
                rt_batch = convert_tensor(test_batch, device=device, non_blocking=True)
                rt_batch = [rt_batch[0][:batch_sampler.temp_batch_size], rt_batch[1][:batch_sampler.temp_batch_size]]

                model.eval()
                rt_acc = model.stream_train_acc(rt_batch, observed_classes=list(test_dataset.observed_labels)) / batch_sampler.temp_batch_size
                logger.info("iteration: {}, real_time_acc: {}".format(iteration, rt_acc))
                train_acc_list.append(rt_acc)
            
            if args.method in memory_only_based_method and (not (args.dataset in ["clear10", "clear100"] and args.method=="etf")) and \
                    iteration % args.num_iters == 0:
                rt_images.append(test_batch[0][-1])
                rt_labels.append(test_batch[1][-1])

            if args.method in memory_only_based_method and (not (args.dataset in ["clear10", "clear100"] and args.method=="etf")) and \
                    iteration % (args.num_iters * batch_sampler.temp_batch_size) == 0:

                rt_batch = [torch.stack(rt_images, dim=0), torch.stack(rt_labels, dim=0)]
                rt_batch = convert_tensor(rt_batch, device=device, non_blocking=True)

                model.eval()
                
                rt_acc = model.stream_train_acc(rt_batch) / batch_sampler.temp_batch_size
                logger.info("iteration: {}, real_time_acc: {}".format(iteration, rt_acc))
                train_acc_list.append(rt_acc)

                rt_images = []
                rt_labels = []


            if loss is not None:
                scaler.scale(loss).backward()
                if getattr(model, "clip_grad_", None) is not None:
                    model.clip_grad_()
                scaler.step(optimizer)
                scaler.update()

            # Define `after_model_update function` if you want to do something after each iteration
            if getattr(model, "after_model_update_", None) is not None:
                model.after_model_update_()

            # Define `optimizer_update function` if you want to update optimizer after each iteration
            if getattr(model, "optimizer_update_", None) is not None:
                model.optimizer_update_(optimizer)

            if loss is not None:
                print(f"[{iteration:6d} / {len(train_dataloader):6d}] [bsz {batch[1].shape[0]}] [loss {loss.item():.4f}]", end="\r")
            else:
                print(f"[{iteration:6d} / {len(train_dataloader):6d}] [bsz {batch[1].shape[0]}] [loss None]", end="\r")
        
            current_num_iters -= int(current_num_iters)
        else:
            if args.method in ["der", "xder"]:
                batch_sampler.return_idx()

        if iteration % (args.num_iters * args.test_freq) == 0:
            model.eval()
            if args.method in ["etf", "scr"]:
                #memory_dataset_test_transform = datasets.get_memory(dataset, sum([v for v in batch_sampler.memory.values()], start=[]), test_dataset.dataset.transform)
                memory_dataset_test_transform = datasets.get_memory(dataset, batch_sampler.memory_buffers[0], test_dataset.dataset.transform)
                memory_loader_test_transform = torch.utils.data.DataLoader(
                    memory_dataset_test_transform,
                    batch_size=50,
                    num_workers=2,
                )
                if args.residual_addition:
                    model.update_residual(memory_loader_test_transform, device)
            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.test_batch_size,
                num_workers=args.num_workers,
            )

            if args.method == "bic":
                corrects = []
                total_num_dict = defaultdict(int)
                correct_num_dict = defaultdict(int)
                feature_vec_dict = defaultdict(list)
                for batch in test_dataloader:
                    batch = convert_tensor(batch, device=device, non_blocking=True)
                    y = batch[1][0].item()
                    val_loader = None
                    val_set = datasets.get_memory(dataset, batch_sampler.get_val(), test_dataset.dataset.transform)
                    val_loader = torch.utils.data.DataLoader(val_set, batch_size=50, num_workers=2)
                    correct= model(batch, observed_classes=list(test_dataset.observed_labels), batch_sampler=batch_sampler, val_loader=val_loader)
                    corrects.append(correct)

                acc = torch.cat(corrects, dim=0).float().mean()
                
            elif args.method == "cloc":
                acc, tmp, total_num_dict, correct_num_dict = model.lr_check(test_dataloader, iteration, args.test_freq, optimizer)
                if tmp is not None:
                    optimizer = tmp
            else:
                with torch.no_grad():
                    corrects = []
                    total_num_dict = defaultdict(int)
                    correct_num_dict = defaultdict(int)
                    feature_vec_dict = defaultdict(list)
                    for batch in test_dataloader:
                        batch = convert_tensor(batch, device=device, non_blocking=True)
                        y = batch[1][0].item()
                        correct= model(batch, observed_classes=list(test_dataset.observed_labels), batch_sampler=batch_sampler, val_loader=val_loader)
                        total_num_dict[y] += len(correct)
                        correct_num_dict[y] += torch.sum(correct).item()
                        corrects.append(correct)

                    # if args.method == "etf" and args.random_seed == 1 and args.pickle_store:
                    #     for key in list(feature_vec_dict.keys()):
                    #         cka_mat_key = model.interpret_cka(len(list(test_dataset.observed_labels)) - 1, torch.cat(feature_vec_dict[key])[:100], device)
                    #         cka_mat_dict[key].append(cka_mat_key.cpu())
                    #         cka_mat_pickle_name = pickle_prefix + "_key_" + str(key) + '.pickle'
                    #         print("cka_mat_pickle_name", cka_mat_pickle_name)
                    #         with open(cka_mat_pickle_name, 'wb') as f:
                    #             pickle.dump(cka_mat_dict[key], f, pickle.HIGHEST_PROTOCOL)

                    acc = torch.cat(corrects, dim=0).float().mean()

            stream_iteration = iteration // args.num_iters
            logger.info(f"[{stream_iteration:6d} / {len(stream):6d}] [acc {acc.item():.4f}]")
            
            if domain_incre:
                total_cls_acc = 0
                cls_num = 0
                for key in list(correct_num_dict.keys()):
                    cls_acc = correct_num_dict[key] / total_num_dict[key]
                    total_cls_acc += cls_acc
                    cls_num +=1
                    logger.info(f"[{stream_iteration:6d} / {len(stream):6d}] [cls{key} {cls_acc:.4f}]")
                cls_avg_acc = total_cls_acc/cls_num
                logger.info(f"[{stream_iteration:6d} / {len(stream):6d}] [cls avg acc {cls_avg_acc:.4f}]")
            elif args.method=="etf" and args.random_seed==1:
                for key in list(correct_num_dict.keys()):
                    cls_acc = correct_num_dict[key] / total_num_dict[key]
                    logger.info(f"[{stream_iteration:6d} / {len(stream):6d}] [cls{key} {cls_acc:.4f}]")
            tb_writer.add_scalar("acc", acc.item(), stream_iteration)
            acc_list.append(acc)

            '''
            ### check distance matrix ###
            if args.method == "etf":
                feature_list = torch.cat(model.cls_features)
                label_list = torch.cat(model.cls_feature_labels)
                get_distance_matrix(feature_list, label_list)
                model.reset_buffer()
            '''
      
         # Finalize Base initialization
        if args.method in baseinit_based_method and iteration == round(args.num_iters*args.baseinit_samples):
            baseinit_dataset = datasets.get_memory(dataset, batch_sampler.memory_buffers[0], transform=test_dataset.dataset.transform, return_idx=args.return_idx)
            baseinit_dataloader = torch.utils.data.DataLoader(baseinit_dataset,batch_size=args.batch_size, shuffle=False)
            args.after_baseinit = True
            model.finalize_baseinit(len(batch_sampler.memory_buffers[0]))
            model.eval()
            with torch.no_grad():
                for baseinit_batch in baseinit_dataloader:
                    baseinit_batch = convert_tensor(baseinit_batch, device=device, non_blocking=True)  
                    baseinit_features, labels_data, item_ixs_data = model.train_pq(baseinit_batch)
                feature_memory.update_feature(baseinit_features.cpu(), labels_data, item_ixs_data)
            print("Finish Baseinitialization")
            model.after_baseinit = True
            optimizer = optim.Adam(model.parameters(), lr=args.lr)    
            
        try:
            batch_sampler.memory_buffers.pop(0)
        except:
            print("memory_buffers is empty")
            
        if ("clear" in args.dataset):
            if getattr(model, "after_task_", None) is not None and iteration == round(args.num_iters * task_boundaries[task_num]):
                model.after_task_(memory=memory_loader_train_transform)
                if task_num != len(task_boundaries)-1:
                    task_num+=1
        else:
            if (getattr(model, "after_task_", None) is not None) and (iteration % round(args.num_iters * args.samples_per_task) == 0):
                if args.method == "xder":
                    model.eval()
                model.after_task_(memory=memory_loader_train_transform)
                if args.method == "memo":
                    optimizer = optim.Adam(model.parameters(), lr=args.lr)



    logger.info(f"[auc acc {sum(acc_list)/len(acc_list):.4f}]")
    logger.info(f"[train auc acc {sum(train_acc_list)/len(train_acc_list):.4f}]")

    # DONE
    tb_writer.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
