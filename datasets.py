
from typing import List, Tuple
import json
import random
from collections import defaultdict
import os
import subprocess
import numpy as np
from glob import glob
import PIL
from PIL import Image
import copy
import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from collections import Counter
from utils import TrainTransform, TestTransform, GaussianBlur

random.seed(10)
torch.manual_seed(10)

def get_memory(dataset, indices, transform, other_transform=None, return_idx=None):
    return MemoryDataset(dataset.data, dataset.labels, indices,  transform=transform, other_transform=other_transform, return_idx=return_idx)

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def get_dataset(
    name: str = "cifar10",
    root: str = "/data",
    split: str = "train",
    stream: str = "disjoint",
    method: str = "er",
    random_seed: int = 1,
    class_to_label = None,
    domain_incre = False,
    domain_to_cls = None,
    
    num_classes: int = 10,
    return_idx: bool = False,
):
    """
    return a Dataset instance containing the following attributes:
    - num_classes (int)
    - labels (list[int])
    - image_size (tuple)
    """

    assert split in ["train", "test"]

    if name == "cifar10":
        mean = [0.4914, 0.4822, 0.4465]
        std  = [0.2023, 0.1994, 0.2010]
        image_size = 32
    
    elif name == "cifar100":    
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        image_size = 32
    
    if name == "clear10":
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        image_size = 224
    
    elif name == "clear100":    
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image_size = 224
    
    elif name == "tinyimagenet":
        mean = [0.4802, 0.4481, 0.3975]
        std = [0.2302, 0.2265, 0.2262]
        image_size = 64
    
    elif name == "imagenet" or name == "imagenet200":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image_size = 224

    transform = get_transform(split, mean, std, image_size, method)
    test_transform = get_transform("test", mean, std, image_size, method)
    d = ContinualDataset(name, root, stream, random_seed, train=(split == "train"), download=True, transform=transform, test_transform=test_transform, class_to_label=class_to_label, return_idx=return_idx, domain_to_cls=domain_to_cls, domain_incre=domain_incre)
    d.num_classes = num_classes
    d.image_size = image_size
    d.transform = transform
    return d

def get_transform(
    split: str,
    mean: list,
    std: list,
    image_size: int = 224,
    method: str = "er"
):
    test_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    
    if split == "train":
        '''
        base_transform = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandAugment(),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean, std),
        ])
        '''
        '''
        # CPU version
        base_transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.RandomCrop(image_size, padding=4),
                T.RandomHorizontalFlip(),
                T.RandAugment(),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )
        '''
        # GPU version
        base_transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.PILToTensor(),
                T.RandomCrop(image_size, padding=4),
                T.RandomHorizontalFlip(),
                T.RandAugment(),
                T.ConvertImageDtype(torch.float32),
                T.Normalize(mean, std),
            ])
        transform = TrainTransform(base_transform = base_transform, test_transform = test_transform, method = method)
    else:
        transform = TestTransform(base_transform = test_transform)
    return transform

def construct_stream(
    dataset: torch.utils.data.Dataset
):
    stream = []
    for idx, label in enumerate(dataset.labels):
        stream.append((idx, label))

    return stream

# class ClearBatchSampler(torch.utils.data.sampler.BatchSampler):
#     def __init__(
#         self,
#         stream: List[Tuple[int, int]],
#         memory_size: int = 500,
#         batch_size: int = 64,
#         num_iterations: int = 1
#     ):
#         self.stream = stream
#         self.memory_size = memory_size
#         self.batch_size = batch_size
#         self.temp_batch_size = self.batch_size // 2
#         self.num_iterations = num_iterations
#         self.num_classes = max([y for _, y in self.stream]) + 1
#         self.memory = defaultdict(list)
#         self.memory_buffers = []
        
#         self.cls_seen_count = defaultdict(int)
#         self.cls_count = defaultdict(int)
#         self.seen_classes = []

#     def __iter__(self):
#         counter = torch.zeros(self.num_classes, dtype=torch.long)
#         for idx, y in self.stream:
#             self.cls_seen_count[y] += 1
#             if y not in self.seen_classes:
#                 self.seen_classes.append(y)
#             if sum(len(v) for v in self.memory.values()) == self.memory_size:
#                 replace=True
#                 if counter[y] > self.memory_size // len(self.seen_classes):
#                     # for easy implementation, assume counter.max() > 1
#                     j = np.random.randint(0, self.cls_seen_count[y])
#                     if j > self.memory_size // len(self.seen_classes):
#                         replace = False
#                 if replace:
#                     classes = torch.where(counter == counter.max())[0].tolist()
#                     cls = y if y in classes else random.choice(classes)
#                     self.memory[cls].pop(random.randrange(len(self.memory[cls])))
#                     counter[cls] -= 1
#                     self.memory[y].append(idx)
#                     counter[y] += 1
#             else:
#                 self.memory[y].append(idx)
#                 counter[y] += 1
#             memory_data = sum([v for v in self.memory.values()], start=[])
#             for _ in range(self.num_iterations):
#                 self.memory_buffers.append(sum([v for v in self.memory.values()], start=[]))
#                 batch = random.sample(memory_data, k=min(len(memory_data), self.batch_size))
#                 #print("yield idx", idx, "len mem buffer", len(self.memory_buffers))
#                 yield batch + [idx]

#     def __len__(self):
#         return len(self.stream) * self.num_iterations

class MemoryBatchSampler(torch.utils.data.sampler.BatchSampler):
    def __init__(
        self,
        stream: List[Tuple[int, int]],
        memory_size: int = 500,
        batch_size: int = 64,
        num_iterations: int = 1
    ):
        self.stream = stream
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.temp_batch_size = self.batch_size // 2
        self.num_iterations = num_iterations
        self.num_classes = max([y for _, y in self.stream]) + 1
        self.memory = defaultdict(list)
        self.memory_buffers = []

    def __iter__(self):
        counter = torch.zeros(self.num_classes, dtype=torch.long)
        for idx, y in self.stream:
            if sum(len(v) for v in self.memory.values()) == self.memory_size:
                # for easy implementation, assume counter.max() > 1
                classes = torch.where(counter == counter.max())[0].tolist()
                cls = y if y in classes else random.choice(classes)
                self.memory[cls].pop(random.randrange(len(self.memory[cls])))
                counter[cls] -= 1

            self.memory[y].append(idx)
            counter[y] += 1
            memory_data = sum([v for v in self.memory.values()], start=[])
            for _ in range(self.num_iterations):
                self.memory_buffers.append(sum([v for v in self.memory.values()], start=[]))
                batch = random.sample(memory_data, k=min(len(memory_data), self.batch_size))
                #print("yield idx", idx, "len mem buffer", len(self.memory_buffers))
                yield batch + [idx]

    def __len__(self):
        return len(self.stream) * self.num_iterations


class ReservoirBatchSampler(torch.utils.data.sampler.BatchSampler):
    def __init__(
        self,
        stream: List[Tuple[int, int]],
        memory_size: int = 500,
        batch_size: int = 64,
        num_iterations: int = 1,
        method: str = "er"
    ):
        self.stream = stream
        self.memory_size = memory_size
        self.batch_size = batch_size
        if method in ["der", "xder"]:        
            self.temp_batch_size = self.batch_size // 3
        else:
            self.temp_batch_size = self.batch_size // 2
        self.memory_batch_size = self.batch_size - self.temp_batch_size
        self.num_iterations = num_iterations
        self.memory = list()
        self.memory_labels = list()
        self.memory_buffers = []
        self.seen = 0
        self.stream_idx = []
        self.memory_idx = []
        self.buf_idx = []
        self.str_to_mem = {}

    def __iter__(self):
        temp_batch = []
        temp_labels = []
        for idx, y in self.stream:
            self.seen += 1
            temp_batch.append(idx)
            temp_labels.append(y)
            if len(temp_batch) < self.temp_batch_size:
                continue
            # pre define memory batch before memory update
            for temp_idx, temp_label in zip(temp_batch, temp_labels):
                if len(self.memory) == self.memory_size:
                    j = np.random.randint(0, self.seen)
                    if j < self.memory_size:
                        self.str_to_mem[temp_idx] = j
                        self.memory[j] = temp_idx
                        self.memory_labels[j] = temp_label
                else:
                    self.str_to_mem[temp_idx] = len(self.memory)
                    self.memory.append(temp_idx)
                    self.memory_labels.append(temp_label)
                for _ in range(self.num_iterations):
                    self.memory_buffers.append(copy.deepcopy(self.memory))
                    memory_batch = random.sample(self.memory, k=min(len(self.memory), self.memory_batch_size))
                    self.stream_idx.append(temp_batch)
                    self.memory_idx.append(memory_batch)
                    self.buf_idx.append([self.memory.index(value) for value in memory_batch])
                    yield temp_batch + memory_batch
            temp_batch = []
            temp_labels = []
            
        if len(temp_labels)!=0:
            memory_batch_size = self.batch_size - len(temp_labels)
            for temp_idx, temp_label in zip(temp_batch, temp_labels):
                if len(self.memory) == self.memory_size:
                    j = np.random.randint(0, self.seen)
                    if j < self.memory_size:
                        self.memory[j] = temp_idx
                        self.memory_labels[j] = temp_label
                else:
                    self.memory.append(temp_idx)
                    self.memory_labels.append(temp_label)
                    
                for _ in range(self.num_iterations):
                    self.memory_buffers.append(self.memory)
                    memory_batch = random.sample(self.memory, k=min(len(self.memory), memory_batch_size))
                    self.stream_idx.append(temp_batch)
                    self.memory_idx.append(memory_batch)
                    yield temp_batch + memory_batch

    def __len__(self):
        return len(self.stream) * self.num_iterations
    
    def return_idx(self):
        out = self.stream_idx[0], self.memory_idx[0]
        self.stream_idx = self.stream_idx[1:]
        self.memory_idx = self.memory_idx[1:]
        return out   

    def return_buf(self):
        out = self.buf_idx[0]
        self.buf_idx = self.buf_idx[1:]
        return torch.tensor(out)
        


class ERBatchSampler(torch.utils.data.sampler.BatchSampler):
    def __init__(
        self,
        stream: List[Tuple[int, int]],
        memory_size: int = 500,
        batch_size: int = 64,
        num_iterations: int = 1
    ):
        self.stream = stream
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.temp_batch_size = self.batch_size // 2
        self.memory_batch_size = self.batch_size - self.temp_batch_size
        self.num_iterations = num_iterations
        self.num_classes = max([y for _, y in self.stream]) + 1
        self.memory = defaultdict(list)
        self.memory_buffers = []
        self.stream_idx = []
        self.memory_idx = []
        
    def __iter__(self):
        temp_batch = []
        temp_labels = []
        counter = torch.zeros(self.num_classes, dtype=torch.long)
        for idx, y in self.stream:
            temp_batch.append(idx)
            temp_labels.append(y)
            if len(temp_batch) < self.temp_batch_size:
                continue
            # pre define memory batch before memory update
            memory_batch = sum([v for v in self.memory.values()], start=[])
            for temp_idx, temp_label in zip(temp_batch, temp_labels):
                if sum(len(v) for v in self.memory.values()) == self.memory_size:
                    # for easy implementation, assume counter.max() > 1
                    classes = torch.where(counter == counter.max())[0].tolist()
                    cls = temp_label if temp_label in classes else random.choice(classes)
                    self.memory[cls].pop(random.randrange(len(self.memory[cls])))
                    counter[cls] -= 1

                self.memory[temp_label].append(temp_idx)
                counter[temp_label] += 1
                for _ in range(self.num_iterations):
                    self.memory_buffers.append(sum([v for v in self.memory.values()], start=[]))
                    memory_batch = random.sample(memory_batch, k=min(len(memory_batch), self.memory_batch_size))
                    self.stream_idx.append(temp_batch)
                    self.memory_idx.append(memory_batch)
                    yield temp_batch + memory_batch
            temp_batch = []
            temp_labels = []

        if len(temp_labels)!=0:
            memory_batch = sum([v for v in self.memory.values()], start=[])
            memory_batch_size = self.batch_size - len(temp_labels)
            for temp_idx, temp_label in zip(temp_batch, temp_labels):
                if sum(len(v) for v in self.memory.values()) == self.memory_size:
                    # for easy implementation, assume counter.max() > 1
                    classes = torch.where(counter == counter.max())[0].tolist()
                    cls = temp_label if temp_label in classes else random.choice(classes)
                    self.memory[cls].pop(random.randrange(len(self.memory[cls])))
                    counter[cls] -= 1

                self.memory[temp_label].append(temp_idx)
                counter[temp_label] += 1
                for _ in range(self.num_iterations):
                    self.memory_buffers.append(sum([v for v in self.memory.values()], start=[]))
                    memory_batch = random.sample(memory_batch, k=min(len(memory_batch), memory_batch_size))
                    self.stream_idx.append(temp_batch)
                    self.memory_idx.append(memory_batch)
                    yield temp_batch + memory_batch

    def __len__(self):
        return len(self.stream) * self.num_iterations
    
    def return_idx(self):
        out = self.stream_idx[0], self.memory_idx[0]
        self.stream_idx = self.stream_idx[1:]
        self.memory_idx = self.memory_idx[1:]
        return out   


class MemoryDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, indices, transform, other_transform=None, return_idx=None):

        self.transform = transform
        self.other_transform = other_transform
        self.data = np.array(data)[indices]
        self.labels = np.array(labels)[indices]
        self.indices = indices
        self.return_idx = return_idx

    def __getitem__(self, idx):
        img_path, label = self.data[idx], self.labels[idx]
        #img = PIL.Image.open(img_path)
        img = pil_loader(img_path)
        if self.other_transform is None:
            img = self.transform(img)
            if self.return_idx:
                ind = self.indices[idx]
                return img, label, ind
            return img, label
        else:
            img0 = self.transform(img)
            img1 = self.other_transform(img)
            if self.return_idx:
                ind = self.indices[idx]
                return [img0, img1], label, ind
            return [img0, img1], label

    def __len__(self):
    	return len(self.data)
 

class ContinualDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, root_dir, stream, random_seed, download, train, transform, test_transform, return_idx=False, class_to_label=None, domain_to_cls=None, domain_incre=False):
        if download and dataset not in os.listdir(root_dir):
            subprocess.run([root_dir + "./" + dataset + ".sh"], shell=True)
        self.train = train    
        self.dataset = dataset
        self.transform = transform
        self.test_transform = test_transform
        self.data = []
        self.labels = []
        self.domains = []
        if class_to_label is None:
            self.kls_to_label = defaultdict(int)
        else:
            self.kls_to_label = class_to_label

        self.observed_domain = []
        if domain_to_cls is None:
            self.domain_to_cls = defaultdict(list)
        else:
            self.domain_to_cls = domain_to_cls
        self.domain_to_ind = defaultdict(list)
        self.domain_incre = domain_incre
        
        self.observed_klass = []
        self.return_idx = return_idx

        if train:
            json_file_name = os.path.join("collections", dataset, dataset + "_" + stream + "_seed" + str(random_seed) + ".json")
        else:
            json_file_name = os.path.join("collections", dataset, dataset + "_val.json")

        with open(json_file_name) as f:
            datas = json.load(f)

        # for imagenet, there are different types of json file
        try: data_stream = datas["stream"]
        except: data_stream = datas

        for idx, data in enumerate(data_stream):
            
            if data["klass"] not in self.observed_klass and train:
                self.kls_to_label[data["klass"]] = len(self.observed_klass)
                self.observed_klass.append(data["klass"])
            
            # 수정
            if self.domain_incre:
                if data["time"] not in self.observed_domain and train:
                    self.observed_domain.append(data["time"])
                if data["klass"] not in self.domain_to_cls[data["time"]] and train:
                    self.domain_to_cls[data["time"]].append(data["klass"])
                        
                if not train:
                    if data["klass"] in self.domain_to_cls[data["time"]]:
                        self.domain_to_ind[data["time"]].append(idx)
                
            if data["klass"] in self.kls_to_label:
                self.labels.append(self.kls_to_label[data["klass"]])
            
            if data["klass"] in self.kls_to_label:
                if dataset in ["imagenet200", "imagenet"]:
                    self.data.append(os.path.join("/home/user/khs/ILSVRC/Data/CLS-LOC", data["file_name"]))
                else:
                    self.data.append(os.path.join("dataset", dataset, data["file_name"]))
        
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx], self.labels[idx]
        # if not self.train:
        #     print("eval img_path", img_path)
        #img = PIL.Image.open(img_path)
        img_base = pil_loader(img_path)
        if self.domain_incre and self.train:
            domain = img_path.split("/")[-3]
            if self.transform:
                img = self.transform(img_base)
                other_img = self.test_transform(img_base)[0]
            if self.return_idx:
                return img, label, idx, domain, other_img
            return img, label, domain, other_img
        elif self.train:
            if self.transform:
                img = self.transform(img_base)
                other_img = self.test_transform(img_base)[0]
            if self.return_idx:
                return img, label, idx, other_img
            return img, label, other_img
        else:
            if self.transform:
                img = self.transform(img_base)
            if self.return_idx:
                return img, label, idx
            return img, label

    def __len__(self):
    	return len(self.data)


class IncrementalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
    ):
        self.dataset = dataset
        self.observed_labels = set()
        self.label_to_indices = defaultdict(list)
        for idx, y in enumerate(dataset.labels):
            self.label_to_indices[y].append(idx)
        self.indices = []

    def observe(self, y_list):
        novel = False
        for y in y_list:
            if y not in self.observed_labels:
                self.observed_labels.add(y)
                self.indices += self.label_to_indices[y]
                novel = True
        return novel

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

# 수정
class DomainIncrementalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        onlydomainclass_eval=False
    ):
        self.dataset = dataset
        self.observed_labels = set()
        self.observed_domains = set()
        self.domain_to_indices = dataset.domain_to_ind
        self.label_indices = []
        self.indices = []
        self.onlydomainclass_eval = onlydomainclass_eval

    def observe(self, y_list):
        novel = False
        for y in y_list:
            if y not in self.observed_labels:
                self.observed_labels.add(y)
                novel = True
        return novel

    def observe_domain(self, domain_list):
        for domain in domain_list:
            if domain not in self.observed_domains:
                self.observed_domains.add(domain)
                if self.onlydomainclass_eval:
                    self.indices = self.domain_to_indices[domain]
                else:
                    self.indices += self.domain_to_indices[domain]
                # print("test domain_to_ind", self.domain_to_indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class FeatureMemory:
    def __init__(
        self,
        memory_size,
        num_classes,
        device
    ):
        self.memory_size = memory_size
        self.device = device
        self.cls_features = dict()
        self.labels = []
        self.sample_ids = []
        self.ids_features = dict()
        self.future_remove_ids = []
        self.cls_count = [0]*num_classes
        self.cls_idx = dict()
        

    def __len__(self):
        return len(self.sample_ids)
    
    def update_feature(self, features, labels, ids):
        for i in range(len(features)):
            if ids[i] in self.sample_ids:
                continue
            if labels[i] not in self.labels:
                self.cls_features[labels[i]] = []
                self.labels.append(labels[i])
                self.cls_idx[labels[i]] = []
            if len(self.sample_ids) >= self.memory_size:
                remove_label = self.cls_count.index(max(self.cls_count))
                j = np.random.randint(0, len(self.cls_idx[remove_label]))
                self.update_only_features(features[i], labels[i], ids[i], remove_label=remove_label, j=j)
            else:
                self.update_only_features(features[i], labels[i], ids[i])

    def update_only_features(self, feature, label, id, remove_label=None, j=None):
        if j is None:
            self.ids_features[id] = [feature, label]
            self.cls_features[label].append(feature) 
            self.cls_idx[label].append(id)
            self.sample_ids.append(id)
            self.cls_count[label] += 1
        else:
            remove_id = self.cls_idx[remove_label][j]
            remove_label = remove_label
            remove_feature = self.ids_features[remove_id][0]
            del self.ids_features[remove_id]
            self.cls_idx[remove_label].remove(remove_id)
            self.sample_ids.remove(remove_id)
            self.cls_features[remove_label] = [feat for feat in self.cls_features[remove_label] if feat is not remove_feature]
            self.cls_count[remove_label] -= 1

            self.ids_features[id] = [feature, label]
            self.cls_idx[label].append(id)
            self.cls_features[label].append(feature) 
            self.sample_ids.append(id)
            self.cls_count[label] += 1

    def retrieve_feature(self, method, mem_batch_size):
        retrieve_size = min(len(self.sample_ids), mem_batch_size)
        # if method == "ncfscil":
        #     output_features, output_labels = self.retrieve_feature_means(retrieve_size)
        # else:
        output_features, output_labels = self.retrieve_baseinit_features(retrieve_size)
        return output_features, output_labels
    
    def retrieve_baseinit_features(self, mem_batch_size):
        output_features = []
        output_labels = []
        indices = np.random.choice(range(len(self.sample_ids)), size=mem_batch_size, replace=False)
        for ind in indices:
            # output_features.append(torch.tensor(self.ids_features[self.sample_ids[ind]][0]))
            sample_id = self.sample_ids[ind]
            output_features.append(self.ids_features[sample_id][0])
            output_labels.append(torch.tensor(self.ids_features[sample_id][1]))
        return output_features, output_labels

    # def retrieve_feature_means(self, mem_batch_size):
    #     mean_features = []
    #     output_labels = []

    #     label_indices = np.random.choice(range(len(self.labels)), size=mem_batch_size, replace=True)
    #     for label in label_indices:
    #         # mean_features.append(torch.mean(torch.stack(self.cls_features[label]), axis=0))
    #         mean_features.append(np.mean(self.cls_features[label], axis=0))
    #         output_labels.append(torch.tensor(label))
    #     return mean_features, output_labels
