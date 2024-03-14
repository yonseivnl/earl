import copy
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ODDLModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_classes: int,
        device,
        lr,
        dataset
    ):
        super().__init__()
        self.resnet = backbone
        self.num_classes = num_classes
        self.device = device
        self.lr = lr
        if dataset == "cifar10":
            self.threshold = 0.45
        elif dataset == "cifar100":
            self.threshold = 3
        self.backbones = [copy.deepcopy(self.resnet).to(device), copy.deepcopy(self.resnet).to(device)]
        self.classifiers = [nn.Linear(self.resnet.num_features, 1).to(device), nn.Linear(self.resnet.num_features, 1).to(device)]
        
        self.backbone = self.backbones[1]
        self.shared_encoder = nn.Sequential(nn.Linear(3*32*32, 2000), nn.ReLU(), nn.Linear(2000, 1500), nn.ReLU(), nn.Linear(1500, 1000)).to(device)
        self.shared_decoder = nn.Sequential(nn.Linear(1000, 1500), nn.ReLU(), nn.Linear(1500, 2000), nn.ReLU(), nn.Linear(2000, 3*32*32)).to(device)   
        self.encoders = [
            nn.Sequential(nn.Linear(1000, 600), nn.ReLU(), nn.Linear(600, 300), nn.ReLU(), nn.Linear(300, 200)),
            nn.Sequential(nn.Linear(1000, 600), nn.ReLU(), nn.Linear(600, 300), nn.ReLU(), nn.Linear(300, 200))
        ]
        self.decoders = [
            nn.Sequential(nn.Linear(200, 300), nn.ReLU(), nn.Linear(300, 600), nn.ReLU(), nn.Linear(600, 1000)),
            nn.Sequential(nn.Linear(200, 300), nn.ReLU(), nn.Linear(300, 600), nn.ReLU(), nn.Linear(600, 1000))
        ]
        self.vaes = [
            nn.Sequential(self.shared_encoder, nn.ReLU(), self.encoders[0],
                          self.decoders[0], nn.ReLU(), self.shared_decoder).to(device),
            nn.Sequential(self.shared_encoder, nn.ReLU(), self.encoders[1],
                          self.decoders[1], nn.ReLU(), self.shared_decoder).to(device)
        ]
        self.optimizer = [optim.Adam(self.backbones[0].parameters(), lr=self.lr), optim.Adam(self.backbones[1].parameters(), lr=self.lr)]
        self.vae_optimizer = [optim.Adam(self.vaes[0].parameters(), lr=self.lr / 2), optim.Adam(self.vaes[1].parameters(), lr=self.lr / 2)]
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.num_learned_class = 0
        self.num_classifier = 2
        self.do_initial = True
        self.cnt = 0

    def observe_novel_class(self, num_learned_class):
        self.num_learned_class += 1
        if num_learned_class > 1:
            for i, x in enumerate(self.classifiers):
                prev_weight = copy.deepcopy(x.weight.data)
                prev_bias = copy.deepcopy(x.bias.data)
                self.classifiers[i] = nn.Linear(self.resnet.num_features, num_learned_class).cuda()
                with torch.no_grad():
                    if num_learned_class > 1:
                        self.classifiers[i].weight[:prev_weight.shape[0]] = prev_weight
                        self.classifiers[i].bias[:prev_weight.shape[0]] = prev_bias
        if self.do_initial:
            self.optimizer = [optim.Adam(self.backbones[0].parameters(), lr=self.lr), optim.Adam(self.backbones[1].parameters(), lr=self.lr)]
            self.vae_optimizer = [optim.Adam(self.vaes[0].parameters(), lr=self.lr / 2), optim.Adam(self.vaes[1].parameters(), lr=self.lr / 2)]
        else:
            self.optimizer[-1] = optim.Adam(self.backbones[-1].parameters(), lr=self.lr)
            self.vae_optimizer[-1] = optim.Adam(self.vaes[-1].parameters(), lr=self.lr / 2)
            
    def change_batch(self, batch, batch_sampler, memory):
        if self.do_initial:
            self.backbones[0].train()
            self.optimizer[0].zero_grad()
            loss = self.initial_training(batch)
            loss.backward()
            self.optimizer[0].step()
            
        if len(memory.labels) != batch_sampler.memory_size:
            return
        
        if self.do_initial:
            self.do_initial = False
            for param in self.backbones[0].parameters():
                param.requires_grad = False
            for param in self.classifiers[0].parameters():
                param.requires_grad = False
            for param in self.vaes[0][2:4].parameters():
                param.requires_grad = False
        
        self.cnt += 1
        if self.cnt == 10:
            self.cnt = 0                
            memory_loader = torch.utils.data.DataLoader(memory, batch_size=16, num_workers=4)   
            
            if self.calculate_discrepancy(memory_loader):
                self.expansion(batch_sampler)
            
        
    def after_model_update_(self, **kwargs):
        x, _ = kwargs["batch"]
        if self.do_initial:
            self.vaes[0].train()
            self.vae_optimizer[0].zero_grad()
            _, loss = self.vae_forward(self.vaes[0], x.reshape(len(x), -1), 
                                       encoder=nn.Sequential(self.shared_encoder, nn.ReLU(), self.encoders[0]))
            loss.backward()
            self.vae_optimizer[0].step()
            
            self.vaes[1].train()
            self.vae_optimizer[1].zero_grad()
            _, loss = self.vae_forward(self.vaes[1], x.reshape(len(x), -1), 
                                       encoder=nn.Sequential(self.shared_encoder, nn.ReLU(), self.encoders[1]))
            loss.backward()
            self.vae_optimizer[1].step()
        else:
            self.vaes[-1].train()
            self.vae_optimizer[-1].zero_grad()
            _, loss = self.vae_forward(self.vaes[-1], x.reshape(len(x), -1), 
                                       encoder=nn.Sequential(self.shared_encoder, nn.ReLU(), self.encoders[-1]))
            loss.backward()
            self.vae_optimizer[-1].step()
            
    
    def change_optimizer(self):
        return self.optimizer[-1]
    
    def vae_forward(self, model, x, encoder=None):
        # generate_images, loss
        if encoder is not None:
            en_predict = encoder(x)
            mu = torch.mean(en_predict)
            logvar = torch.log(torch.var(en_predict))
        predict = model(x)
        recon_loss = F.mse_loss(x, predict, reduction='mean')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss
        return predict, loss
    

    def initial_training(self, batch):
        x, y = batch
        
        features = self.backbones[0](x)
        preds = self.classifiers[0](features)
        loss = self.criterion(preds, y)
        
        loss = loss.mean()
        return loss
        
    @torch.no_grad()    
    def calculate_discrepancy(self, memory_loader):
        results = torch.Tensor()
        bs = memory_loader.batch_size
        self.backbones[-1].eval()
        for com in range(self.num_classifier - 1):
            pred1 = torch.zeros(len(memory_loader.dataset.labels), self.num_learned_class)
            pred2 = torch.zeros(len(memory_loader.dataset.labels), self.num_learned_class)
            g_pred1 = torch.zeros(len(memory_loader.dataset.labels), self.num_learned_class)
            g_pred2 = torch.zeros(len(memory_loader.dataset.labels), self.num_learned_class)
            for i, batch in enumerate(memory_loader):
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                
                g_x = self.vaes[com](x.reshape(len(x), -1)).reshape(-1, 3, 32, 32).detach()
                pred1[i*bs:(i+1)*bs] = self.classifiers[com](self.backbones[com](x)).detach()
                pred2[i*bs:(i+1)*bs] = self.classifiers[-1](self.backbones[-1](x)).detach()
                g_pred1[i*bs:(i+1)*bs] = self.classifiers[com](self.backbones[com](g_x)).detach()
                g_pred2[i*bs:(i+1)*bs] = self.classifiers[-1](self.backbones[-1](g_x)).detach()

            diff1 = F.cross_entropy(F.softmax(pred1), F.softmax(pred2))
            diff2 = F.cross_entropy(F.softmax(g_pred1), F.softmax(g_pred2))
            results = torch.cat((results, torch.abs(diff1 - diff2).detach().unsqueeze(0)), dim=0)
        
        return self.threshold < torch.min(results)

    def expansion(self, batch_sampler):
        self.num_classifier += 1
        self.backbones.append(copy.deepcopy(self.resnet).to(self.device))
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.classifiers[-1].parameters():
            param.requires_grad = False
        for param in self.vaes[-1][2:4].parameters():
            param.requires_grad = False
        self.backbone = self.backbones[-1]
        
        self.classifiers.append(nn.Linear(self.resnet.num_features, self.num_learned_class).to(self.device))
        self.encoders.append(
            nn.Sequential(nn.Linear(1000, 600), nn.ReLU(), nn.Linear(600, 300), nn.ReLU(), nn.Linear(300, 200))
        )
        self.decoders.append(
            nn.Sequential(nn.Linear(200, 300), nn.ReLU(), nn.Linear(300, 600), nn.ReLU(), nn.Linear(600, 1000))
        )
        self.vaes.append(
            nn.Sequential(self.shared_encoder, nn.ReLU(), self.encoders[-1],
                          self.decoders[-1], nn.ReLU(), self.shared_decoder).to(self.device)
        )
        self.optimizer.append(optim.Adam(self.backbones[-1].parameters(), lr=self.lr))
        self.vae_optimizer.append(optim.Adam(self.vaes[-1].parameters(), lr=self.lr / 2))
            
        for key in batch_sampler.memory:
            batch_sampler.memory[key] = []
        new_buffer = [[]]
        for i in range(len(batch_sampler.memory_buffers) - 1):
            new = list(set(batch_sampler.memory_buffers[i]) - set(batch_sampler.memory_buffers[i + 1]))[0]
            new_buffer.append(sorted(new_buffer[-1] + [new]))
            
        batch_sampler.memory_buffers = new_buffer
        batch_sampler.counter = torch.zeros(len(batch_sampler.counter)) 
        
    def forward(self, batch, return_feature=False, feature=None, distill_coeff = 0.01, **kwargs):
        if self.training:
            x, y = batch
            features = self.backbone(x)
            if return_feature:
                return features.detach()
            preds = self.classifiers[-1](features)
            loss = self.criterion(preds, y)
            loss = loss.mean()
            return loss
        else:
            x, y = batch
            preds = torch.zeros(len(x[0]), self.num_learned_class).to(kwargs["device"])
            for i in range(self.num_classifier):
                preds += self.classifiers[i](self.backbones[i](x[0])).detach()
            return preds.argmax(dim=1) == y
