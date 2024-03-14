import os
import numpy as np
import warnings

warnings.filterwarnings(action='ignore')
dir = './'

def print_from_log(exp_name, seeds=(1, 3)):
    A_auc = []
    A_last = []
    A_online = []
    F_last = []
    IF_avg = []
    KG_avg = []
    FLOPS = []
    for i in seeds:
        f = open(f'{exp_name}/seed_{i}/log.txt', 'r')
        lines = f.readlines()
        for line in lines:
            if 'auc' in line:
                list = line.split(' ')
                A_auc.append(float(list[-1][:-2])*100)
            if '[ 50000 /  50000]' in line:
                list = line.split(' ')
                A_last.append(float(list[-1][:-2])*100)
    if np.isnan(np.mean(A_auc)):
        pass
    else:
        print(f'Exp:{exp_name} \t\t\t {np.mean(A_auc):.2f}/{np.std(A_auc):.2f} \t {np.mean(A_last):.2f}/{np.std(A_last):.2f} \t  {np.mean(IF_avg):.2f}/{np.std(IF_avg):.2f}  \t  {np.mean(KG_avg):.2f}/{np.std(KG_avg):.2f}  \t  {np.mean(FLOPS):.2f}/{np.std(FLOPS):.2f}|')

print("A_auc, A_last, IF_avg, KG_avg FLOPS")

exp_list = sorted([exp for exp in os.listdir(dir)])

for exp in exp_list:
    try:
        print_from_log(exp)
    except:
        pass




