import os
import time
import torch
import argparse
import numpy as np

from torch import optim
import torch.distributed as dist
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader

from Src.Diffusion import UNetModel, MyDataset, collate_fn
from Src.Diffusion import Diffusion











if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",             default=0.000001,               type=float  )
    parser.add_argument("--save_path",      default="./model",              type=str    )
    parser.add_argument("--epoch",          default=50,                    type=int    )
    parser.add_argument("--batchsize",      default=768,                    type=int    )
    args = parser.parse_args()
    init_process_group(backend='nccl')
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()
    epoch = args.epoch
    batchsize = args.batchsize
    num_steps = 1000

    if rank == 0:
        print(f"Build Model.")
    model = UNetModel().to(device_id)
    model = DDP(model,device_ids=[device_id],find_unused_parameters=True)
    
    model.train()
    if rank == 0:
        print(f"Load Data.")
    data = MyDataset()
    all_datanum = len(data)
    train_sample = DistributedSampler(data)
    train_dataloader = DataLoader(
        data,
        batch_size=batchsize,
        sampler = train_sample,
        shuffle=False, 
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    if rank == 0:
        os.makedirs(f"{args.save_path}", exist_ok=True)
        os.makedirs(f"{args.save_path}/model", exist_ok=True)
        with open(f"{args.save_path}/train_step.log","w") as wf:
            wf.write(f"epoch,batch,datatype,tot_loss\n")
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
    )
    if rank == 0:
        print("Run train.")
    diffusion = Diffusion(device_id=device_id)
    for i in range(epoch):
        train_sample.set_epoch(i)
        aloss = []
        starttime = time.time()
        model.train()
        if rank ==0:
            from tqdm import tqdm
            train_dataloader = tqdm(train_dataloader)
        for j,(smi,cond) in enumerate(train_dataloader):
            smi = smi.reshape(batchsize,3,16,16) * 1.7218
            t = torch.randint(0, num_steps, (smi.shape[0],), device=device_id).long()
            loss = diffusion.p_losses(model=model,x_start=smi.to(device_id),cond=cond.to(device_id),label=None,t=t,device_id=device_id)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            aloss.append(loss.item())
            if rank ==0:
                log_file = open(f"{args.save_path}/train_step.log", 'a')
                log_file.write('{},{},{},{}\n'.format( i, j, 'train', loss.item()))
                log_file.close()
            # break
        trainloss = np.array(aloss).mean()
        traintime = time.time() - starttime
        if rank ==0:
            print(f"[{i}/{epoch}]\tTrain Loss : {trainloss}\tTrain time : {traintime}")
            epoch_str = str(i)
            while len(epoch_str) < 3:
                epoch_str = '0' + epoch_str
            save_path = f"{args.save_path}/model/model_{epoch_str}_{trainloss}_.pth"
            torch.save(model.module.state_dict(),save_path)

            