import os
import esm
import torch
import faiss
import random
import argparse
import numpy as np
from tqdm import tqdm, trange
from Src.Diffusion import UNetModel, Diffusion



def get_feature(seq):
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval() 
    data = [
        ("protein1", seq),
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
    c = sequence_representations[0]
    return c

def get_smi(all_samples,topk = 1):
    xq = all_samples
    num = xq.shape[0]
    xq = xq.reshape([num,768]).numpy().astype('float32')
    k = topk
    kkk = k
    file_list = os.listdir(f"Data/vs_data/split")
    result = {}
    for file in tqdm(file_list):
        xb = torch.load(f"Data/vs_data/vsembed/{file}.pth").numpy().astype('float32')
        index = faiss.IndexFlatL2(768)             
        index.add(xb)
        D, I = index.search(xq, k)
        result[file] = {
            "D":D,
            "I":I
        }
    all_result = []
    for k in tqdm(range(1,13),position=0,leave=True):
        k_res = []
        with open(f"Data/vs_data/split/{k}.txt") as rf:
            lines = rf.readlines()
        for i in tqdm(range(num),position=1,leave=False):
            tem = []
            tem_D = result[f"{k}.txt"]["D"][i]
            tem_I = result[f"{k}.txt"]["I"][i]
            for j in range(kkk):
                tem.append(
                {
                    "smi":lines[tem_I[j]],
                    "dist":tem_D[j],
                    "id":tem_I[j],
                    "source":k
                })
            k_res.append(tem)
        all_result.append(k_res)
    output = []
    for i in tqdm(range(num)):
        tem = []
        for k in range(12):
            tem.append(all_result[k][i][0])
        tem_tem = []
        for l in tem:
            tem_tem.append(l["dist"])
        sort_index = np.argsort(tem_tem)
        out = []
        for m in range(kkk):
            out.append(tem[sort_index[m]])
        output.extend(out)
    return output





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--proseq", 
        default="",  
        type=str,
        required=True
    )
    parser.add_argument(
        "--savepath",    
        default="./generate/test",  
        type=str
    )
    parser.add_argument(
        "--topk",    
        default=1,  
        type=int
    )
    parser.add_argument(
        "--n_samples",    
        default=128,  
        type=int
    )
    parser.add_argument(
        "--n_iter",    
        default=1,  
        type=int
    )
    args = parser.parse_args()

    protein_seq = args.proseq
    save_path = args.savepath
    n_samples = args.n_samples
    n_iter = args.n_iter
    os.makedirs(save_path, exist_ok=True)
    model = UNetModel().cuda()
    model.load_state_dict(torch.load(f"Data/diffusion.pth"))
    diffusion = Diffusion(device_id=0)
    diffusion.model = model

    c = get_feature(protein_seq)

    all_samples=[]
    with torch.no_grad():
        c = c.unsqueeze(0).unsqueeze(0).repeat([n_samples,1,1]).cuda()
        for n in trange(n_iter, desc="Sampling",position=0):
            samples_ddpm = diffusion.sample(cond=c,batch_size=n_samples) / 1.7218
            all_samples.append(samples_ddpm.detach().cpu())
    all_samples = torch.cat(all_samples,dim=0)

    output = get_smi(all_samples,topk=args.topk)

    with open(f"{save_path}/out_smi_{args.topk}","w") as wf:
        all_smi = []
        for line in output:
            all_smi.append(line['smi'])
        for line in list(set(all_smi)):
            wf.write(line)