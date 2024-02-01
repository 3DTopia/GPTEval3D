import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import json

C = 400


def _construct_A_(methods, comparisons, results=None, 
                  win_dilation=True, qid_one_idx=True):
    """Construct the A matrix in Section A.1 of GLUID
       https://arxiv.org/abs/2112.10741

    Args:
        methods (List[str]): list of strings
        comparisons (dict): _description_
        results (np.array): (N runs, #comparisons, 1?)
        win_dilation (bool): whether use draws to dilate the winning count. 
                             (Default) True according to GLIDE
    """
    n = len(methods)
    A = np.zeros((n, n)) # A[i, j] = #times i methods wins over j method
    if isinstance(comparisons, list):
        comparisons = {i: c for i, c in enumerate(comparisons)}
    for qid, row in comparisons.items():
        m1 = row["m1"]
        m1_idx = methods.index(m1)
        m2 = row["m2"]
        m2_idx = methods.index(m2)
        if results is not None:
            if qid_one_idx:
                qid = int(qid) - 1
            else:
                qid = int(qid)
            res = results[:, qid].astype(np.int32)
            draws = (res == 0).astype(np.int32).sum()
            m1_wins = (res == -1).astype(np.int32).sum()
            m2_wins = (res == 1).astype(np.int32).sum()
            if win_dilation:
                m1_wins = m1_wins + draws
                m2_wins = m2_wins + draws
        else:
            res = int(row["result"])
            if res == -1:
                m1_wins = 1
                m2_wins = 0
            elif res == 1:
                m1_wins = 0
                m2_wins = 1
            elif res == 0:
                m1_wins = 1
                m2_wins = 1
            else:
                raise ValueError
                
        A[m1_idx, m2_idx] += m1_wins
        A[m2_idx, m1_idx] += m2_wins
    return A


def compute_loss(A, elos):
    elos_diff = elos.unsqueeze(0) - elos.unsqueeze(1)
    loss_matrix = A * torch.log(1 + 10 ** (elos_diff / C))
    loss = loss_matrix.sum() - torch.diagonal(loss_matrix).sum()
    return loss


def compute_glide_elo(methods, comparisons, 
                      results=None, 
                      freeze=None, init_elo=None,
                      niters=10_000, return_aux=False):
    methods = list(methods)
    A = torch.from_numpy(_construct_A_(methods, comparisons, results))
    if init_elo is not None:
        elos = nn.parameter.Parameter(
            data=torch.from_numpy(np.array(init_elo)),
            requires_grad=True
        )
    else:
        elos = nn.parameter.Parameter(
            data=torch.zeros((len(methods))),
            requires_grad=True
        )
    opt = optim.Adam([elos], lr=1e-1)
    if freeze is not None:
        freeze = 1 - torch.from_numpy(
            np.array(freeze).astype(np.float32)).float()
    else:
        freeze = torch.ones_like(elos)

    pbar = tqdm.tqdm(range(niters))
    for i in pbar:
        # (n, n), elos_diff[i, j]
        opt.zero_grad()
        loss = compute_loss(A, elos)
        loss.backward()
        elos.grad *= freeze 
        opt.step()
        
        pbar.set_description("loss:%s" % loss.item()) 
    elos_out = {m:float(elos[i]) for i, m in enumerate(methods)}
    elos_ranking = sorted(
        [(m, float(elos[i])) for i, m in enumerate(methods)],
        key=lambda x: -x[1]
    )
    if return_aux:
        return elos_out, {
            "A": A, 
            "methods": methods, 
            "ranking": elos_ranking
        }
    else:
        return elos_out