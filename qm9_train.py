from config import cfg
import torch
from tqdm import tqdm
#from main import run
from layers.SignNet_layers import SignNetGNN, GNN
from data.ZINC import ZINC_loader
from layers.SAN import SAN
from layers.RWPE_layers import RWPE_SAN
from models.GT_model import GT_model
from layers.GRIT_layers import GRIT
def create_SignNet(cfg):
    if cfg.model_SignNet.net_type == "SignNet":
        model = SignNetGNN(nfeat_node=None, nfeat_edge=None, n_in=40, n_hid=cfg.model_SignNet.hidden_size, n_out=cfg.model_SignNet.n_out_phi, n_out_gnn=cfg.model_SignNet.n_out_GNN, nl_signnet=cfg.model_SignNet.num_layers_sign,nl_rho=cfg.model_SignNet.num_layers_rho,nl_gnn=cfg.model_SignNet.num_layer_GNN,dropout=cfg.train_SignNet.dropout,pool=cfg.model_SignNet.pool,bn=cfg.model_SignNet.bn)
    else:
        model = GNN(None, None, n_hid=cfg.model_SignNet.n_hid_GNN,n_out=cfg.model_SignNet.n_out_GNN,n_layer=cfg.model_SignNet.n_layer, gnn_type=cfg.model_SignNet.gnn_type, dropout=cfg.train_SignNet.dropout, pool=cfg.model_SignNet.pool, res=True, bn=cfg.model_SignNet.bn)
    return model

def create_SAN(cfg, edge=False):

    model = SAN(cfg, edge_lpe=edge)
    return model

def create_RWPE(cfg):
    model = RWPE_SAN(cfg)
    return model

def create_GT(cfg):
    model = GT_model(cfg)
    return model

def create_GRIT(cfg):
    if cfg.dataset == "QM9":
        model = GRIT(dim_in=1,dim_out=19,cfg=cfg)
    else:
        model = GRIT(dim_in=1,dim_out=1,cfg=cfg)
    return model
def train(train_loader, model, optimizer,loss, device):
    target = 0
    scaler = torch.cuda.amp.GradScaler()
    with tqdm(train_loader, unit="batch") as tepoch:
        total_loss = 0
        N = 0
        for data in tepoch:
            data = data.to(device)
            batch_y = data.y.to(device)
            num_graphs = data.num_graphs

            #print(data.y)
            loss_calc = (model(data).squeeze() - batch_y).abs().mean()
            scaler.scale(loss_calc).backward()
            #print(loss_calc)
            #loss_calc = loss(model(data).squeeze(), batch_y)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            #loss_calc.backward()
            total_loss += loss_calc.item() * num_graphs
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            N += num_graphs
        return total_loss/N

@torch.no_grad()
def test(loader, model, device, std=0):
    #error = 0
    error = torch.zeros([1,19]).to(device)
    total_error = torch.zeros([1,19]).to(device)
    N = 0
    target = 0
    for data in loader:
        if isinstance(data, list):
            data, y, num_graphs = [d.to(device) for d in data], data[0].y[:,target].to(device), data[0].num_graphs
        else:
            data, y, num_graphs = data.to(device), data.y.to(device), data.num_graphs
        #total_error += (model(data ).squeeze() - y).abs().sum(dim=0)
        #N += num_graphs
        #print(data.y)
        #x = model(data)
        #print(y * std)
        #print(x*std)
        error += ((y * std - model(data) * std).abs() / std).sum(dim=0)
        #print(error)
    #test_perf = total_error / len(loader.dataset)
    test_perf = error / len(loader.dataset)
    return test_perf.mean().item()
    #return test_perf

#if __name__ == "__main__":
#    cfg.merge_from_file("modules/config/zinc.yaml")
#    configuration = cfg.update_config(cfg)
#    run(config=configuration, create_dataset=ZINC_loader, create_model=create_SignNet, train=train, test=test)
