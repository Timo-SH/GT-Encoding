import random
import torch
import time
import warnings
import numpy as np
from train import train, create_SAN, create_GT, test
from config import cfg, update_config
from data.QM9 import QM9_loader
import wandb
from utils import CosineWithWarmupLR
from data.Dataloader import DataLoader
import argparse
import torch_geometric

def run(config, create_dataset, create_model, train, test):
    """Function to run the pytorch geometric experiments using the dataset and model defined by the user."""
    #setting random seed
    set_random_seed(config.seed)
    #Create Dataset
    train_dataset, val_dataset, test_dataset, mean, std = create_dataset.load_data(model=config.model)


    #Create Loader
    if config.GT_dataset.node_encoder_name "BasisNet_PE" or config.GT_dataset.node_encoder_name "SAN_PE" or config.GT_dataset.node_encoder_name "RWPE_PE" or config.GT_dataset.node_encoder_name "RRWP_PE":
        train_loader = DataLoader(train_dataset, config.train_batch_size, shuffle=True, num_workers=config.num_workers, drop_last=True)
        test_loader = DataLoader(test_dataset, config.test_batch_size, shuffle=False, num_workers=config.num_workers, drop_last=True)

        val_loader = DataLoader(val_dataset, config.val_batch_size, shuffle=False, num_workers=config.num_workers, drop_last=True)
    else:
        train_loader = torch_geometric.loader.DataLoader(train_dataset, config.train_batch_size, shuffle=True, num_workers=config.num_workers, drop_last=True)
        test_loader = torch_geometric.loader.DataLoader(test_dataset, config.test_batch_size, shuffle=False, num_workers=config.num_workers, drop_last=True)

        val_loader = torch_geometric.loader.DataLoader(val_dataset, config.val_batch_size, shuffle=False, num_workers=config.num_workers, drop_last=True)
    
    test_perf = []
    val_perf = []
    #Create model
    for i in range(1, config.train_runs + 1):

        if config.model == "GT" and config.GT_dataset.node_encoder_name == "BasisNet_PE":
            model = create_model(config).to(config.device)
            # model.reset_parameters()

        else:
            model = create_model(config).to(config.device)
            model.reset_parameters()
        #optimizer and scheduler
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-3, betas=(0.9,0.95), fused=True)
        scheduler = CosineWithWarmupLR(optimizer=optimizer,warmup_iters= int(0.025 * config.train_epochs ), lr=1e-3, lr_decay_iters=config.train_epochs, min_lr=0)


        print(f"Total number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        start_outer =time.time()
        best_val_perf = test_perf = float("inf")
        wandb.watch(model, criterion=torch.nn.L1Loss(), log="all", log_freq=10)
    #Train the model
        for epoch in range(1, config.train_epochs +1):
            start = time.time()
            model.train()
            loss = torch.nn.L1Loss()

            train_loss = train(train_loader, model, optimizer,loss, config.device)


            model.eval()
            val_perf = test(val_loader, model, config.device, std)

            scheduler.step()
            if val_perf < best_val_perf:
                best_val_perf = val_perf
                test_perf = test(test_loader, model, config.device, std)
            time_per_epoch = time.time() -start
            #Save pretrained model
            if cfg.pre_train and epoch==1000:
                torch.save(model.state_dict(), "saved_models/ZINC/model.pth")

    #5 wrap up
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
                  f'Val: {val_perf:.4f}, Test: {test_perf:.4f}, Seconds: {time_per_epoch:.4f}, '
                  )
            wandb.log({"epoch": epoch, "train loss": train_loss, "val perf": val_perf, "test perf": test_perf, "time per epoch": time_per_epoch})
        time_average_epoch = time.time() - start_outer
        print(
            f'Run {run}, Vali: {best_val_perf}, Test: {test_perf}, Seconds/epoch: {time_average_epoch / config.train.epochs}.')
        test_perf.append(test_perf)
        val_perf.append(best_val_perf)


def set_random_seed(seed=0, cuda_deterministic=True):
    """
    Function to set seeds for random, Pytorch, Numpy and Cuda
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        warnings.warn('You have chosen to seed training with CUDNN deterministic setting,'
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        warnings.warn('You have chosen to seed training WITHOUT CUDNN deterministic. '
                       'This is much faster but less reproducible')




if __name__ == "__main__":

    cfg.merge_from_file("models/config/qm9.yaml")
    configuration = update_config(cfg)
    wandb.init(project="SAN_ZINC")
    loader_QM9 = QM9_loader(configuration)

    if configuration.model == "GT":
        run(config=configuration, create_dataset=loader_QM9, create_model=create_GT, train=train, test=test)
    elif configuration.model == "SAN_node":
        run(config=configuration, create_dataset=loader_QM9, create_model=create_SAN, train=train, test=test)
