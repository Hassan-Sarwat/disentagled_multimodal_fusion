import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict
from pprint import pprint
import wandb
from dotenv import load_dotenv

import torch
from torch.utils.data import DataLoader, Subset, random_split

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import json
import models.baselines as baselines
from dataset import CUB, Caltech, HandWritten, PIE, Scene
from classifiers import IdentityEncoder
from analysis import evaluate_subjective_model, evaluate_subjective_model_with_shared, build_metrics_dataframe_datasets
from  models.dmvae import DMVAE
from models.evidential_probe  import EvidentialProbeModule, DisentangledEvidentialProbeModule

load_dotenv()
api_key = os.getenv('WAND_API_KEY')
wandb.login(key=api_key)


def get_normal_data(dataset_name, batch_size):
    if dataset_name == 'CUB':
        dataset = CUB()
    elif dataset_name == 'CalTech':
        dataset = Caltech()
    elif dataset_name == 'HandWritten':
        dataset = HandWritten()
    elif dataset_name == 'PIE':
        dataset = PIE()
    elif dataset_name == 'Scene':
        dataset = Scene()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    num_samples = len(dataset)
    num_classes = dataset.num_classes
    num_views = dataset.num_views
    dims = list(np.squeeze(dataset.dims))
    index = np.arange(num_samples)
    np.random.shuffle(index)
    train_index, test_index = index[:int(0.8 * num_samples)], index[int(0.8 * num_samples):]


    train_loader = DataLoader(Subset(dataset, train_index), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(Subset(dataset, test_index), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, num_classes, num_views, dims

def get_conflict_data(dataset_name, batch_size):
    if dataset_name == 'CUB':
        dataset = CUB()
    elif dataset_name == 'CalTech':
        dataset = Caltech()
    elif dataset_name == 'HandWritten':
        dataset = HandWritten()
    elif dataset_name == 'PIE':
        dataset = PIE()
    elif dataset_name == 'Scene':
        dataset = Scene()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    num_samples = len(dataset)
    num_classes = dataset.num_classes
    num_views = dataset.num_views
    dims = list(np.squeeze(dataset.dims))
    index = np.arange(num_samples)
    np.random.shuffle(index)
    train_index, test_index = index[:int(0.8 * num_samples)], index[int(0.8 * num_samples):]

    # create a test set with conflict instances
    dataset.postprocessing(test_index, addNoise=False, sigma=0.5, ratio_noise=0.0, addConflict=True, ratio_conflict=1.0)

    train_loader = DataLoader(Subset(dataset, train_index), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(Subset(dataset, test_index), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, num_classes, num_views, dims


seeds = [0,1,2,3,4]
batch_size = 100

model_parameters ={
"dropout_p" : 0.1,
"annealing_start" : 50,
"model_epochs":200,
"model_hidden_dim" : (128,)
}

dataset_lr = {'CalTech':0.0003,'Scene':0.01,'CUB':0.003,'HandWritten':0.003,'PIE':0.003}



rows = {}
for seed in seeds:
    pl.seed_everything(seed)
    rows[seed] = {}
    ###  Normal Loop
    rows[seed]['Normal'] =  {}
    for dataset_name in ['CUB']:#, 'CalTech', 'HandWritten', 'PIE', 'Scene']:
        rows[seed]['Normal'][dataset_name] = {}
        train_loader, test_loader, num_classes, num_views, dims = get_normal_data(dataset_name, batch_size)
        model_parameters["classes"] = num_classes
        model_parameters["output_dims"] = dims
        model_parameters['classifiers'] = [(IdentityEncoder, {}) for i in range(len(dims))]
        model_parameters['lr']  = dataset_lr[dataset_name]

        dmvae_model = DMVAE(feature_encoders=model_parameters['classifiers'], output_dim=model_parameters['output_dims'], dropout=0, a=1e-5, hidden_dim=512,  
                        embed_dim=200, lr=1e-4, num_epochs=100)
        

        trainer = pl.Trainer(
            accelerator="auto",
            max_epochs=100,
            enable_progress_bar=True,
            log_every_n_steps=20,
        )
        trainer.fit(dmvae_model, train_dataloaders=train_loader)
        model_name = f'dmvae_dataset{dataset_name}_seed{seed}_a{1e-5}_normal'
        path = f'checkpoints/{model_name}.ckpt'
        trainer.save_checkpoint(path)

        dis_dmvae_model = DisentangledEvidentialProbeModule(dmvae_model, num_classes=model_parameters['classes'], lr=model_parameters['lr'], annealing_start=model_parameters['annealing_start'], 
                            hidden_dim=model_parameters['model_hidden_dim'],  dropout=model_parameters['dropout_p'],  input_dim=200)
        cml_dmvae_model = EvidentialProbeModule(dmvae_model, num_classes=model_parameters['classes'], lr=model_parameters['lr'], annealing_start=model_parameters['annealing_start'], 
                            hidden_dim=model_parameters['model_hidden_dim'],  dropout=model_parameters['dropout_p'], aggregation='cml', input_dim=200)
        joint_dmvae_model =  EvidentialProbeModule(dmvae_model, num_classes=model_parameters['classes'], lr=model_parameters['lr'], annealing_start=model_parameters['annealing_start'], 
                            hidden_dim=model_parameters['model_hidden_dim'],  dropout=model_parameters['dropout_p'], aggregation='joint', input_dim=200)
        
        dbf_fusion = baselines.LateFusion(model_parameters["classifiers"], model_parameters["output_dims"], model_parameters["classes"], dropout=model_parameters["dropout_p"], 
                            aggregation='dbf', lr=model_parameters['lr'], annealing_start=model_parameters['annealing_start'], hidden_dim=model_parameters['model_hidden_dim'])
        cml_fusion = baselines.LateFusion(model_parameters["classifiers"], model_parameters["output_dims"], model_parameters["classes"], dropout=model_parameters["dropout_p"], 
                            aggregation='cml', lr=model_parameters['lr'], annealing_start=model_parameters['annealing_start'], hidden_dim=model_parameters['model_hidden_dim'])
        avg_fusion = baselines.LateFusion(model_parameters["classifiers"], model_parameters["output_dims"], model_parameters["classes"], dropout=model_parameters["dropout_p"], 
                            aggregation='avg', lr=model_parameters['lr'], annealing_start=model_parameters['annealing_start'], hidden_dim=model_parameters['model_hidden_dim'])
        
        models = [dis_dmvae_model, cml_dmvae_model, joint_dmvae_model, dbf_fusion, cml_fusion, avg_fusion]
        model_names = ['dmvae_dis','dmvae_cml','dmvae_joint','dbf_fusion','cml_fusion','avg_fusion']
        
        
        for model, name in zip(models, model_names):
            model_name = f'{name}_fusion_ds{dataset_name}_seed{seed}'
            print(f'Training model {model_name}')
            tags=[name,str(seed), dataset_name,'Normal']
            model_parameters.update({"dataset": dataset_name, "seed": seed,'aggregation':name ,'UQ':'Normal'})
            wandb_logger = WandbLogger(
                project="MDU",
                entity="hassan-sarwat-technical-university-of-munich",
                name=model_name,
                tags=tags,
                log_model=True,
                config=model_parameters,
            )

            gpus = 1 if torch.cuda.is_available() else 0
            trainer = pl.Trainer(
                accelerator="auto" if gpus else "cpu",
                devices=gpus if gpus else None,
                max_epochs=model_parameters['model_epochs'],
                log_every_n_steps=20, # For speed
                enable_progress_bar=True,
                enable_model_summary=False,
                logger=wandb_logger        # turn off TensorBoard for brevity
            )

            # Lightning will call model.training_step & model.validation_step
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
            # Load best checkpoint after training
            # Test best model on the test set
            test_metrics = trainer.test(model, dataloaders=test_loader, verbose=False)
            path = f'checkpoints/{model_name}.ckpt'
            trainer.save_checkpoint(path)
            print(f'Model {model_name}')
            pprint(test_metrics)
            if name == 'dmvae_dis':
                rows[seed]['Normal'][dataset_name][name] = evaluate_subjective_model(model, test_loader)
            else:    
                rows[seed]['Normal'][dataset_name][name] = evaluate_subjective_model_with_shared(model, test_loader)
            rows[seed]['Normal'][dataset_name][name].update({'path':path})
            # wandb_logger.log_metrics(test_metrics, step=trainer.global_step)
            wandb_logger.experiment.finish()

    rows[seed]['Conflict'] = {}
    for dataset_name in ['CUB']:#, 'CalTech', 'HandWritten', 'PIE', 'Scene']:
        rows[seed]['Conflict'][dataset_name] = {}
        train_loader, test_loader, num_classes, num_views, dims = get_conflict_data(dataset_name, batch_size)
        model_parameters["classes"] = num_classes
        model_parameters["output_dims"] = dims
        model_parameters['classifiers'] = [(IdentityEncoder, {}) for _ in range(len(dims))]
        model_parameters['lr']  = dataset_lr[dataset_name]

        dmvae_model = DMVAE(feature_encoders=model_parameters['classifiers'], output_dim=model_parameters['output_dims'], dropout=0, a=1e-5, hidden_dim=512,  
                        embed_dim=200, lr=1e-4, num_epochs=100)
        

        trainer = pl.Trainer(
            accelerator="auto",
            max_epochs=100,
            enable_progress_bar=True,
            log_every_n_steps=20,
        )
        trainer.fit(dmvae_model, train_dataloaders=train_loader)
        model_name = f'dmvae_dataset{dataset_name}_seed{seed}_a{1e-5}_conflict'
        path = f'checkpoints/{model_name}.ckpt'
        trainer.save_checkpoint(path)

        dis_dmvae_model = DisentangledEvidentialProbeModule(dmvae_model, num_classes=model_parameters['classes'], lr=model_parameters['lr'], annealing_start=model_parameters['annealing_start'], 
                            hidden_dim=model_parameters['model_hidden_dim'],  dropout=model_parameters['dropout_p'],  input_dim=200)
        cml_dmvae_model = EvidentialProbeModule(dmvae_model, num_classes=model_parameters['classes'], lr=model_parameters['lr'], annealing_start=model_parameters['annealing_start'], 
                            hidden_dim=model_parameters['model_hidden_dim'],  dropout=model_parameters['dropout_p'], aggregation='cml', input_dim=200)
        joint_dmvae_model =  EvidentialProbeModule(dmvae_model, num_classes=model_parameters['classes'], lr=model_parameters['lr'], annealing_start=model_parameters['annealing_start'], 
                            hidden_dim=model_parameters['model_hidden_dim'],  dropout=model_parameters['dropout_p'], aggregation='joint', input_dim=200)
        
        dbf_fusion = baselines.LateFusion(model_parameters["classifiers"], model_parameters["output_dims"], model_parameters["classes"], dropout=model_parameters["dropout_p"], 
                            aggregation='dbf', lr=model_parameters['lr'], annealing_start=model_parameters['annealing_start'], hidden_dim=model_parameters['model_hidden_dim'])
        cml_fusion = baselines.LateFusion(model_parameters["classifiers"], model_parameters["output_dims"], model_parameters["classes"], dropout=model_parameters["dropout_p"], 
                            aggregation='cml', lr=model_parameters['lr'], annealing_start=model_parameters['annealing_start'], hidden_dim=model_parameters['model_hidden_dim'])
        avg_fusion = baselines.LateFusion(model_parameters["classifiers"], model_parameters["output_dims"], model_parameters["classes"], dropout=model_parameters["dropout_p"], 
                            aggregation='avg', lr=model_parameters['lr'], annealing_start=model_parameters['annealing_start'], hidden_dim=model_parameters['model_hidden_dim'])
        
        models = [dis_dmvae_model, cml_dmvae_model, joint_dmvae_model, dbf_fusion, cml_fusion, avg_fusion]
        model_names = ['dmvae_dis','dmvae_cml','dmvae_joint','dbf_fusion','cml_fusion','avg_fusion']

        for model, name in zip(models,model_names):
            model_name = f'{name}_fusion_ds{dataset_name}_seed{seed}_conflict'
            print(f'Training model {model_name}')
            tags=[name,str(seed),dataset_name,'Conflict']
            model_parameters.update({"dataset": dataset_name, "seed": seed,'aggregation':name, 'UQ':'Conflict'})
            wandb_logger = WandbLogger(
                project="MDU",
                entity="hassan-sarwat-technical-university-of-munich",
                name=model_name,
                tags=tags,
                log_model=True,
                config=model_parameters,
            )
            gpus = 1 if torch.cuda.is_available() else 0
            trainer = pl.Trainer(
                accelerator="auto" if gpus else "cpu",
                devices=gpus if gpus else None,
                max_epochs=model_parameters['model_epochs'],
                log_every_n_steps=20, # For speed
                enable_progress_bar=True,
                enable_model_summary=False,
                logger=wandb_logger        # turn off TensorBoard for brevity
            )

            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
            test_metrics = trainer.test(model, dataloaders=test_loader, verbose=False)
            path = f'checkpoints/{model_name}.ckpt'
            trainer.save_checkpoint(path)
            print(f'Model {model_name}')
            pprint(test_metrics)
            if name == 'dmvae_dis':
                rows[seed]['Conflict'][dataset_name][name] = evaluate_subjective_model(model, test_loader)
            else:    
                rows[seed]['Conflict'][dataset_name][name] = evaluate_subjective_model_with_shared(model, test_loader)
            rows[seed]['Conflict'][dataset_name][name].update({'path':path})
            # wandb_logger.log_metrics(test_metrics, step=trainer.global_step)
            wandb_logger.experiment.finish()


df = build_metrics_dataframe_datasets(rows)
df['seed'] = df['seed'].astype(int)
df_main = df[['seed','type','dataset','model','view_0_evidence_mean','view_1_evidence_mean', 'shared_evidence_mean', 'fused_evidence_mean',
                  'view_0_aleatoric_mean', 'view_1_aleatoric_mean','shared_aleatoric_mean','fused_aleatoric_mean',
                  'view_0_epistemic_mean','view_1_epistemic_mean','shared_epistemic_mean','fused_epistemic_mean',
                 'view_0_accuracy',  'view_1_accuracy', 'shared_accuracy', 'fused_accuracy']]

df_grouped = df.groupby(['type','dataset','model']).mean().reset_index()
df_grouped.sort_values(by=['type','dataset','model'],inplace=True)
df_main_grouped = df_main.groupby(['type','dataset','model']).mean().reset_index()
df_main_grouped.sort_values(by=['type','dataset','model'],inplace=True)
with  pd.ExcelWriter('logs/dataset_analysis.xlsx') as writer:
    df_main_grouped.to_excel(writer, sheet_name='main_grouped',index=False)
    df.to_excel(writer, sheet_name='all_results',index=False)
    df_grouped.to_excel(writer,sheet_name='grouped_results',index=False)



