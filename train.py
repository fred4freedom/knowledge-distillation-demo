import os
import argparse
import pickle
import json

import toml
import torch
import numpy as np

from utils.dataloader import create_loaders
import models
from trainers.epoch import TrainEpoch, ValidEpoch, DistillEpoch

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="Training mode: student, teacher, distill")
    parser.add_argument("--config", help="Configuration file", default="./config.toml")

    args = parser.parse_args()
    config = toml.load(args.config)
    config['training']['mode'] = args.mode

    return config



def train(config):
    """
    Perform training
    """

    # Get configurations

    # Data specific
    data_dir = config['data']['data_dir']
    valid_split = config['data']['validation_split']
    shuffle = config['data']['shuffle']
    random_seed = config['data']['random_seed']

    # Output specific
    save_interval = config['output']['save_interval']
    best_model_file = config['output']['best_model_file']
    model_file = config['output']['model_file']
    distill_teacher_model_file = config['output']['distill_teacher_model_file']
    model_subdir = config['output']['model_subdir']
    log_subdir = config['output']['log_subdir']
    student_dir = config['output']['student_dir']
    teacher_dir = config['output']['teacher_dir']
    distill_dir = config['output']['distill_dir']
    verbose = config['output']['verbose']

    # Training specific
    mode = config['training']['mode']
    use_cuda = config['training']['use_cuda']
    resume = config['training']['resume']
    small_model = config['training']['small_model']
    large_model = config['training']['large_model']
    temperature = config['training']['temperature']
    max_iter = config['training']['max_iter']
    start_iter = config['training']['start_iter']
    learning_rate = config['training']['learning_rate']
    teacher_weight = config['training']['teacher_weight']
    teacher_best_model_file = config['training']['teacher_best_model_file']
    

    if verbose:
        print("Configurations")
        print("==============")
        print(json.dumps(config, indent=2, sort_keys=True))
        print("==============")

    # Determine the device to train on
    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
    if verbose:
        print(f"Using {device} for training...")

    # Create models
    if mode == 'student':
        model_name = small_model
        teacher = None

        model_path = student_dir.format(model_name)
    elif mode == 'teacher':
        model_name = large_model
        teacher = None

        model_path = teacher_dir.format(model_name)
    else: # distill
        model_name = small_model

        teacher_path = teacher_dir.format(large_model)
        model_path = distill_dir.format(small_model, large_model, teacher_weight, temperature)
        teacher_path = os.path.join(teacher_path, model_subdir)
        teacher = torch.load(os.path.join(teacher_path, teacher_best_model_file))
        teacher = teacher.to(device)

    # Define model and log path
    log_path = os.path.join(model_path, log_subdir)    
    model_path = os.path.join(model_path, model_subdir)
    log_file = os.path.join(log_path, 'logs.p')

    # Create directories if they do not exists
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    if resume:
        if not os.path.exists(log_file):
            resume = False
        else:
            with open(log_file, 'rb') as f:
                logs = pickle.load(f)
            if start_iter > 0:
                logs['train_logs'] = logs['train_logs'][:start_iter]
                logs['valid_logs'] = logs['valid_logs'][:start_iter]
                logs['test_logs'] = None
                scores = [k['accuracy'] for k in logs['valid_logs']]
                idx = np.argmax(scores)
                logs['best_validation'] = logs['valid_logs'][idx]
                logs['best_iter'] = idx + 1
                logs['track_best'] = [k for k in logs['track_best'] if k[0] <= start_iter]
            else:
                start_iter = len(logs['train_logs'])
            cur_model_file = os.path.join(model_path, model_file.format(start_iter))
            if os.path.exists(cur_model_file):
                model = torch.load(cur_model_file)
            else:
                resume = False
            start_iter += 1

    if not resume:
        Model = getattr(models, model_name)
        model = Model().to(device)
        start_iter = 1
        logs = {
            'train_logs': [],
            'valid_logs': [],
            'test_logs': None,
            'best_validation': {'cross_entropy': 0, 'accuracy': 0},
            'best_iter': None,
            'track_best': []
        }

    # Save configurations
    if config is not None:
        config_outfile = os.path.join(log_path, 'config.toml')
        with open(config_outfile, 'w') as f:
            toml.dump(config, f)

    # Save teacher model (if exists)
    if teacher is not None:
        torch.save(teacher.cpu(), distill_teacher_model_file)

    # Get total parameters
    total_parameters = sum(p.numel() for p in model.parameters())
    total_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if verbose:
        print(f"Creating model... mode = {mode}, model = {model_name}, resume = {resume}, parameters = {total_parameters}, trainable parameters = {total_trainable_parameters}")
        print(f"Model path: {model_path}")
        if teacher is not None:
            print(f"Teacher model path: {teacher_path}")

    # Create loaders
    if verbose:
        print("Creating data loaders...")
    train_loader, valid_loader, _ = create_loaders(model, config, device)

    # Create optimizer
    if verbose:
        print(f"Using ADAM-W optimizer with learning rate {learning_rate}")

    optimizer = torch.optim.AdamW([ 
        dict(params=model.parameters(), lr=learning_rate),
    ])

    if mode == 'distill':
        train_epoch = DistillEpoch(
            model, 
            teacher=teacher,
            optimizer=optimizer,
            device=device,
            teacher_weight=teacher_weight,
            temperature=temperature,
            verbose=verbose
        )
    else:
        train_epoch = TrainEpoch(
            model,
            optimizer=optimizer,
            device=device,
            verbose=verbose
        )

    valid_epoch = ValidEpoch(
        model, 
        device=device,
        verbose=verbose
    )

    # Start training
    if verbose:
        print("Starting training...")
    for cur_iter in range(start_iter, max_iter+1):
        if verbose:
            print(f"Epoch {cur_iter}/{max_iter} Best Validation Score: {logs['best_validation']['accuracy']}")

        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)     

        logs['train_logs'].append(train_logs)
        logs['valid_logs'].append(valid_logs)
        
        if valid_logs['accuracy'] > logs['best_validation']['accuracy']:
            logs['best_iter'] = cur_iter
            logs['best_validation'] = valid_logs
            logs['track_best'].append((cur_iter, train_logs, valid_logs))

            # Save model
            if verbose:
                print("Saving model...")
            cur_model_file = os.path.join(model_path, best_model_file)
            torch.save(model, cur_model_file)

            # Save logs
            with open(log_file, 'wb') as f:
                pickle.dump(logs, f)

        if cur_iter % save_interval == 0:
            # Save logs
            with open(log_file, 'wb') as f:
                pickle.dump(logs, f)

            # Save model
            cur_model_file = os.path.join(model_path, model_file.format(cur_iter))
            torch.save(model, cur_model_file)

    # Training completed
    if verbose:
        print(f"Training completed... Best Validation Score: {logs['best_validation']['accuracy']}")
    
    # Perform testing
    if verbose:
        print("Evaluating on test data... Loading best model...")

    cur_model_file = os.path.join(model_path, best_model_file)        
    model = torch.load(cur_model_file).to(device)
    _, _, test_loader = create_loaders(model, config)
    test_epoch = ValidEpoch(
        model, 
        device=device,
        verbose=verbose
    )
    test_logs = test_epoch.run(test_loader) 
    logs['test_logs'] = test_logs

    # Save logs
    with open(log_file, 'wb') as f:
        pickle.dump(logs, f)

    # Print summary
    if verbose:
        print("Summary")
        print("=======")
        print(f"Validation: {logs['best_validation']['accuracy']} Testing: {logs['test_logs']['accuracy']}")



if __name__ == '__main__':
    config = parse_args()
    train(config)