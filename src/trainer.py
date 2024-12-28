import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from tqdm import tqdm
import logging
import os
import wandb
from .models import ModelEnsemble, FocalLoss

logger = logging.getLogger(__name__)

class LearningRateScheduler:
    def __init__(self, optimizer, config):
        """Initialize learning rate scheduler with warmup
        
        Args:
            optimizer: PyTorch optimizer
            config: Training configuration dictionary
        """
        self.optimizer = optimizer
        self.config = config['training']
        self.current_step = 0
        # Calculate warmup steps
        self.warmup_steps = self.config['warmup_epochs'] * self.config.get('steps_per_epoch', 100)
        
    def step(self):
        """Update learning rate based on current step
        
        Returns:
            float: Current learning rate
        """
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            lr = self.config['max_lr'] * (self.current_step / self.warmup_steps)
        else:
            lr = self.config['max_lr'] * (0.1 ** (self.current_step // self.warmup_steps))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

class Trainer:
    def __init__(self, model, config, device):
        """Initialize trainer with model and training configuration
        
        Args:
            model: Neural network model
            config: Training configuration dictionary
            device: Computing device (CPU/GPU)
        """
        self.model = model
        self.config = config
        self.device = device
        
        self.criterion = FocalLoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['initial_lr'],
            weight_decay=config['training']['weight_decay']
        )
        self.scheduler = LearningRateScheduler(self.optimizer, config)
        
    def train_epoch(self, train_loader):
        """Train model for one epoch
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Check if data loader is empty
        if len(train_loader) == 0:
            logger.error("Training data loader is empty!")
            return 0.0, 0.0
            
        progress_bar = tqdm(train_loader, desc='Training')
        for batch_idx, (features, labels) in enumerate(progress_bar):
            try:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass and loss calculation
                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs.squeeze(), labels)
                
                # Backward pass and optimization
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['grad_clip']
                )
                self.optimizer.step()
                
                # Update learning rate
                lr = self.scheduler.step()
                
                # Calculate metrics
                total_loss += loss.item()
                predicted = (outputs.squeeze() > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': total_loss / (batch_idx + 1),
                    'acc': 100. * correct / total if total > 0 else 0.0,
                    'lr': lr
                })
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {str(e)}")
                continue
        
        # Calculate average metrics
        n_batches = len(train_loader)
        avg_loss = total_loss / n_batches if n_batches > 0 else 0
        accuracy = correct / total if total > 0 else 0
        
        return avg_loss, accuracy
        
    def evaluate(self, val_loader):
        """Evaluate model on validation data
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(features)
                loss = self.criterion(outputs.squeeze(), labels)
                total_loss += loss.item()
                
                predicted = (outputs.squeeze() > 0.5).float()
                predictions.extend(predicted.cpu().numpy())
                targets.extend(labels.cpu().numpy())
        
        # Calculate metrics
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        metrics = {
            'val_loss': total_loss / len(val_loader),
            'val_accuracy': (predictions == targets).mean(),
            'val_precision': precision_score(targets, predictions),
            'val_recall': recall_score(targets, predictions),
            'val_f1': f1_score(targets, predictions)
        }
        
        return metrics
        
    def train(self, train_loader, val_loader):
        """Train model with early stopping
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            
        Returns:
            float: Best validation F1 score
        """
        # Initialize WandB if enabled
        if self.config['wandb']['enabled']:
            self.wandb_run = wandb.init(
                project=self.config['wandb']['project_name'],
                config=self.config,
                mode=self.config['wandb']['mode']
            )
        else:
            self.wandb_run = None
            
        best_val_f1 = 0
        patience = self.config['training']['patience']
        patience_counter = 0
        
        for epoch in range(self.config['training']['num_epochs']):
            # Train epoch
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Evaluate
            metrics = self.evaluate(val_loader)
            
            # Log metrics
            logger.info(f'Epoch {epoch+1}/{self.config["training"]["num_epochs"]}')
            logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            for metric, value in metrics.items():
                logger.info(f'{metric}: {value:.4f}')
            
            # Log to WandB
            if self.wandb_run is not None:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    **metrics
                })
            
            # Early stopping check
            if metrics['val_f1'] > best_val_f1:
                best_val_f1 = metrics['val_f1']
                patience_counter = 0
                self.save_checkpoint(epoch, metrics, is_best=True)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f'\nEarly stopping triggered after {epoch+1} epochs')
                    break
        
        # Close WandB
        if self.wandb_run is not None:
            wandb.finish()
        
        return best_val_f1

    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of evaluation metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(
            self.config['paths']['checkpoints_dir'],
            f'checkpoint_epoch_{epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model separately
        if is_best:
            best_path = self.config['paths']['model_save_path']
            torch.save(checkpoint, best_path)

def k_fold_cross_validation(dataset, config, device):
    """Perform k-fold cross-validation training
    
    Args:
        dataset: Complete dataset
        config: Training configuration
        device: Computing device (CPU/GPU)
        
    Returns:
        dict: Average metrics across all folds
    """
    splits = StratifiedKFold(
        n_splits=config['training']['k_folds'], 
        shuffle=True, 
        random_state=42
    )
    
    labels = [label.item() for _, label in dataset]
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(splits.split(range(len(dataset)), labels)):
        logger.info(f'\nFold {fold + 1}/{config["training"]["k_folds"]}')
        
        # Create data loaders
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(
            dataset,
            batch_size=config['training']['batch_size'],
            sampler=train_sampler,
            num_workers=config['training']['num_workers']
        )
        
        val_loader = DataLoader(
            dataset,
            batch_size=config['training']['batch_size'],
            sampler=val_sampler,
            num_workers=config['training']['num_workers']
        )
        
        # Initialize model
        model = ModelEnsemble(
            config['model']['input_size'],
            config
        ).to(device)
        
        # Train model
        trainer = Trainer(model, config, device)
        best_f1 = trainer.train(train_loader, val_loader)
        
        # Load best model
        checkpoint = torch.load(config['paths']['model_save_path'])
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate best model
        metrics = trainer.evaluate(val_loader)
        fold_metrics.append(metrics)
        
        logger.info(f'Fold {fold + 1} metrics:')
        for metric, value in metrics.items():
            logger.info(f'{metric}: {value:.4f}')
    
    # Calculate average metrics
    avg_metrics = {}
    for metric in fold_metrics[0].keys():
        avg_metrics[metric] = np.mean([fold[metric] for fold in fold_metrics])
    
    logger.info('\nAverage metrics across all folds:')
    for metric, value in avg_metrics.items():
        logger.info(f'{metric}: {value:.4f}')
    
    return avg_metrics