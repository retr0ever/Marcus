#!/usr/bin/env python3
"""
Train Marcus on the LFW dataset.

Usage:
    python scripts/train_lfw.py
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import random


class LFWDataset(Dataset):
    """LFW dataset loader for triplet training."""
    
    def __init__(self, root_dir: str, transform=None, min_images_per_person: int = 2):
        self.root_dir = Path(root_dir)
        self.transform = transform or transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Build identity to images mapping
        self.identity_to_images = defaultdict(list)
        self.all_images = []
        
        print(f"Scanning {root_dir}...")
        
        for person_dir in sorted(self.root_dir.iterdir()):
            if not person_dir.is_dir():
                continue
                
            images = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))
            
            if len(images) >= min_images_per_person:
                identity = person_dir.name
                for img_path in images:
                    self.identity_to_images[identity].append(str(img_path))
                    self.all_images.append((str(img_path), identity))
        
        self.identities = list(self.identity_to_images.keys())
        print(f"Found {len(self.identities)} identities with {len(self.all_images)} images")
    
    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, idx):
        anchor_path, anchor_identity = self.all_images[idx]
        
        # Get positive (same identity, different image)
        positive_candidates = [p for p in self.identity_to_images[anchor_identity] if p != anchor_path]
        if not positive_candidates:
            positive_path = anchor_path
        else:
            positive_path = random.choice(positive_candidates)
        
        # Get negative (different identity)
        negative_identity = random.choice([i for i in self.identities if i != anchor_identity])
        negative_path = random.choice(self.identity_to_images[negative_identity])
        
        # Load images
        anchor = self._load_image(anchor_path)
        positive = self._load_image(positive_path)
        negative = self._load_image(negative_path)
        
        return anchor, positive, negative
    
    def _load_image(self, path: str):
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


class TripletLoss(nn.Module):
    """Triplet loss with hard margin."""
    
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        return loss.mean()


class SimpleEmbeddingNet(nn.Module):
    """Simple CNN for face embeddings."""
    
    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        
        self.conv = nn.Sequential(
            # 112x112 -> 56x56
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 56x56 -> 28x28
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 28x28 -> 14x14
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # 14x14 -> 7x7
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x


def train_on_lfw(
    dataset_path: str,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.0001,
    margin: float = 0.3,
    save_path: str = "models/lfw_trained.pth"
):
    """Train embeddings on LFW dataset."""
    
    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Load dataset
    print("\n=== Loading LFW Dataset ===")
    dataset = LFWDataset(dataset_path, min_images_per_person=2)
    
    if len(dataset) == 0:
        print("ERROR: No valid training samples found.")
        print(f"Check that {dataset_path} contains person folders with 2+ images each.")
        return
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    # Create model
    print("\n=== Initialising Model ===")
    model = SimpleEmbeddingNet(embedding_dim=512).to(device)
    
    # Loss and optimiser
    criterion = TripletLoss(margin=margin)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=3, gamma=0.5)
    
    # Training loop
    print("\n=== Starting Training ===")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Margin: {margin}")
    print(f"Device: {device}")
    print("-" * 50)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for anchor, positive, negative in progress:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            # Forward pass
            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)
            
            # Compute loss
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            
            # Backward pass
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimiser_state_dict': optimiser.state_dict(),
                'loss': avg_loss,
            }, save_path)
            print(f"  -> Saved best model (loss: {avg_loss:.4f})")
    
    print("\n=== Training Complete ===")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Model saved to: {save_path}")
    
    return model


if __name__ == "__main__":
    # Default path to your LFW dataset
    LFW_PATH = "/Users/selin/Desktop/PROJECTS/Computer Vision/Marcus/lfw/lfw-deepfunneled/lfw-deepfunneled"
    
    train_on_lfw(
        dataset_path=LFW_PATH,
        epochs=10,
        batch_size=32,
        learning_rate=0.0001,
        margin=0.3,
        save_path="models/lfw_trained.pth"
    )
