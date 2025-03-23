# Vision Transformer for Food Classification

## Overview
This project utilizes a Vision Transformer (ViT) model for image classification of various food items. It leverages a pre-trained ViT model from `torchvision` and fine-tunes it on a custom dataset.

## Features
- Uses a pre-trained ViT model (`ViT_B_16`)
- Fine-tunes the classification head for specific food categories
- Implements data loading, training, and evaluation
- Utilizes `torchvision` transforms for preprocessing
- Implements learning rate scheduling with `CosineAnnealingLR`
- Supports multi-worker data loading for efficiency

## Installation
### Prerequisites
Ensure you have the following dependencies installed:

```bash
pip install torch torchvision torchinfo matplotlib
```

## Dataset
The dataset is structured into training and testing directories:
```
food-101/
    ├── subset_train_v2/  # Training images
    ├── subset_test_v2/   # Testing images
```
Update the paths in the script accordingly:
```python
train_dir = 'E:/Deep learning/Project/OTHER_DATA/2/archive/food-101/food-101/sub_v2/subset_train_v2/'
test_dir = 'E:/Deep learning/Project/OTHER_DATA/2/archive/food-101/food-101/sub_v2/subset_test_v2/'
```

## Model Training
1. Load the pre-trained ViT model:
    ```python
    pretrained_vit = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT).to(device)
    ```
2. Modify the classifier head:
    ```python
    pretrained_vit.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(device)
    ```
3. Define optimizer, loss function, and scheduler:
    ```python
    optimizer = torch.optim.Adam(params=pretrained_vit.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    ```
4. Train the model:
    ```python
    results = engine.train(model=pretrained_vit,
                           train_dataloader=train_dataloader_pretrained,
                           test_dataloader=test_dataloader_pretrained,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           loss_fn=loss_fn,
                           epochs=2,
                           device=device)
    ```

## Results Visualization
After training, plot loss curves using:
```python
from helper_functions import plot_loss_curves
plot_loss_curves(results)
```

## Summary
This project demonstrates how to fine-tune a Vision Transformer model for food classification. It leverages `torchvision` and PyTorch’s built-in capabilities for model training and evaluation.

## License
MIT License

