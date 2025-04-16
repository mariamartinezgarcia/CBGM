import torch
import matplotlib.pyplot as plt

from utils import get_dataset

# Create a test configuration
config = {
    "dataset": {
        "name": "fashion_mnist",
        "root_path": "/home/molotkova_s/data",
        "concept_dir_path": "/home/molotkova_s/Desktop/mlteam/data/FashionMNIST/concept_vectors",
        "test_batch_size": 5,
        "img_size": 28
    }
}

test_dataloader = get_dataset(config, istesting=True)

if test_dataloader is not None:
    batch = next(iter(test_dataloader))
    
    print(f"Batch type: {type(batch)}")
    
    if isinstance(batch, torch.Tensor):
        print(f"Batch shape: {batch.shape}")
    elif isinstance(batch, (tuple, list)):
        print(f"Batch contains {len(batch)} elements")
        for i, item in enumerate(batch):
            if isinstance(item, torch.Tensor):
                print(f"  Element {i} shape: {item.shape}")
            else:
                print(f"  Element {i} type: {type(item)}")
    elif isinstance(batch, dict):
        print(f"Batch contains {len(batch)} keys")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  Key {key} shape: {value.shape}")
            else:
                print(f"  Key {key} type: {type(value)}")
    
    dataset_size = len(test_dataloader.dataset)
    print(f"Total dataset size: {dataset_size}")
    print(f"Number of batches: {len(test_dataloader)}")
    
    print("\nDisplaying one item from the dataset:")
    sample_idx = 2

    sample = test_dataloader.dataset[sample_idx]
    
    # Expected item structure: tuple[Any, torch.Tensor, int]
    # Where elements are (image, concepts_tensor, label)
    if isinstance(sample, (tuple, list)) and len(sample) == 3:
        image, concepts, label = sample
        
        print(f"Sample structure: tuple of length {len(sample)}")
        
        if isinstance(image, torch.Tensor):
            print(f"  Image shape: {image.shape}")
            plt.figure(figsize=(5, 5))
            if image.shape[0] == 1:
                plt.imshow(image.squeeze().numpy(), cmap='gray')
            else:
                plt.imshow(image.permute(1, 2, 0).numpy())
            plt.title(f"Fashion MNIST Sample {sample_idx} (Label: {label})")
            plt.show()
        else:
            print(f"  Image type: {type(image)}")
        
        if isinstance(concepts, torch.Tensor):
            print(f"  Concepts tensor shape: {concepts.shape}")
            print(f"  Concepts values: {concepts}")
        else:
            print(f"  Concepts type: {type(concepts)}")

        print(f"  Label: {label} (type: {type(label)})")
        
    else:
        print(f"Unexpected sample structure: {type(sample)}")
        if isinstance(sample, (tuple, list)):
            print(f"Sample contains {len(sample)} elements")
            for i, item in enumerate(sample):
                print(f"  Element {i} type: {type(item)}")
                if isinstance(item, torch.Tensor):
                    print(f"    Shape: {item.shape}")
                elif hasattr(item, "__len__"):
                    print(f"    Length: {len(item)}")
                else:
                    print(f"    Value: {item}")
else:
    print("Dataloader is None. Check the implementation of get_dataset.")