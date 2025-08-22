# dataloader.py
import torch
from torchvision import datasets, transforms

def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True
):
    """
    data_dir: 데이터셋이 들어 있는 상위 폴더
              ├── train
              │    └── class1, class2, ...
              └── test
                   └── class1, class2, ...
    batch_size, num_workers, pin_memory은 외부에서 지정 가능
    """

    # 데이터 전처리 (필요에 맞게 수정 가능)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # torchvision.datasets.ImageFolder 이용
    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
    test_dataset  = datasets.ImageFolder(root=f"{data_dir}/test",  transform=transform)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, test_loader
