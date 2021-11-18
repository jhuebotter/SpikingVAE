from pathlib import Path
import argparse
from torchvision import datasets, transforms


def download_MNIST(root):
    """Download MNIST dataset."""

    datasets.MNIST(
        root=root, train=True, download=True, transform=transforms.ToTensor()
    )


def download_FASHION(root):
    """Download Fashion-MNIST dataset."""

    datasets.FashionMNIST(
        root=root, train=True, download=True, transform=transforms.ToTensor()
    )

def download_CIFAR10(root):
    """Download CIFAR-10 dataset."""

    datasets.CIFAR10(
        root=root, train=True, download=True, transform=transforms.ToTensor()
    )

def download_CELEBA(root):
    """Download CELEBA dataset."""

    import gdown, zipfile

    # Path to folder with the dataset
    dataset_folder = f'{root}/img_align_celeba'
    # URL for the CelebA dataset
    url = 'https://drive.google.com/uc?id=1cNIac61PSA_LqDFYFUeyaQYekYPc75NH'
    # Path to download the dataset to
    download_path = f'{root}/img_align_celeba.zip'
    # Download the dataset from google drive
    gdown.download(url, download_path, quiet=False)
    # Unzip the downloaded file
    with zipfile.ZipFile(download_path, 'r') as ziphandler:
        ziphandler.extractall(dataset_folder)

    # this is still not fixed by pytorch dev team
    #datasets.CelebA(
    #    root=root, download=True, transform=transforms.ToTensor()
    #)


if __name__ == "__main__":
    """Setup result directories and download supported datasets."""

    # Create project folder organization
    #Path.mkdir(Path("results/logs"), parents=True, exist_ok=True)
    #Path.mkdir(Path("results/images"), parents=True, exist_ok=True)
    Path.mkdir(Path("results/checkpoints"), parents=True, exist_ok=True)

    # Read datasets to download
    parser = argparse.ArgumentParser(
        description="Download datasets for VSC experiments"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        choices=["mnist", "fashion", "cifar10", "celeba"],
        help="name of dataset to download [mnist, fashion, cifar10, celeba]",
        default="cifar10",
    )
    args = parser.parse_args()

    # Download datasets for experiments
    if "mnist" in args.datasets:
        print("Downloading MNIST dataset...")
        Path.mkdir(Path("data/mnist"), parents=True, exist_ok=True)
        download_MNIST("data/mnist")
    if "fashion" in args.datasets:
        print("Downloading Fashion-MNIST dataset...")
        Path.mkdir(Path("data/fashion-mnist"), parents=True, exist_ok=True)
        download_FASHION("data/fashion-mnist")
    if "cifar10" in args.datasets:
        print("Downloading CIFAR-10 dataset...")
        Path.mkdir(Path("data/cifar10"), parents=True, exist_ok=True)
        download_CIFAR10("data/cifar10")
    if "celeba" in args.datasets:
        print("Downloading CELEBA dataset...")
        Path.mkdir(Path("data/celeba"), parents=True, exist_ok=True)
        download_CELEBA("data/celeba")
