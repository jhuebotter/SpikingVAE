import torch
import torch.nn as nn
import math
from glob import glob
from pathlib import Path
from tqdm import tqdm

from logger import Logger


class BaseModel:
    """Custom neural network base model."""

    def __init__(
        self,
        input_width,
        input_height,
        input_channels,
        dataset,
        learning_rate,
        weight_decay,
        device,
        log_interval,
        verbose=False,
        log_func=print,
    ):

        self.dataset = dataset
        # set input dimensions
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.input_sz = (self.input_channels, self.input_width, self.input_height)

        self.lr = learning_rate
        self.wd = weight_decay
        self.device = device
        self.log_interval = log_interval
        self.log_func = log_func

        # To be implemented by subclasses
        self.model = None
        self.optimizer = None
        self.metrics = {}

        self.verbose = verbose

    def count_parameters(self):
        """Helper function to count trainable parameters of the model."""

        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def init_weights(self):
        """Initializes network weights."""

        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                variance1 = math.sqrt(2.0 / n)
                m.weight.data.normal_(0, variance1)
                # define threshold
                m.threshold = 1

            elif isinstance(m, nn.Linear):
                size = m.weight.size()
                fan_in = size[1]
                variance2 = math.sqrt(2.0 / fan_in)
                m.weight.data.normal_(0.0, variance2)
                # define threshold
                m.threshold = 1

    def loss_function(self):
        """Loss function applied during learning is to be implemented by subclass."""

        raise NotImplementedError

    def update_(self):
        """Update some internal variables during training such as a scheduled or conditional learning rate."""

        pass

    def step(self, data, target, train=False):
        """Pass data through the model for training or evaluation.

        :param data: Tensor, batch of features from DataLoader object.
        :param target: Tensor, batch of labels from DataLoader object.
        :param train: bool, flag if model weights should be updated based on loss.

        :return result: dict, contains performance metric information.
        """

        # TODO: CHECK HOW THE METRICS REALLY WORK!

        if train:
            self.optimizer.zero_grad()
        result = dict()
        out = self.model(data)
        loss = self.loss_function(out, target)  # , train=train)
        result["loss"] = loss.item()
        for k, v in self.metrics.items():
            result[k] = v(out, y)
        if train:
            loss.backward()
            self.optimizer.step()

        return result  # loss.item()

    def train(self, train_loader, epoch):
        """Train the model for a single epoch.

        :param train_loader: DataLoader object, containing training data.
        :param epoch: int, current training epoch.

        :return train_loss: int, average loss on training dataset.
        """

        self.model.train()
        train_loss = 0

        # end = time.time()
        for batch_idx, (X, y) in enumerate(tqdm(train_loader)):
            X = X.to(self.device)
            y = y.to(self.device)
            result = self.step(X, y, train=True)
            loss = result["loss"]
            train_loss += loss
            if batch_idx and self.verbose and batch_idx % self.log_interval == 0:
                self.log_func(
                    f"Train Epoch: {epoch} [{batch_idx * len(X)}/{len(train_loader.dataset)} ({100.0 * batch_idx / len(train_loader):.1f}%)]\tLoss: {loss / len(X):.6f}"
                )
        if self.verbose:
            self.log_func(
                f"====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}"
            )

        train_loss /= len(train_loader.dataset)

        return train_loss

    def test(self, test_loader):
        """Evaluate the model on a validation or test set.

        :param test_loader: DataLoader object, containing test dataset.

        :return test_loss: int, average loss on training dataset.
        """

        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in tqdm(test_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                result = self.step(data, target, train=False)
                loss = result["loss"]
                test_loss += loss

        VLB = test_loss / len(test_loader)
        # Optional to normalize VLB on testset
        name = self.model.__class__.__name__
        test_loss /= len(test_loader.dataset)
        if self.verbose:
            self.log_func(
                f"====> Test set loss: {test_loss:.4f} - VLB-{name} : {VLB:.4f}"
            )

        return test_loss

    def load_last_model(self, checkpoints_path):
        """Load a pretrained model from the saved checkpoints.

        :param checkpoints_path: str, path to load a pretrained model from.

        :return start_epoch: int, epoch number to start training from.
        """

        name = self.model.__class__.__name__
        # Search for all previous checkpoints
        models = glob(f"{checkpoints_path}/*.pth")
        model_ids = []
        for f in models:

            run_name = Path(f).stem
            model_name, dataset, _, _, _, epoch = run_name.split(
                "_"
            )  # modelname_dataset_startepoch_epochs_lr_epoch
            # model_name, dataset, _, _, latent_sz, _, epoch = run_name.split(
            #    "_"
            # )  # modelname_dataset_startepoch_epochs_latentsize_lr_epoch
            if (
                model_name == name
                and dataset == self.dataset
                # and int(latent_sz) == self.latent_sz
            ):
                model_ids.append((int(epoch), f))

        # If no checkpoints available
        if len(model_ids) == 0:
            self.log_func(f"Training {name} model from scratch...")
            return 1

        # Load model from last checkpoint
        start_epoch, last_checkpoint = max(model_ids, key=lambda item: item[0])
        self.log_func("Last checkpoint: ", last_checkpoint)
        self.model.load_state_dict(torch.load(last_checkpoint))
        self.log_func(f"Loading {name} model from last checkpoint ({start_epoch})...")

        return start_epoch + 1

    def run_training(
        self,
        train_loader,
        test_loader,
        epochs,
        save_interval,
        reload_model="",
        checkpoints_path="../results/checkpoints",
        logs_path="../results/logs",
        images_path="../results/images",
        start_epoch=None,
    ):
        """
        Runs training on the training set for the set number of epochs while continuously evaluating and logging.

        :param train_loader: DataLoader object containing the training set.
        :param test_loader: DataLoader object containing the validation set.
        :param epochs: int, number of epochs to train for.
        :param save_interval: int, number of episodes after which a report is triggered.
        :param reload_model: str, name of the model to reload from last checkpoint.
        :param checkpoints_path: str, directory to save and load model (default: ../results/checkpoints).
        :param logs_path: str, directory to save the logs to (default: ../results/logs).
        :param images_path: directory to save example images (default: ../results/images).
        :param start_epoch: epoch to start training from (default: None).

        """

        # Start fresh or load previous model
        if start_epoch is None:
            start_epoch = self.load_last_model(checkpoints_path) if reload_model else 1

        # Name the experiment for logging and saving
        name = self.model.__class__.__name__
        run_name = (
            f"{name}_{self.dataset}_{start_epoch}_{epochs}_"
            f"{str(self.lr).replace('.', '-')}"
        )
        # run_name = (
        #    f"{name}_{self.dataset}_{start_epoch}_{epochs}_"
        #    f"{self.latent_sz}_{str(self.lr).replace('.', '-')}"
        # )
        logger = Logger(f"{logs_path}/{run_name}")
        self.log_func(f"Training {name} model...")

        # Train for desired number of epochs
        for epoch in range(start_epoch, start_epoch + epochs):
            train_loss = self.train(train_loader, epoch)
            test_loss = self.test(test_loader)

            # Store log
            logger.add_scalars(
                "Loss", {"Train": train_loss, "Validation": test_loss}, epoch
            )

            # Optional update
            self.update_()

            # For each report interval store model
            if epoch % save_interval == 0:
                with torch.no_grad():
                    torch.save(
                        self.model.state_dict(),
                        f"{checkpoints_path}/{run_name}_{epoch}.pth",
                    )

        logger.close()


if __name__ == "__main__":
    model = BaseModel()
