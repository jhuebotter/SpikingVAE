import torch
import torch.nn as nn
import math
from glob import glob
from pathlib import Path
from tqdm import tqdm
import utils as u
import metrics as m
import shutil

#import wandb
#from logger import WandBLogger


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
        self.log_func = log_func

        # To be implemented by subclasses
        self.model = None
        self.optimizer = None
        self.task = None
        #self.metrics = {}

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

    def step(self, data_batch, target_batch, train=False):
        """Pass data through the model for training or evaluation.

        :param data_batch: Tensor, batch of features from DataLoader object.
        :param target_batch: Tensor, batch of labels from DataLoader object.
        :param train: bool, flag if model weights should be updated based on loss.

        :return result: dict, contains performance metric information.
        """

        if train:
            self.optimizer.zero_grad()
        out_batch = self.model(data_batch)
        loss = self.loss_function(out_batch["output"], target_batch)  # , train=train)
        result = dict(
            input=data_batch,
            target=target_batch,
            loss=loss.item(),
        )
        result.update(out_batch)

        if train:
            loss.backward()
            self.optimizer.step()

        return result

    def train(self, train_loader, epoch, metrics={}):
        """Train the model for a single epoch.

        :param train_loader: DataLoader object, containing training data.
        :param epoch: int, current training epoch.

        :return train_loss: int, average loss on training dataset.
        """

        self.model.train()

        summary = {metric: u.RunningAverage() for metric in metrics.keys()}

        loss_avg = u.RunningAverage()

        with tqdm(total=len(train_loader)) as t:
            for batch_idx, (data_batch, target_batch) in enumerate(train_loader):
                data_batch = data_batch.to(self.device)
                target_batch = target_batch.to(self.device)
                if self.task == "reconstruction":
                    target_batch = data_batch
                result = self.step(data_batch, target_batch, train=True)
                loss = result["loss"]

                # update the average loss
                loss_avg.update(loss / len(target_batch))

                # calculate metrics
                for metric, metric_fn in metrics.items():
                    summary[metric].update(metric_fn(**result))

                t.set_postfix(loss="{:05.4f}".format(loss_avg()))
                t.update()

        # create the results dict
        train_results = dict(loss=loss_avg())
        for metric in metrics.keys():
            train_results[metric] = summary[metric]()

        if self.verbose:
            self.log_func(f"====> Epoch {epoch}: Average loss = {loss_avg():.4f}")

        return train_results

    def evaluate(self, test_loader, metrics={}, samplers={}, sample_freq=0):
        """Evaluate the model on a validation or test set.

        :param test_loader: DataLoader object, containing test dataset.

        :return test_loss: int, average loss on training dataset.
        """

        self.model.eval()

        summary = {metric: u.RunningAverage() for metric in metrics.keys()}

        samples = {}

        loss_avg = u.RunningAverage()

        with torch.no_grad():
            with tqdm(total=len(test_loader)) as t:
                for batch_idx, (data_batch, target_batch) in enumerate(test_loader):
                    data_batch = data_batch.to(self.device)
                    target_batch = target_batch.to(self.device)
                    if self.task == "reconstruction":
                        target_batch = data_batch
                    result = self.step(data_batch, target_batch, train=False)
                    loss = result["loss"]

                    # update the average loss
                    loss_avg.update(loss / len(target_batch))
                    t.set_postfix(loss="{:05.4f}".format(loss_avg()))
                    t.update()

                    # calculate the metrics
                    for metric, metric_fn in metrics.items():
                        summary[metric].update(metric_fn(**result))

                    if sample_freq and batch_idx % sample_freq == 0:
                        for sampler, sampler_fn in samplers.items():
                            samples.update({f"batch {batch_idx}": sampler_fn(**result)})

        # create the results dict
        test_results = dict(loss=loss_avg())
        for metric in metrics.keys():
            test_results[metric] = summary[metric]()

        if self.verbose:
            name = self.model.__class__.__name__
            self.log_func(f"====> Test: {name} Average loss = {loss_avg():.4f}")

        return test_results, samples

    '''
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

    
    def save_model(self, dir, name, verbose=True):

        path = Path.joinpath(Path(dir), f"{name}.pth")
        torch.save(self.model.state_dict(), path)

        if verbose:
            self.log_func(f"Saved model at {path}\n")
    '''

    def save_checkpoint(self, checkpoint, dir, name, is_best, verbose=True):

        Path.mkdir(Path(dir), exist_ok=True)
        path = Path.joinpath(Path(dir), f"{name}_latest.pth")
        torch.save(checkpoint, path)
        if is_best:
            best_path = Path.joinpath(Path(dir), f"{name}_best.pth")
            shutil.copyfile(path, best_path)

        if verbose:
            self.log_func(f"Saved model at {path}\n")
            if is_best:
                self.log_func(f"Saved model at {best_path}\n")

    def load_checkpoint(self, path, model, optimizer=None, verbose=True):

        path = Path(path)

        if self.verbose:
            self.log_func(f"Loading model parameters from checkpoint {path}...")

        if not Path.exists(path):
            raise FileNotFoundError(f"File {path} doesn't exist")

        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["state_dict"])

        if optimizer:
            optimizer.load_state_dict(checkpoint["optim_dict"])

        if verbose:
            self.log_func("Found checkpoint entries:")
            for key in checkpoint.keys():
                self.log_func(key)
            self.log_func(f"Loaded model state from {path}\n")
            if optimizer:
                self.log_func(f"Loaded optimizer state from {path}\n")

        return checkpoint

    def train_and_evaluate(
        self,
        train_loader,
        val_loader,
        epochs,
        model_name,
        metrics=[],
        key_metric="validation loss",
        goal="minimize",
        load="",
        logger=None,
        checkpoints_dir="../results/checkpoints",
        eval_first=True,
        samplers=[],
        sample_freq=0,
    ):
        """
        Runs training on the training set for the set number of epochs while continuously evaluating and logging.

        :param train_loader: DataLoader object containing the training set.
        :param val_loader: DataLoader object containing the validation set.
        :param epochs: int, number of epochs to train for.
        # :param save_interval: int, number of episodes after which a report is triggered.
        :param load: str, path of the model to reload from last checkpoint.
        :param checkpoints_dir: str, directory to save and load model (default: ../results/checkpoints).
        # :param logs_dir: str, directory to save the logs to (default: ../results/logs).
        # :param images_path: directory to save example images (default: ../results/images).
        # :param start_epoch: epoch to start training from (default: None).

        """

        # Start fresh or load previous model
        if load:
            checkpoint = self.load_checkpoint(
                path=load,
                model=self.model,
                optimizer=self.optimizer,
                verbose=self.verbose,
            )
            start_epoch = checkpoint["epoch"] + 1
            training_summary = checkpoint["training_summary"]
        else:
            start_epoch = 1
            training_summary = {}

        # initialize metrics
        metrics = m.get_metrics(metrics)

        # initialize samplers
        samplers = m.get_samplers(samplers)

        if goal.lower() == "minimize":
            lower_is_better = True
            best_key_score = math.inf
        elif goal.lower() == "maximize":
            lower_is_better = False
            best_key_score = -math.inf

        if eval_first:
            val_results, samples = self.evaluate(val_loader, metrics, samplers, sample_freq)
            # Store results in log
            summary = dict()
            for key, value in val_results.items():
                summary[f"validation {key}"] = value
            if logger is not None:
                logger.log(summary, step=start_epoch - 1)
                for batch, sample in samples.items():
                    for key, value in sample.items():
                        logger.log({f"{batch}_{key}": value}, step=start_epoch - 1)

        if self.verbose:
            self.log_func(f"Training {model_name} model for {epochs} epochs...")

        # Train for desired number of epochs
        for epoch in range(start_epoch, start_epoch + epochs):
            train_results = self.train(train_loader, epoch, metrics)
            val_results, samples = self.evaluate(val_loader, metrics, samplers, sample_freq)

            train_loss = train_results["loss"]
            val_loss = val_results["loss"]

            # Store results in log
            summary = dict()
            for key, value in train_results.items():
                summary[f"training {key}"] = value
            for key, value in val_results.items():
                summary[f"validation {key}"] = value
            training_summary[epoch] = summary
            if logger is not None:
                logger.log(summary, step=epoch)
                for batch, sample in samples.items():
                    for key, value in sample.items():
                        logger.log({f"{batch}_{key}": value}, step=epoch)

            if self.verbose:
                self.log_func("Episode summary:")
                for key, value in summary.items():
                    self.log_func(f"{key:30s} {value:10.4f}")

            # Check if the model is the best model
            key_score = summary[key_metric]
            is_best = False
            if (key_score <= best_key_score and lower_is_better) or (
                key_score >= best_key_score and not lower_is_better
            ):
                is_best = True
                best_key_score = key_score
                if self.verbose:
                    self.log_func(f"New best model: {key_metric}    {best_key_score:.4f} \n")

            # Save the latest model
            checkpoint = dict(
                epoch=epoch,
                state_dict=self.model.state_dict(),
                optim_dict=self.optimizer.state_dict(),
                train_loss=train_loss,
                val_loss=val_loss,
                key_score=key_score,
                training_summary=training_summary,
                run_id=logger.run.id if logger is not None else None,
            )
            self.save_checkpoint(
                checkpoint=checkpoint,
                dir=checkpoints_dir,
                name=model_name,
                is_best=is_best,
                verbose=self.verbose,
            )

            # Optional update
            self.update_()


class SpikingBaseModel:
    """Custom neural network base model."""

    def __init__(
            self,
            input_width,
            input_height,
            input_channels,
            dataset,
            learning_rate,
            weight_decay,
            steps,
            threshold,
            decay,
            device,
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
        self.steps = steps
        self.device = device
        self.log_func = log_func
        self.decay = decay
        self.threshold = threshold

        # To be implemented by subclasses
        self.model = None
        self.optimizer = None
        self.task = None
        # self.metrics = {}

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

    def grad_cal(self, decay, LF_output, Total_output):
        Total_output = Total_output + (Total_output < 1e-3).type(torch.cuda.FloatTensor)
        out = LF_output.gt(1e-3).type(torch.cuda.FloatTensor) + math.log(decay) * torch.div(LF_output, Total_output)
        print(out.size())
        return out

    def step(self, data_batch, target_batch, train=False):
        """Pass data through the model for training or evaluation.

        :param data_batch: Tensor, batch of features from DataLoader object.
        :param target_batch: Tensor, batch of labels from DataLoader object.
        :param train: bool, flag if model weights should be updated based on loss.

        :return result: dict, contains performance metric information.
        """

        if train:
            self.optimizer.zero_grad()
        out_batch = self.model(data_batch, steps=self.steps)

        for key, value in out_batch.items():
            print(f"{key}: {[len(v) for v in value]}")

        if train:
            # compute gradient
            gradients = [self.grad_cal(self.decay, out_batch["lf_outs"][i], out_batch["total_outs"][i]) for i in range(len(out_batch["total_outs"])-1)]
            print(self.steps)
            # apply gradient
            for t in range(self.steps):
                for i in range(len(gradients)):
                    out_batch["out_temps"][i][t].register_hook(lambda grad: torch.mul(grad, gradients[i]))

        targetN = out_batch["output"].data.clone().zero_()
        targetN.scatter_(1, target_batch.unsqueeze(1), 1)
        targetN = targetN.type(torch.FloatTensor).to(self.device)

        #print(out_batch["output"].size())
        #print(targetN.size())

        loss = self.loss_function(out_batch["output"], targetN)  # , train=train)
        result = dict(
            input=data_batch,
            target=target_batch,
            loss=loss.item(),
        )
        result.update(out_batch)




        if train:
            loss.backward()
            self.optimizer.step()

            for i in range(len(out_batch["out_temps"])-1):
                out_batch["out_temps"][i] = None
                gradients[i] = None

        return result

    def train(self, train_loader, epoch, metrics={}):
        """Train the model for a single epoch.

        :param train_loader: DataLoader object, containing training data.
        :param epoch: int, current training epoch.

        :return train_loss: int, average loss on training dataset.
        """

        self.model.train()

        summary = {metric: u.RunningAverage() for metric in metrics.keys()}

        loss_avg = u.RunningAverage()

        with tqdm(total=len(train_loader)) as t:
            for batch_idx, (data_batch, target_batch) in enumerate(train_loader):
                data_batch = data_batch.to(self.device)
                target_batch = target_batch.to(self.device)
                if self.task == "reconstruction":
                    target_batch = data_batch
                result = self.step(data_batch, target_batch, train=True)
                loss = result["loss"]

                # update the average loss
                loss_avg.update(loss / len(target_batch))

                # calculate metrics
                for metric, metric_fn in metrics.items():
                    summary[metric].update(metric_fn(**result))

                t.set_postfix(loss="{:05.4f}".format(loss_avg()))
                t.update()

        # create the results dict
        train_results = dict(loss=loss_avg())
        for metric in metrics.keys():
            train_results[metric] = summary[metric]()

        if self.verbose:
            self.log_func(f"====> Epoch {epoch}: Average loss = {loss_avg():.4f}")

        return train_results


    def evaluate(self, test_loader, metrics={}, samplers={}, sample_freq=0):
        """Evaluate the model on a validation or test set.

        :param test_loader: DataLoader object, containing test dataset.

        :return test_loss: int, average loss on training dataset.
        """

        self.model.eval()

        summary = {metric: u.RunningAverage() for metric in metrics.keys()}

        samples = {}

        loss_avg = u.RunningAverage()

        with torch.no_grad():
            with tqdm(total=len(test_loader)) as t:
                for batch_idx, (data_batch, target_batch) in enumerate(test_loader):
                    data_batch = data_batch.to(self.device)
                    target_batch = target_batch.to(self.device)
                    if self.task == "reconstruction":
                        target_batch = data_batch
                    result = self.step(data_batch, target_batch, train=False)
                    loss = result["loss"]

                    # update the average loss
                    loss_avg.update(loss / len(target_batch))
                    t.set_postfix(loss="{:05.4f}".format(loss_avg()))
                    t.update()

                    # calculate the metrics
                    for metric, metric_fn in metrics.items():
                        summary[metric].update(metric_fn(**result))

                    if sample_freq and batch_idx % sample_freq == 0:
                        for sampler, sampler_fn in samplers.items():
                            samples.update({f"batch {batch_idx}": sampler_fn(**result)})

        # create the results dict
        test_results = dict(loss=loss_avg())
        for metric in metrics.keys():
            test_results[metric] = summary[metric]()

        if self.verbose:
            name = self.model.__class__.__name__
            self.log_func(f"====> Test: {name} Average loss = {loss_avg():.4f}")

        return test_results, samples

    '''
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


    def save_model(self, dir, name, verbose=True):

        path = Path.joinpath(Path(dir), f"{name}.pth")
        torch.save(self.model.state_dict(), path)

        if verbose:
            self.log_func(f"Saved model at {path}\n")
    '''

    def save_checkpoint(self, checkpoint, dir, name, is_best, verbose=True):

        Path.mkdir(Path(dir), exist_ok=True)
        path = Path.joinpath(Path(dir), f"{name}_latest.pth")
        torch.save(checkpoint, path)
        if is_best:
            best_path = Path.joinpath(Path(dir), f"{name}_best.pth")
            shutil.copyfile(path, best_path)

        if verbose:
            self.log_func(f"Saved model at {path}\n")
            if is_best:
                self.log_func(f"Saved model at {best_path}\n")

    def load_checkpoint(self, path, model, optimizer=None, verbose=True):

        path = Path(path)

        if self.verbose:
            self.log_func(f"Loading model parameters from checkpoint {path}...")

        if not Path.exists(path):
            raise FileNotFoundError(f"File {path} doesn't exist")

        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["state_dict"])

        if optimizer:
            optimizer.load_state_dict(checkpoint["optim_dict"])

        if verbose:
            self.log_func("Found checkpoint entries:")
            for key in checkpoint.keys():
                self.log_func(key)
            self.log_func(f"Loaded model state from {path}\n")
            if optimizer:
                self.log_func(f"Loaded optimizer state from {path}\n")

        return checkpoint

    def train_and_evaluate(
            self,
            train_loader,
            val_loader,
            epochs,
            model_name,
            metrics=[],
            key_metric="validation loss",
            goal="minimize",
            load="",
            logger=None,
            checkpoints_dir="../results/checkpoints",
            eval_first=True,
            samplers=[],
            sample_freq=0,
    ):
        """
        Runs training on the training set for the set number of epochs while continuously evaluating and logging.

        :param train_loader: DataLoader object containing the training set.
        :param val_loader: DataLoader object containing the validation set.
        :param epochs: int, number of epochs to train for.
        # :param save_interval: int, number of episodes after which a report is triggered.
        :param load: str, path of the model to reload from last checkpoint.
        :param checkpoints_dir: str, directory to save and load model (default: ../results/checkpoints).
        # :param logs_dir: str, directory to save the logs to (default: ../results/logs).
        # :param images_path: directory to save example images (default: ../results/images).
        # :param start_epoch: epoch to start training from (default: None).

        """

        # Start fresh or load previous model
        if load:
            checkpoint = self.load_checkpoint(
                path=load,
                model=self.model,
                optimizer=self.optimizer,
                verbose=self.verbose,
            )
            start_epoch = checkpoint["epoch"] + 1
            training_summary = checkpoint["training_summary"]
        else:
            start_epoch = 1
            training_summary = {}

        # initialize metrics
        metrics = m.get_metrics(metrics)

        # initialize samplers
        samplers = m.get_samplers(samplers)

        if goal.lower() == "minimize":
            lower_is_better = True
            best_key_score = math.inf
        elif goal.lower() == "maximize":
            lower_is_better = False
            best_key_score = -math.inf

        if eval_first:
            val_results, samples = self.evaluate(val_loader, metrics, samplers, sample_freq)
            # Store results in log
            summary = dict()
            for key, value in val_results.items():
                summary[f"validation {key}"] = value
            if logger is not None:
                logger.log(summary, step=start_epoch - 1)
                for batch, sample in samples.items():
                    for key, value in sample.items():
                        logger.log({f"{batch}_{key}": value}, step=start_epoch - 1)

        if self.verbose:
            self.log_func(f"Training {model_name} model for {epochs} epochs...")

        # Train for desired number of epochs
        for epoch in range(start_epoch, start_epoch + epochs):
            train_results = self.train(train_loader, epoch, metrics)
            val_results, samples = self.evaluate(val_loader, metrics, samplers, sample_freq)

            train_loss = train_results["loss"]
            val_loss = val_results["loss"]

            # Store results in log
            summary = dict()
            for key, value in train_results.items():
                summary[f"training {key}"] = value
            for key, value in val_results.items():
                summary[f"validation {key}"] = value
            training_summary[epoch] = summary
            if logger is not None:
                logger.log(summary, step=epoch)
                for batch, sample in samples.items():
                    for key, value in sample.items():
                        logger.log({f"{batch}_{key}": value}, step=epoch)

            if self.verbose:
                self.log_func("Episode summary:")
                for key, value in summary.items():
                    self.log_func(f"{key:30s} {value:10.4f}")

            # Check if the model is the best model
            key_score = summary[key_metric]
            is_best = False
            if (key_score <= best_key_score and lower_is_better) or (
                    key_score >= best_key_score and not lower_is_better
            ):
                is_best = True
                best_key_score = key_score
                if self.verbose:
                    self.log_func(f"New best model: {key_metric}    {best_key_score:.4f} \n")

            # Save the latest model
            checkpoint = dict(
                epoch=epoch,
                state_dict=self.model.state_dict(),
                optim_dict=self.optimizer.state_dict(),
                train_loss=train_loss,
                val_loss=val_loss,
                key_score=key_score,
                training_summary=training_summary,
                run_id=logger.run.id if logger is not None else None,
            )
            self.save_checkpoint(
                checkpoint=checkpoint,
                dir=checkpoints_dir,
                name=model_name,
                is_best=is_best,
                verbose=self.verbose,
            )

            # Optional update
            self.update_()


if __name__ == "__main__":
    model = BaseModel()
