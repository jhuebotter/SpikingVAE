import torch
import torch.nn as nn
import math
from pathlib import Path
from tqdm import tqdm
import utils as u
import metrics as met
import samplers as sam
import losses
import shutil
import time

class BaseModel:
    """Custom neural network base model."""

    def __init__(
        self,
        input_width,
        input_height,
        input_channels,
        #dataset,
        learning_rate,
        weight_decay,
        device,
        loss,
        verbose=True,
        log_func=print,
    ):
        """ Initializes a custom neural network base model

        Args:
            input_width (int): pixel width of input data
            input_height (int): pixel height of input data
            input_channels (int): number of channels in input data, grayscale = 1, RGB = 3
            learning_rate (float): step size for parameter updates during learning
            weight_decay (float): weight of the L2 regularization applied during learning
            device (str): device where model parameters are stored and trained on
            loss (str): name of the loss function to use during training.
                Choice must be a valid option defined in losses.py
            verbose (bool, optional): flag to determine if process summary is given (default: True)
            log_func (function, optional): function to use for summary output (default: print)

        """

        #self.dataset = dataset
        # set input dimensions
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.input_sz = (self.input_channels, self.input_width, self.input_height)

        self.lr = learning_rate
        self.wd = weight_decay
        self.device = device
        self.log_func = log_func
        self.training_step = 0

        # To be implemented by subclasses
        self.model = None
        self.optimizer = None
        self.task = None
        self.input_layer = None
        self.output_layer = None

        # initialize loss function
        self.loss_function = loss

        self.verbose = verbose

    def count_parameters(self):
        """Helper function to count trainable parameters of the model

        Returns:
            int: number of trainable parameters of the model
        """

        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def init_weights(self):
        """Initializes network weights"""

        self.model.module_list = [
            m for m in self.model.modules()
            if not issubclass(type(m), nn.modules.container.ModuleDict)
        ]

        for i, m in enumerate(self.model.modules()):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                variance1 = math.sqrt(2.0 / n)
                m.weight.data.normal_(0, variance1)

            elif isinstance(m, nn.Linear):
                size = m.weight.size()
                fan_in = size[1]
                variance2 = math.sqrt(2.0 / fan_in)
                m.weight.data.normal_(0.0, 2*variance2)
                
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                variance3 = math.sqrt(2.0 / n)
                m.weight.data.normal_(0, variance3)

    def loss_function(self, **kwargs):
        """Loss function applied during learning is to be implemented by subclass"""

        raise NotImplementedError

    def update_(self, **kwargs):
        """Update some internal variables during training such as a scheduled or conditional learning rate"""

        pass

    def step(self, data_batch, target_batch, train=False):
        """Pass data through the model for training or evaluation

        Args:
            data_batch (torch.Tensor): batch of features from DataLoader object
            target_batch (torch.Tensor): batch of targets from DataLoader object
            train (bool, optional): flag if model weights should be updated based on loss (default: False)

        Returns:
            dict: contains performance metric information and possibly additional data from models forward function
        """

        if train:
            self.optimizer.zero_grad()
        out_batch = self.model(data_batch)
        out_batch["target"] = target_batch
        out_batch["weights"] = [p for p in self.model.parameters()]
        losses = self.loss_function(**out_batch)

        if train:
            losses["loss"].backward()
            self.optimizer.step()
            self.training_step += 1
            self.update_(step=self.training_step)

        result = dict(
            input=data_batch,
            target=target_batch,
        )
        result.update(out_batch)
        for k, v in losses.items():
            result.update({k: v.item()})

        return result

    def train(self, train_loader, epoch, metrics={}, max_batches=0):
        """Train the model for a single epoch

        Args:
            train_loader (DataLoader): object containing training data
            epoch (int): current training epoch
            metrics (dict, optional): contains names and functions of metrics
                to be calculated during training (default: {})
            max_batches (int, optional): limits number of batches per epoch, 0 = no limit (default: 0)

        Returns:
            dict: average loss and optional metrics on training dataset
        """

        self.model.train()

        summary = {"batch time": u.RunningAverage()}
        summary.update({m: u.RunningAverage() for m in metrics.keys()})
        summary.update({l: u.RunningAverage() for l in self.loss_function.loss_labels})

        n = min(max_batches, len(train_loader)) if max_batches else len(train_loader)

        with tqdm(total=n) as t:
            for batch_idx, (data_batch, target_batch) in enumerate(train_loader):

                t0 = time.time()

                data_batch = data_batch.to(self.device)
                target_batch = target_batch.to(self.device)
                if self.task == "reconstruction":
                    target_batch = data_batch
                result = self.step(data_batch, target_batch, train=True)

                # calculate metrics
                summary["batch time"].update(time.time() - t0)
                for metric, metric_fn in metrics.items():
                    summary[metric].update(metric_fn(**result))
                for l in self.loss_function.loss_labels:
                    summary[l].update(result[l])

                del result

                t.set_postfix(loss="{:05.4f}".format(summary["loss"]()))
                t.update()

                if batch_idx+1 == max_batches:
                    break

        # create the results dict
        train_results = {key: value() for (key, value) in summary.items()}

        """
        train_results = {}
        for metric in metrics.keys():
            train_results[metric] = summary[metric]()
        for l in self.loss_function.loss_labels:
            train_results[l] = summary[l]()
        """

        if self.verbose:
            self.log_func(f"====> Epoch {epoch}: Average loss = {summary['loss']():.4f}")

        return train_results

    def evaluate(self, test_loader, metrics={}, samplers={}, sample_freq=0, max_batches=0):
        """Evaluate the model on a validation or test set

        Args:
            test_loader (DataLoader): object containing test or validation data
            metrics (dict, optional): contains names and functions of metrics
                to be calculated during testing (default: {})
            samplers (dict, optional): contains names and functions of samplers
                to be applied during evaluation (default: {})
            sample_freq (int, optional): determines how often to sample the current data batch.
                Has no effect if samplers is empty.
                0 = no samples, 1 = sample every batch, 10 = sample every 10 batches (default: 0)
            max_batches (int, optional): limits how many batches should be evaluated.
                0 = no limit (default: 0)

        Returns:
            dict: average loss and optional metrics on test or validation data
            dict: contains samples drawn from batches with sampler functions
        """

        self.model.eval()

        summary = {"batch time": u.RunningAverage()}
        summary.update({m: u.RunningAverage() for m in metrics.keys()})
        summary.update({l: u.RunningAverage() for l in self.loss_function.loss_labels})

        samples = {}

        n = min(max_batches, len(test_loader)) if max_batches else len(test_loader)

        with torch.no_grad():
            with tqdm(total=n) as t:
                for batch_idx, (data_batch, target_batch) in enumerate(test_loader):

                    t0 = time.time()

                    data_batch = data_batch.to(self.device)
                    target_batch = target_batch.to(self.device)
                    label_batch = target_batch
                    if self.task == "reconstruction":
                        target_batch = data_batch
                    result = self.step(data_batch, target_batch, train=False)

                    summary["batch time"].update(time.time() - t0)

                    result["labels"] = label_batch
                    if self.input_layer is not None:
                        result["input weights"] = self.input_layer.weight.data.cpu()
                    if self.output_layer is not None:
                        result["output weights"] = self.output_layer.weight.data.cpu()

                    # calculate the metrics
                    for metric, metric_fn in metrics.items():
                        summary[metric].update(metric_fn(**result))
                    for l in self.loss_function.loss_labels:
                        summary[l].update(result[l])

                    if sample_freq and batch_idx % sample_freq == 0:
                        for sampler, sampler_fn in samplers.items():
                            samples.update({f"{sampler} batch {batch_idx}": sampler_fn(**result)})

                    del result

                    t.set_postfix(loss="{:05.4f}".format(summary["loss"]()))
                    t.update()

                    if max_batches and batch_idx == max_batches-1:
                        break

        # create the results dict
        test_results = {key: value() for (key, value) in summary.items()}

        """
        for metric in metrics.keys():
            test_results[metric] = summary[metric]()
        for l in self.loss_function.loss_labels:
            test_results[l] = summary[l]()
        """

        if self.verbose:
            name = self.model.__class__.__name__
            self.log_func(f"====> Test: {name} Average loss = {summary['loss']():.4f}")

        return test_results, samples


    def save_checkpoint(self, checkpoint, dir, name, is_best=False, verbose=True):
        """Save a checkpoint with model parameters for later use

        Args:
            checkpoint (dict): contains important variables from the model such as weights and performance info
            dir (str): directory to save the checkpoint in
            name (str): name to save the checkpoint as
            is_best (bool, optional): indicates if this currently the best available model of this run (default: False)
            verbose (bool, optional): indicates if function should print success statements (default: True)
        """

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
        """Load model parameters from a checkpoint

        Args:
            path (str): location of the checkpoint file to load
            model (torch.nn.Module object): model to load the weights from checkpoint
            optimizer (torch.optim object, optional): optimizer to load parameters from checkpoint (default: None)
            verbose (bool, optional): indicates if function should print success statements (default: True)

        Retruns:
            dict: contains important variables from the model such as weights and performance info
        """

        path = Path(path)

        if self.verbose:
            self.log_func(f"Loading model parameters from checkpoint {path}...")

        if not Path.exists(path):
            raise FileNotFoundError(f"File {path} doesn't exist")

        checkpoint = torch.load(path.as_posix())
        model.load_state_dict(checkpoint["state_dict"])

        if optimizer:
            optimizer.load_state_dict(checkpoint["optim_dict"])

        if verbose:
            self.log_func("Found checkpoint entries:")
            for k, v in checkpoint.items():
                self.log_func(f"{k:20} {type(v)}")
            self.log_func(f"Loaded model state from {path}\n")
            if optimizer:
                self.log_func(f"Loaded optimizer state from {path}\n")

        return checkpoint

    def train_and_evaluate(
            self,
            train_loader,
            val_loader,
            epochs,
            model_name=None,
            metrics=[],
            key_metric="validation loss",
            goal="minimize",
            load="",
            logger=None,
            checkpoints_dir="../results/checkpoints",
            eval_first=True,
            samplers=[],
            sample_freq=0,
            max_epoch_batches=0,
            plot_weights=[],
    ):
        """
        Runs training on the training set for the set number of epochs while continuously evaluating and logging.

        Args:
            train_loader (DataLoader): object containing the training data
            val_loader (DataLoader): object containing the validation data
            epochs (int): number of epochs to run training for
            model_name (str, optional): name to save the model under.
                If None model class name will be used (default: None)
            metrics (list, optional): names of metrics to compute during training and evaluation.
                All names given must correspond to a valid method defined in metrics.py (default: [])
            key_metric (str, optional): key of the metric to monitor to determine model improvements
                (default: validation loss)
            goal (str, optional): weather to maximize or minimize the key metric (default: minimize)
            load (str, optional): path to model checkpoint to load from disk, ignored if empty (default: "")
            logger (object, optional): if not None, logger.log() will be called to log dictionaries
                of metrics and samples at each epoch (default: None)
            checkpoints_dir (str, optional): directory to save and load model checkpoints (default: ../results/checkpoints)
            eval_first (bool, optional): if True, evaluation will occur before the first training episode (default: True)
            samplers (list, optional): names of samplers to use during evaluation.
                All names given must correspond to a valid method defined in samplers.py (default: [])
            sample_freq (int, optional): determines how often to sample the current data batch.
                Has no effect if samplers is empty.
                0 = no samples, 1 = sample every batch, 10 = sample every 10 batches (default: 0)
            max_epoch_batches (int, optional): limits number of batches per epoch, 0 = no limit (default: 0)

        """

        # Check if model name was specified
        if model_name is None:
            model_name = self.model.__class__.__name__

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
            del checkpoint
        else:
            start_epoch = 1
            training_summary = {}

        # initialize metrics
        metrics = met.get_metrics(metrics)

        # initialize samplers
        samplers = sam.get_samplers(samplers)

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
                logger.save_summary(summary, epoch=start_epoch - 1)
                logger.log(summary, step=start_epoch - 1)
                for batch, sample in samples.items():
                    for key, value in sample.items():
                        logger.log({f"{batch}_{key}": value}, step=start_epoch - 1)

            del val_results, samples, summary

        if self.verbose:
            self.log_func(f"Training {model_name} model for {epochs} epochs...")

        # Train for desired number of epochs
        for epoch in range(start_epoch, start_epoch + epochs):
            train_results = self.train(train_loader, epoch, metrics, max_epoch_batches)
            val_results, samples = self.evaluate(val_loader, metrics, samplers, sample_freq, max_epoch_batches)

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
                logger.save_summary(summary, epoch=epoch)
                logger.log(summary, step=epoch)
                for batch, sample in samples.items():
                    for key, value in sample.items():
                        logger.log({f"{batch}_{key}": value}, step=epoch)

            # print epoch summary
            if self.verbose:
                self.log_func(f"Summary epoch {epoch}:")
                for key, value in summary.items():
                    self.log_func(f"{key:50s} {value:10.4f}")

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

            del checkpoint, train_results, val_results, samples, summary







class SpikingBaseModel(BaseModel):
    """Custom spiking neural network base model"""

    def __init__(
            self,
            input_width,
            input_height,
            input_channels,
            #dataset,
            learning_rate,
            weight_decay,
            steps,
            threshold,
            decay,
            device,
            loss,
            grad_clip=0.0,
            use_extra_grad=True,
            verbose=False,
            log_func=print,
    ):

        """ Initializes a custom neural network base model

        Args:
            input_width (int): pixel width of input data
            input_height (int): pixel height of input data
            input_channels (int): number of channels in input data, grayscale = 1, RGB = 3
            learning_rate (float): step size for parameter updates during learning
            weight_decay (float): weight of the L2 regularization applied during learning
            steps (int): number of timesteps per example for spiking simulation
            threshold (int): firing threshold for LIF neurons
            decay (float): temporal variable controlling LIF membrane potential decay per step
            device (str): device where model parameters are stored and trained on
            loss (str): name of the loss function to use during training.
                Choice must be a valid option defined in losses.py
            verbose (bool, optional): flag to determine if process summary is given (default: True)
            log_func (function, optional): function to use for summary output (default: print)

        """

        super(SpikingBaseModel, self).__init__(
            input_width,
            input_height,
            input_channels,
            #dataset,
            learning_rate,
            weight_decay,
            device,
            loss,
            verbose=verbose,
            log_func=log_func,
        )

        self.steps = steps
        self.decay = decay
        self.threshold = threshold
        self.grad_clip = grad_clip
        self.use_extra_grad = use_extra_grad


    def grad_cal(self, decay, LF_output, Total_output):
        """Calculates gradients for spiking neurons based on this paper:
        https://www.frontiersin.org/articles/10.3389/fnins.2020.00119/full

        Args:
            decay (float): temporal variable describing the leaky property of used LIF neurons
            LF_output (torch.Tensor):
            Total_output (torch.Tensor):

        Returns:
            torch.Tensor: gradients


        TODO: Check if this works on CPU or if torch.cuda.FloatTensor needs to be replaced

        """

        Total_output = Total_output + (Total_output < 1e-3).type(torch.cuda.FloatTensor)
        out = LF_output.gt(1e-3).type(torch.cuda.FloatTensor) + math.log(decay) * torch.div(LF_output, Total_output)

        return out


    def step(self, data_batch, target_batch, train=False):
        """Pass data through the model for training or evaluation

        Args:
            data_batch (torch.Tensor): batch of features from DataLoader object
            target_batch (torch.Tensor): batch of targets from DataLoader object
            train (bool, optional): flag if model weights should be updated based on loss (default: False)

        Returns:
            dict: contains performance metric information and possibly additional data from models forward function
        """

        if train:
            self.optimizer.zero_grad()

        out_batch = self.model(data_batch, steps=self.steps)

        if train:

            if self.use_extra_grad:
                # compute gradient
                gradients = [self.grad_cal(self.decay, out_batch["lf_outs"][i], out_batch["total_outs"][i]) for i in range(len(out_batch["total_outs"])-1)]

                # apply gradient
                for t in range(self.steps):
                    for i in range(len(gradients)):
                        out_batch["out_temps"][i][t].register_hook(lambda grad, i=i: torch.mul(grad, gradients[i]))

        if type(self.loss_function) == nn.MSELoss and self.task is not "reconstruction":
            target = out_batch["output"].data.clone().zero_()
            target.scatter_(1, target_batch.unsqueeze(1), 1)
            target = target.type(torch.FloatTensor).to(self.device)
        else:
            target = target_batch

        out_batch["target"] = target
        out_batch["weights"] = [p for p in self.model.parameters()]

        losses = self.loss_function(**out_batch)
        result = dict(
            input=data_batch,
            target=target_batch,
        )

        result.update(out_batch)
        for k, v in losses.items():
            result.update({k: v.item()})

        if train:
            losses["loss"].backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_value_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            if self.use_extra_grad:
                for i in range(len(out_batch["out_temps"])-1):
                    out_batch["out_temps"][i] = None
                    gradients[i] = None

        return result
