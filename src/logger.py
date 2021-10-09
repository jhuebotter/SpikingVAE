#from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import wandb
from pathlib import Path


class TensorboardLogger(object):
    """Logger class to graphical summary in tensorboard"""

    def __init__(self, log_dir):
        # Writer will output to ./runs/ directory by default
        self.writer = SummaryWriter(log_dir)

    def write_image_batch(self, title, img_batch, step=0):
        self.writer.add_images(title, img_batch, step)

    def add_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def add_scalars(self, tag, value_dict, step):
        self.writer.add_scalars(tag, value_dict, step)

    def close(self):
        self.writer.close()


class WandBLogger(object):
    """Logger class for graphical summary with Weights and Biases"""

    def __init__(self, args, name=None, dir="../results", id=None, upload_models=False):
        #if id is None:
        #id = wandb.util.generate_id()
        if args.experiment is not "":
            dir = Path(dir, args.experiment)
            dir.mkdir(exist_ok=True, parents=True)

        self.results_dir = dir
        self.run = wandb.init(
            name=name,
            config=args,
            resume="allow",
            project="spiking-vae",
            dir=dir,
        )

        self.result = pd.DataFrame([])

        # save models in run dir
        if upload_models:
            wandb.save("*.pth")


    def watch(self, model):
        # tell logger to watch the model
        wandb.watch(model, log="all")

    def log(self, content={}, step=None):
        if step is None:
            self.run.log(content)
        else:
            assert isinstance(step, int)
            self.run.log(content, step=step)

    def save_summary(self, summary, epoch=0):
        
        result = dict(epoch=epoch, name=self.run.name, run_id=self.run.id)
        for k, v in self.run.config.items():
            if type(v) == list:
                s = ''
                for i in v:
                    s += f'{i}, '
                result[k] = s
            elif type(v) in (str, int, float, bool):
                result[k] = v
        for k, v in summary.items():
            result[k] = v

        df = pd.DataFrame([result])
        self.result = pd.concat([self.result, df])
        self.save_result()


    def save_result(self, path='all_results.csv'):

        if self.result.empty:
            print("nothing to save")
        else:
            path = Path(self.results_dir, path)
            print("saving results at", path)
            if path.exists():
                print("loading existing results")
                df = pd.read_csv(path)
            else:
                print("Making new results frame")
                df = pd.DataFrame([])
            df = pd.concat([df, self.result], ignore_index=True)
            df.to_csv(path, index=False)
            self.result = pd.DataFrame([])


if __name__ == "__main__":
    """Test for the implementations above."""

    log_dir = "../results/logs/exp1/g4"
    logger = TensorboardLogger(log_dir)
    writer = logger.writer

    import time

    for run in range(5):
        for n_iter in range(100):
            time.sleep(1)
            writer.add_scalar(f"{run}/f1/Loss/train", np.random.random(), n_iter)
            writer.add_scalar(f"{run}/f1/Loss/test", np.random.random(), n_iter)
            writer.add_scalar(f"{run}/f1/Accuracy/train", np.random.random(), n_iter)
            writer.add_scalar(f"{run}/f1/Accuracy/test", np.random.random(), n_iter)
            writer.close()

    img = np.zeros((3, 100, 100))
    img[0] = np.arange(0, 10000).reshape(100, 100) / 10000
    img[1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000

    img_HWC = np.zeros((100, 100, 3))
    img_HWC[:, :, 0] = np.arange(0, 10000).reshape(100, 100) / 10000
    img_HWC[:, :, 1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000

    writer.add_image("my_image", img, 0)

    # If you have non-default dimension setting, set the dataformats argument.
    writer.add_image("my_image_HWC", img_HWC, 0, dataformats="HWC")
    writer.close()
