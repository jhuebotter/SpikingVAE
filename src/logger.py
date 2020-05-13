from torch.utils.tensorboard import SummaryWriter
import numpy as np


class Logger(object):
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


if __name__ == "__main__":
    """Test for the implementations above."""

    log_dir = "../results/logs/exp1/g4"
    logger = Logger(log_dir)
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
