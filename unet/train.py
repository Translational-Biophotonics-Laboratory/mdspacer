import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import albumentations as A
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
from torch.utils.data.dataloader import DataLoader
from torchinfo import summary
from albumentations.pytorch import ToTensorV2
from os.path import abspath, join, dirname, normpath
from tqdm import tqdm
from unet.model import UNET
from unet.dataset import COCODataset
from tools.utils import create_directory

# Hyperparameters
LEARNING_RATE = 5e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if torch.cuda.is_available() else False
BATCH_SIZE = 100
NUM_CLASSES = 80 + 1 # Background class
NUM_EPOCHS = 80
NUM_WORKERS = 4
IMAGE_SIZE = 256
CHECKPOINT_DIR = "/home/dkermany/BoneSegmentation/checkpoints/{}".format(
    datetime.today().strftime("%m%d%Y")
)

class UNetMetrics:
    """
    Class used to track, compute, and manage metrics throughout UNet training
    and validation

    Arguments:
        - num_classes (int): number of classes
    """
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self._train = True

        # parameters to be passed to torchmetric classes
        params = {
            "num_classes": self.num_classes,
            "ignore_index": 0,
            "mdmc_average": "global",
            "average": "micro",
        }

        # holder for torchmetric instances
        self.metrics: dict[str, dict[str, object]] = {
            stage: {
                "acc": torchmetrics.Accuracy(**params).to(device=DEVICE),
                "dice": torchmetrics.Dice(**params).to(device=DEVICE),
              # "iou": torchmetrics.JaccardIndex(**params).to(device=DEVICE),
            } for stage in ["train", "val"]
        }

        self.plots: dict[str, dict[str, list]] = {
            stage: {
                "step": [],
                "loss": [],
                "acc": [],
                "dice": [],
            } for stage in ["train", "val"]
        }

    def train(self):
        """
        Puts instance into "train" mode and updates the training metrics and plot
        """
        self._train = True

    def eval(self):
        """
        Puts instance into "eval" mode and updates the validation metrics and plot
        """
        self._train = False

    def update(self, step: int, loss: float, preds: Tensor, targets: Tensor):
        """
        Update metrics with new batch of data

        Arguments:
            - step (int): batch index corresponding with the provided data
            - loss (float): mean batch cross-entropy loss
            - preds (torch.Tensor): predicted labels with shape (N, H, W)
            - targets (torch.Tensor): targets with shape (N, H, W)
        """
        stage = "train" if self._train else "val"

        acc = self.metrics[stage]["acc"](preds, targets)
        dice = self.metrics[stage]["dice"](preds, targets)

        self.plots[stage]["step"].append(step)
        self.plots[stage]["loss"].append(loss)
        self.plots[stage]["acc"].append(acc)
        self.plots[stage]["dice"].append(dice)

    def compute(self) -> dict[str, dict[str, float]]:
        """
        Returns dictionary containing final average metrics
        """
        result = {}
        for stage, m in self.metrics.items():
            for metric_name, metric in m.items():
                result[stage] = {
                    metric_name: metric.compute()
                }
        return result

    def _to_dataframe(self) -> pd.DataFrame:
        """
        Rearrange metric data into "long-form" pandas dataframe
        https://seaborn.pydata.org/tutorial/data_structure.html
        """
        df_list = []
        for stage_name, stage in self.plots.items():
            for metric_name, values in stage.items():
                # Exclude batch number from metrics
                if metric_name != "batch":
                    stage_col = [alt_names[stage_name]] * len(values)
                    metric_col = [alt_names[metric_name]] * len(values)

                    # Concatenate rows into df_list
                    df_list += list(zip(
                        self.plots[stage_name]["batch"],
                        stage_col,
                        metric_col,
                        values
                    ))

        # Generate dataframe
        df = pd.DataFrame(df_list, columns=[
            "Batch",
            "Stage",
            "Metrics",
            "Value",
        ])
        return df

    def plot_graphs(self, output_dir: str):
        """
        Plots training and validation curves using seaborn and saves the images
        to the "output" path

        Arguments:
            - output_dir (str): path to directory in which plot will be saved
        """
        # Key to better formatted strings for plot
        alt_names = {
            "loss": "Loss",
            "acc": "Accuracy",
            "dice": "Dice Score",
            "train": "Training",
            "val": "Validation",
        }

        # Format data into "long-form" dataframe for seaborn
        df = self.to_dataframe()

        # Set seaborn image size and font scaling
        sns.set(
            rc={"figure.figsize": (16, 9)},
            font_scale=1.5,
        )

        # Generate seaborn lineplot
        # Colors different based on metric
        # Style (solid v. dashed) based on training or validation
        # Dashes specified with list of tuples: (segment, gap)
        lplot = sns.lineplot(
            x="Batch",
            y="Value",
            hue="Metrics",
            style="Stage",
            data=df,
            dashes=[(1,0),(6,3)],
        )

        # Remove y-axis label
        lplot.set(ylabel=None)

        # Set plot title
        plt.title("Training Loss and Accuracy Curves")

        # Move legend to outside upper-right corner of image
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

        # Save image to disk
        lplot.figure.savefig(join(output_dir, "training_curves.png"))

    def write_to_file(self, filename: str):
        """
        Writes data to csv in "long form"

        Arguments:
            - filename (str): path to .csv file to be saved
        """
        df = self._to_dataframe()
        df.to_csv(filename, mode="w", index=False)


class UNetTrainer:
    """
    Performs training and transfer learning using UNet

    Default behavior is to train learnable parameters from scratch,
    however passing a path to a *.pth.tar checkpoint in the checkpoint_path
    parameter will:
        1. Freeze all layers


    Usage
        > trainer = UNetTrainer(**args)
        > trainer.run()

    Arguments:
        - checkpoint_path (str): path to .pth.tar file containing UNet
                                 parameters to reinitialize
        - freeze (bool): If False (default), all learnable parameters are
                         left unfrozen. If True, all weights except for
                         the final convolutional layer will be frozen for
                         retraining of final layer, followed by unfreezing
                         the layers and finetuning at a lower learning
                         rate.
    """
    def __init__(
        self,
        image_size: int,
        num_classes: int,
        batch_size: int = 1,
        num_epochs: int = 1,
        learning_rate: float = 3e-4,
        checkpoint_dir: str = "",
        freeze: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        self.image_size = image_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.checkpoint_dir = checkpoint_dir
        self.freeze = freeze
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Initialize UNet model
        self.model = UNET(in_channels=3, out_channels=self.num_classes)

        # Initialize loss function
        # TODO: Add class_weights after implementing
        self.loss_fn = nn.CrossEntropyLoss()

        # Initialize optimizer
        # passing only those paramters that explicitly require grad
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
        )

        if self.checkpoint_dir != "":
            self.load_checkpoint()

        # Casts model to selected device (CPU vs GPU)
        self.model.to(device=DEVICE)

        # Parallelizes model across multiple GPUs
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs...")
            self.model = nn.DataParallel(self.model)

        self.scaler = torch.cuda.amp.GradScaler()
        create_directory(CHECKPOINT_DIR)

        # Create UNetMetrics instance
        self.metrics = UNetMetrics(self.num_classes)

    def load_checkpoint(self) -> None:
        """
        Sets checkpoint path and specifies if loaded weights will be frozen for
        retraining or left unfrozen for evaluation or continuation of training
        """
        if not self.checkpoint_path.endswith(".pth.tar"):
            err = f"""Checkpoint {self.checkpoint_path} not valid checkpoint file
                      type (expected: .pth.tar)"""
            raise ValueError(err)

        print(f"==> Loading checkpoint: {self.checkpoint_path}")

        # loads checkpoint into model and optimizer using state dicts
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        # Unfrozen, to continue or evaluate
        if not freeze:
            for param in self.model.parameters():
                param.requires_grad = True
        # Freeze all layers, except final layer
        else:
            for name, param in self.model.named_parameters():
                if name.startswith("final_conv"):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def save_checkpoint(self, state: dict[str, object], filename: str) -> None:
        print(f"=> Saving checkpoint: {filename}")
        torch.save(state, filename)

    def get_transforms(self) -> dict[str, A.Compose]:
        train_transform = A.Compose(
            [
                A.Resize(
                    height=int(self.image_size*1.13),
                    width=int(self.image_size*1.13),
                ),
                A.RandomCrop(height=self.image_size, width=self.image_size),
                A.Rotate(limit=90, p=0.9),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.CLAHE(clip_limit=4.0, p=0.35),
                A.ColorJitter(p=0.15),
                A.GaussNoise(p=0.15),
                A.Normalize(
                    mean=[0.4690, 0.4456, 0.4062],
                    std=[0.2752, 0.2701, 0.2847],
                    max_pixel_value=255.0
                ),
                ToTensorV2()
            ]
        )
        val_transform = A.Compose(
            [
                A.Resize(height=self.image_size, width=self.image_size),
                A.Normalize(
                    mean=[0.4690, 0.4456, 0.4062],
                    std=[0.2752, 0.2701, 0.2847],
                    max_pixel_value=255.0
                ),
                ToTensorV2()
            ]
        )
        return {"train": train_transform, "val": val_transform}

    def get_coco_loaders(
        self,
        transforms: dict[str, A.Compose]
    ) -> tuple[DataLoader[object]]:

        train_ds = COCODataset(
            join(FLAGS.images, "train2017"),
            join(FLAGS.masks, "train2017"),
            image_ext="jpg",
            mask_ext="png",
            transform=transforms["train"]
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        val_ds = COCODataset(
            join(FLAGS.images, "val2017"),
            join(FLAGS.masks, "val2017"),
            image_ext="jpg",
            mask_ext="png",
            transform=transforms["val"]
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return train_loader, val_loader

    def one_hot_encoding(self, label: torch.Tensor) -> torch.Tensor:
        """
        One-Hot Encoding for segmentation masks
        Example: Converts (batch, 256, 256) => (batch, num_classes, 256, 256)
        with convention of (B, C, H, W)
        """
        # nn.function.one_hot returns in CHANNEL_LAST formatting
        # permute needed to convert to CHANNEL_FIRST
        one_hot = nn.functional.one_hot(
            label.long(),
            num_classes=self.num_classes
        )
        return one_hot.permute(0,3,1,2)

    def train_fn(
        self,
        loader: DataLoader[object],
        epoch: int,
    ):
        loop = tqdm(loader)
        for i, (data, targets) in enumerate(loop):
            data = data.float().to(device=DEVICE)
            targets = self.one_hot_encoding(targets).float().to(device=DEVICE)

            # forward
            with torch.cuda.amp.autocast():
                logits = self.model(data)
                loss = self.loss_fn(logits, targets)

            # backward
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            targets = torch.argmax(targets, axis=1).int()
            predictions = torch.argmax(logits, dim=1)

            # Pixel-wise accuracy
            accuracy = (predictions == targets).sum() / torch.numel(targets)

            # Add metrics
            for _, metric in self.metrics["train"].items():
                result = metric(predictions, targets)

            # update tqdm loop
            loop.set_description("Epoch: {}/{} - Loss: {:.2f}, Acc: {:.2%}".format(
                epoch+1,
                self.num_epochs,
                loss.item(),
                accuracy.item()
            ))

    def validate_one_batch(self, loader: DataLoader[object]) -> dict:
        """
        Performs inference only on the first batch of the validation set and
        calculates loss and the mean jaccard index. Meant to be used after
        each epoch to track the model's performance over the epochs
        """
        # Set model into evaluation mode
        self.model.eval()
        with torch.no_grad():

            # Get only the same first batch of images
            # Make sure loader shuffle is set to False to ensure geting the
            # same batch each time
            data, targets = next(iter(loader))

            data = data.float().to(device=DEVICE)
            targets = self.one_hot_encoding(targets).float().to(device=DEVICE)
            with torch.cuda.amp.autocast():
                logits = torch.softmax(self.model(data), dim=1)
                loss = self.loss_fn(logits, targets)

            # Change tensor shape 
            # (batch, n_classes, size, size) -> (batch, size, size)
            targets = torch.argmax(targets, axis=1).int()
            predictions = torch.argmax(logits, dim=1)

            # Pixel-wise accuracy
            accuracy = (predictions == targets).sum() / torch.numel(targets)

            # Add metrics
            print(f"""Metric Calculations - input sizes - pred:
                  {predictions.shape}, targets: {targets.shape}""")
            for name, metric in self.metrics["val"].items():
                result = metric(predictions, targets)
                print("val", name, result)

        # Return model to train mode
        self.model.train()

        return {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
        }

    def validate_fn(self, loader: DataLoader[object]) -> dict:
        """
        Performs inference only on the entire validation set and
        calculates loss and the mean jaccard index. Meant to be used after
        model training is complete
        """
        # Set model into evaluation mode
        self.model.eval()
        with torch.no_grad():
            running_loss = 0.
            running_accuracy = 0.
            loop = tqdm(loader)
            for data, targets in loop:
                data = data.float().to(device=DEVICE)
                targets = self.one_hot_encoding(targets).float().to(device=DEVICE)
                with torch.cuda.amp.autocast():
                    logits = torch.softmax(self.model(data), dim=1)
                    loss = self.loss_fn(logits, targets)

                # Change tensor shape 
                # (batch, n_classes, size, size) -> (batch, size, size)
                targets = torch.argmax(targets, axis=1).int()
                predictions = torch.argmax(logits, dim=1)

                # Add metrics
                print(f"""Metric Calculations - input sizes - pred:
                      {predictions.shape}, targets: {targets.shape}""")
                for name, metric in self.metrics["val"].items():
                    result = metric(predictions, targets).item()
                    print("val", name, result)

                running_loss += loss.item()
                running_accuracy += accuracy.item()

        mean_loss = running_loss / len(loader)
        mean_accuracy = running_accuracy / len(loader)

        # Return model to train mode
        self.model.train()

        return {
            "loss": mean_loss,
            "accuracy": mean_accuracy,
        }

    def run(self):
        # Load transforms
        transforms = self.get_transforms()

        # Load Datasets
        train_loader, val_loader = self.get_coco_loaders(transforms)


        best_loss = 9999.
        for epoch in range(self.num_epochs):
            #self.train_fn(train_loader, epoch)

            # Check accuracy
            #results = self.validate_one_batch(val_loader)
            results = self.validate_fn(val_loader)
            return

            print("Epoch {} Validation - Loss: {:.2f}, Acc: {:.2%}".format(
                epoch+1,
                results["loss"],
                results["accuracy"],
            ))

            # If this is the best performance so far, save checkpoint
            if results["loss"] < best_loss:
                best_loss = results["loss"]

                # Save model
                checkpoint = {
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                }
                if not FLAGS.debug:
                    save_checkpoint(
                        checkpoint,
                        filename=join(CHECKPOINT_DIR,
                                      f"unet1.0-coco-epoch{epoch}.pth.tar")
                    )

        # Evaluate model on entire validation set
        results = self.validate_fn(val_loader)
        print("Final Validation - Loss: {}, mIoU: {}, Acc: {}".format(
                epoch,
                results["loss"],
                results["jaccard"],
                results["accuracy"],
        ))

        # Save metrics to files
        print("Saving metrics to files...")
        pd.DataFrame(
            self.metrics["loss"], 
            columns=["Batch Number", "Loss", "Mode"],
        ).to_csv(join(CHECKPOINT_DIR, "loss.csv"))
        pd.DataFrame(
            self.metrics["acc"], 
            columns=["Batch Number", "Accuracy", "Mode"],
        ).to_csv(join(CHECKPOINT_DIR, "accuracy.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images",
        required=True,
        type=str,
        help="Path to train/val/test images directory"
    )
    parser.add_argument(
        "--masks",
        required=True,
        type=str,
        help="Path to train/val/test masks directory"
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        type=str,
        help="Path to checkpoint (.pth.tar) from which to load"
    )
    parser.add_argument(
        "--freeze",
        default=False,
        type=bool,
        help="Whether to perform training or inference on validation"
    )
    parser.add_argument(
        "--debug",
        default=False,
        type=bool,
        help="When true, does not save checkpoints"
    )

    FLAGS, _ = parser.parse_known_args()

    if FLAGS.freeze and FLAGS.checkpoint == "":
        raise ValueError("Must specify a checkpoint file (.pth.tar) for freeze")

    trainer = UNetTrainer(
        image_size=IMAGE_SIZE,
        num_classes=NUM_CLASSES,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        freeze=FLAGS.freeze,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )


    trainer.run()

