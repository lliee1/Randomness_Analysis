from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
import numpy as np
import random
import math
import torchvision
import torch.nn as nn
class TinyimagenetModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        aug: str,
        model_name: str,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        

        
        self.net = net
        
        if model_name != 'wide_resnet':
            num_classes = 200
            net.reset_classifier(num_classes=num_classes)
            

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=200)
        self.val_acc = Accuracy(task="multiclass", num_classes=200)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.aug = aug
        
    # scheduler
    def learning_rate(self,epoch):
        optim_factor = 0
        if(epoch > 160):
            optim_factor = 3
        elif(epoch > 120):
            optim_factor = 2
        elif(epoch > 60):
            optim_factor = 1
        elif(epoch > 240):
            optim_factor = 4

        return 0.1*math.pow(0.2, optim_factor)    
     
    def rand_bbox(self, size, lam):
        W = size[2] # Batch_size x Channel x Width x Height 
        H = size[3]

        cut_rat = np.sqrt(1. - lam)

        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()

    def train_model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        inputs, targets = batch
        r = np.random.rand(1)
        beta = 1.0
        if self.aug == 'randour_cutmix' and r < 0.5:
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(inputs.size()[0]).cuda()

            target_a = targets
            target_b = targets[rand_index]

            bbx1, bby1, bbx2, bby2 = self.rand_bbox(inputs.size(), lam)

            transform_cutmix = ['', 'cutmix_imgs[i] = torchvision.transforms.functional.autocontrast(cutmix_imgs[i])','cutmix_imgs[i] = torchvision.transforms.functional.invert(cutmix_imgs[i])',
                        'cutmix_imgs[i] = torchvision.transforms.functional.adjust_brightness(cutmix_imgs[i],2)','cutmix_imgs[i] = torchvision.transforms.functional.adjust_sharpness(cutmix_imgs[i],2)',
                        'cutmix_imgs[i] = torchvision.transforms.RandomRotation(180)(cutmix_imgs[i])',
                        'cutmix_imgs[i] = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)(cutmix_imgs[i])',
                        'cutmix_imgs[i] = torchvision.transforms.RandomAffine(0,(0.2,0))(cutmix_imgs[i])', 'cutmix_imgs[i] = torchvision.transforms.RandomAffine(0,(0,0.2))(cutmix_imgs[i])',
                        'cutmix_imgs[i] = torchvision.transforms.RandomAffine(0,shear=(-20,20,0,0))(cutmix_imgs[i])', 'cutmix_imgs[i] = torchvision.transforms.RandomAffine(0,shear=(-0,0,-20,20))(cutmix_imgs[i])']
            
            transform_original = ['', 'inputs[i] = torchvision.transforms.functional.autocontrast(inputs[i])','inputs[i] = torchvision.transforms.functional.invert(inputs[i])',
                        'inputs[i] = torchvision.transforms.functional.adjust_brightness(inputs[i],2)','inputs[i] = torchvision.transforms.functional.adjust_sharpness(inputs[i],2)',
                        'inputs[i] = torchvision.transforms.RandomRotation(180)(inputs[i])',
                        'inputs[i] = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)(inputs[i])',
                        'inputs[i] = torchvision.transforms.RandomAffine(0,(0.2,0))(inputs[i])', 'inputs[i] = torchvision.transforms.RandomAffine(0,(0,0.2))(inputs[i])',
                        'inputs[i] = torchvision.transforms.RandomAffine(0,shear=(-20,20,0,0))(inputs[i])', 'inputs[i] = torchvision.transforms.RandomAffine(0,shear=(-0,0,-20,20))(inputs[i])']

            cutmix_imgs = inputs[rand_index, :, :, :]
            for i in range(inputs.size()[0]):
                chocie_cutmix = random.randrange(0,len(transform_cutmix))
                choice_original = random.randrange(0,len(transform_original))

                exec(transform_cutmix[chocie_cutmix])
                exec(transform_original[choice_original])

            inputs[:, :, bbx1:bbx2, bby1:bby2] = cutmix_imgs[:, :, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
            outputs = self.forward(inputs)
            loss = self.criterion(outputs, target_a) * lam + self.criterion(outputs, target_b) * (1. - lam)

        elif self.aug == 'cutmix_randour' and r < 0.5:
            transform_original = ['', 'inputs[i] = torchvision.transforms.functional.autocontrast(inputs[i])','inputs[i] = torchvision.transforms.functional.invert(inputs[i])',
                        'inputs[i] = torchvision.transforms.functional.adjust_brightness(inputs[i],2)','inputs[i] = torchvision.transforms.functional.adjust_sharpness(inputs[i],2)',
                        'inputs[i] = torchvision.transforms.RandomRotation(180)(inputs[i])',
                        'inputs[i] = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)(inputs[i])',
                        'inputs[i] = torchvision.transforms.RandomAffine(0,(0.2,0))(inputs[i])', 'inputs[i] = torchvision.transforms.RandomAffine(0,(0,0.2))(inputs[i])',
                        'inputs[i] = torchvision.transforms.RandomAffine(0,shear=(-20,20,0,0))(inputs[i])', 'inputs[i] = torchvision.transforms.RandomAffine(0,shear=(-0,0,-20,20))(inputs[i])']

            # 1. cutmix
            lam = np.random.beta(beta, beta)

            rand_index = torch.randperm(inputs.size()[0]).cuda()
            target_a = targets
            target_b = targets[rand_index]

            bbx1, bby1, bbx2, bby2 = self.rand_bbox(inputs.size(), lam)
            inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
            
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
            
            # random aug at inputs
            for i in range(inputs.size()[0]):
                choice_transform = random.randrange(0, len(transform_original))
                exec(transform_original[choice_transform])
                
            outputs = self.forward(inputs)
            loss = self.criterion(outputs, target_a) * lam + self.criterion(outputs, target_b) * (1. - lam)
            
        elif self.aug == 'cutmix' and r < 0.5:
            lam = np.random.beta(beta, beta)

            rand_index = torch.randperm(inputs.size()[0]).cuda()
            target_a = targets
            target_b = targets[rand_index]

            bbx1, bby1, bbx2, bby2 = self.rand_bbox(inputs.size(), lam)
            inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
            
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
            
            outputs = self.forward(inputs)
            loss = self.criterion(outputs, target_a) * lam + self.criterion(outputs, target_b) * (1. - lam)
        
        elif self.aug == 'mixup' and r < 0.5:
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(inputs.size()[0]).cuda()
            target_a = targets
            target_b = targets[rand_index]
            
            inputs = lam * inputs + (1 - lam) * inputs[rand_index, :]
            outputs = self.forward(inputs)
            loss = self.criterion(outputs, target_a) * lam + self.criterion(outputs, target_b) * (1. - lam)

        elif self.aug == 'randaug_mixup' and r < 0.5:
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(inputs.size()[0]).cuda()
            target_a = targets
            target_b = targets[rand_index]
            
            inputs = lam * inputs + (1 - lam) * inputs[rand_index, :]
            outputs = self.forward(inputs)
            loss = self.criterion(outputs, target_a) * lam + self.criterion(outputs, target_b) * (1. - lam)
                           
               
        elif self.aug == 'randaug_cutmix' and r < 0.5:
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(inputs.size()[0]).cuda()

            target_a = targets
            target_b = targets[rand_index]

            bbx1, bby1, bbx2, bby2 = self.rand_bbox(inputs.size(), lam)

            cutmix_imgs = inputs[rand_index, :, :, :]
                
            inputs[:, :, bbx1:bbx2, bby1:bby2] = cutmix_imgs[:, :, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))

            # compute output
            outputs = self.forward(inputs)
            loss = self.criterion(outputs, target_a) * lam + self.criterion(outputs, target_b) * (1. - lam)
            
            
        elif self.aug == 'cutmix_randaug' and r < 0.5:
            in_aug = torchvision.transforms.Compose([
                                torchvision.transforms.RandAugment(num_ops=2, magnitude=9)
            ])
            norm = torchvision.transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(inputs.size()[0]).cuda()

            target_a = targets
            target_b = targets[rand_index]

            bbx1, bby1, bbx2, bby2 = self.rand_bbox(inputs.size(), lam)

            cutmix_imgs = inputs[rand_index, :, :, :]
                
            inputs[:, :, bbx1:bbx2, bby1:bby2] = cutmix_imgs[:, :, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))

            inputs = (inputs * 255).to(torch.uint8)
            inputs = norm((in_aug(inputs) / 255))
            # compute output
            outputs = self.forward(inputs)
            loss = self.criterion(outputs, target_a) * lam + self.criterion(outputs, target_b) * (1. - lam)
                        
        else:
            outputs = self.forward(inputs)
            loss = self.criterion(outputs, targets)
        preds = torch.argmax(outputs, dim=1)
        return loss, preds, targets

    def val_model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        inputs, targets = batch
        outputs = self.forward(inputs) 
        preds = torch.argmax(outputs, dim=1)
        loss = self.criterion(outputs, targets)
        return loss, preds, targets
    
    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.train_model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=True, on_epoch=False, prog_bar=True)

        # return loss or backpropagation will fail
        return loss


    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass


    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.val_model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)


    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        self.hparams.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self.learning_rate)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/acc",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = MNISTLitModule(None, None, None, None)
