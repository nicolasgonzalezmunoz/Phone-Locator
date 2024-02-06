"""Implement classes to be used on training or to train models."""
import copy
import torch
from torch.utils.data import DataLoader
from typing import Optional
from enum import Enum


class SelectBy(Enum):
    LOSS = 'loss'
    ACCURACY = 'accuracy'


class EarlyStopping:
    """
    Implement an early stopping algorithm based on validation loss.
    Parameters
    ----------
    patience: int, default=1
        Number of epochs with performance loss to wait before stopping
        training.
    min_delta: float, default=0.0
        Minimum variation with respect of previous validation loss to consider
        as performance loss.

    Attributes
    ----------
    min_validation_loss: float
        Minimum loss value seen on all epochs.
    counter: int
        Number of epochs where validation loss is over
        min_validation_loss + min_delta.
    """
    def __init__(self, patience: int = 1, min_delta: float = 0.0) -> None:
        if patience < 0:
            raise ValueError('Parameter "patience" must be a positive int.')
        if min_delta < 0.0:
            raise ValueError('Parameter "min_delta" must be non negative.')
        self.patience = patience
        self.min_delta = min_delta
        self._counter = 0
        self._min_validation_loss = float('inf')


    @property
    def counter(self) -> int:
        return self._counter

    @counter.setter
    def counter(self, new_counter: int) -> None:
        self._counter = new_counter


    @property
    def min_validation_loss(self):
        return self._min_validation_loss

    @min_validation_loss.setter
    def min_validation_loss(self, new_loss: float) -> None:
        self._min_validation_loss = new_loss


    def __call__(self, validation_loss: float) -> bool:
        """
        Enables the class to be used as a function.

        Parameters
        ----------
        validation_loss: float
            The validation loss on the current epoch.

        Return
        ------
        stop: bool
            Whether training should be stopped or not.
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                # Restart and stop
                self.reset()
                return True
        return False


    def reset(self) -> None:
        """Reset the instance's counter back to zero."""
        self.counter = 0
        self.min_validation_loss = float('inf')


class ModelTrainer:
    """
    Train a model with the given parameters.

    Parameters
    ----------
    train_dataloader: DataLoader
        A DataLoader instance containing the training samples.
    test_dataloader: DataLoader
        A DataLoader instance containing the test samples.
    loss_fn: nn.Module
        The loss_fn to use to minimize.
    optimizer_class: torch.optim.Optimizer
        The pytorch optimizer to use for the task. Default: Adam.
    early_stopping: EarlyStopping
        A EarlyStopping instance to use on training. If none is passed,
        the training will be made to its full completion.
    scheduler_class: torch.optim.lr_scheduler.LRScheduler, default=None
        A learning rate scheduler class.
    device: torch.device, default=None
        device to use for computations. If a torch.device is passed, the class
        will try to set all the operations on the given device. If device is
        not available or is None, the instance will set a device from the ones
        available.

    Attributes
    ----------
    optimizer_: torch.optim.Optimizer
        An instance of optimizer_class to optimize the model's parameters.
        Created and restarted when the instance is called as a function or
        when calling run_epochs.
    scheduler_: torch.optim.lr_scheduler.LRScheduler or None
        An instance of scheduler_class, or None if scheduler_class is None.
    device_: torch.device
        The actual device to use for computations.
    """
    def __init__(
        self,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        loss_fn: torch.nn.Module,
        optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
        early_stopping: Optional[EarlyStopping] = None,
        scheduler_class: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        device: Optional[torch.device] = None
    ) -> None:
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.loss_fn = loss_fn
        self.optimizer_class = optimizer_class
        self.early_stopping = early_stopping
        self.scheduler_class = scheduler_class
        self.device = device


    @property
    def device_(self) -> torch.device:
        return self._device_

    @device_.setter
    def device_(self, new_device: torch.device) -> None:
        self._device_ = new_device


    def __call__(
        self,
        epochs: int,
        model: torch.nn.Module,
        lr: float = 1e-4,
        silent: bool = False,
        select_by: SelectBy = 'accuracy'
    ) -> None:
        """
        Calls the instance as a function and runs training on several epochs.

        Parameters
        ----------
        epochs: int
            Number of training epochs to run.
        model: nn.Module
            Model to train.
        lr: float, default=1e-4
            Learning rate of the optimizer instance.
        silent: bool, default=False
            Level of verbose of the function.
        select_by: {'loss', 'accuracy}, default='accuracy'
            Define how the best model is chosen.

        Return
        ------
        None
        """
        self.run_epochs(epochs, model, lr, silent, select_by)


    def _train(self, model: torch.nn.Module, silent: bool = False) -> None:
        """
        Run one training epoch on model.

        Parameters
        ----------
        model: nn.Module
            Model to train.
        silent: bool, default=False
            Level of verbose of the function.

        Return
        ------
        None
        """
        size = len(self.train_dataloader.dataset)
        cum_loss = 0
        model.train()
        for batch, (X, y) in enumerate(self.train_dataloader):
            X, y = X.to(self.device_), y.to(self.device_)
            batch_size = len(X)

            # Compute prediction error
            pred = model(X)
            loss = self.loss_fn(pred, y)
            cum_loss += loss
            loss /= batch_size

            # Backpropagation
            self.optimizer_.zero_grad()
            loss.backward()
            self.optimizer_.step()

            if batch % 32 == 0 and not silent:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        cum_loss /= size
        if not silent:
            print(f" Cummulative loss: {cum_loss:>7f}")


    def _test(self, model: torch.nn.Module, silent: bool = False) -> None:
        """
        Test the model at the end of an epoch.

        Parameters
        ----------
        model: nn.Module
            Model to test.
        silent: bool, default=False
            Level of verbose of the function.

        Return
        ------
        None
        """
        size = len(self.test_dataloader.dataset)
        model.eval()
        test_loss, accuracy = 0, 0
        with torch.no_grad():
            for X, y in self.test_dataloader:
                X, y = X.to(self.device_), y.to(self.device_)
                pred = model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct = torch.sum((y - pred)**2, dim=-1) < 0.05
                accuracy += correct.type(torch.float).sum().item()
        accuracy /= size
        test_loss /= size
        if self.scheduler_ is not None:
            if isinstance(self.scheduler_, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler_.step(test_loss)
            else:
                self.scheduler_.step()
        if not silent:
            msg = (
                f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, "
                f"Avg loss: {test_loss:>8f} \n"
            )
            print(msg)
        if self.early_stopping is not None:
            early_stop = self.early_stopping(test_loss)
            return early_stop, test_loss, accuracy
        return False, test_loss, accuracy


    def run_epochs(
        self,
        epochs: int,
        model: torch.nn.Module,
        lr: float = 1e-4,
        silent: bool = False,
        select_by: SelectBy = 'accuracy'
    ) -> None:
        """
        Run one or more training epochs on model, and test it at the end of each epoch.

        Parameters
        ----------
        epochs: int
            Number of training epochs to run.
        model: nn.Module
            Model to train.
        lr: float, default=1e-4
            Learning rate of the optimizer instance.
        silent: bool, default=False
            Level of verbose of the function.
        select_by: {'loss', 'accuracy}, default='accuracy'
            Define how the best model is chosen.

        Return
        ------
        None
        """
        if self.device is None:
            self.set_default_device()
        else:
            mps_is_available = (torch.backends.mps.is_available())
            cuda_is_available = torch.cuda.is_available()
            set_cuda = (self.device.startswith('cuda') and cuda_is_available)
            set_mps = (self.device.startswith('mps') and mps_is_available)
            set_default = (not set_cuda) or (not set_mps)
            if set_default:
                self.set_default_device()
            else:
                self.device_ = self.device
        model = model.to(self.device_)
        self.optimizer_ = self.optimizer_class(
            [param  for param in model.parameters() if param.requires_grad],
            lr=lr,
            maximize=False
        )
        if self.scheduler_class is not None:
            self.scheduler_ = self.scheduler_class(self.optimizer_)
        best_loss = float('inf')
        best_accuracy = 0.0
        best_model_weights = copy.deepcopy(model.state_dict())
        if self.early_stopping is not None:
            self.early_stopping.reset()
        for t in range(epochs):
            if not silent:
                print(f"Epoch {t+1}\n-------------------------------")
            self._train(model, silent)
            early_stop, test_loss, accuracy = self._test(model, silent)
            if select_by == 'loss':
                is_best = (test_loss < best_loss)
            else:
                is_best = (accuracy > best_accuracy)
            if is_best:
                best_accuracy = accuracy
                best_loss = test_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                torch.save(best_model_weights, 'best_model_wts.pth')
            if early_stop:
                break
        model.load_state_dict(best_model_weights)
        if not silent:
            print("Done!")


    def set_default_device(self) -> None:
        """Set device_ to default value, depending on available devices."""
        self.device_ = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
