from typing import Union, Optional, Any, List
import numpy as np
import eagerpy as ep
import tensorflow as tf

from ..devutils import atleast_kd

from ..models import Model

from ..distances import Distance

from ..criteria import Criterion

from .base import FlexibleDistanceMinimizationAttack
from .base import T
from .base import get_criterion
from .base import raise_if_kwargs


class DatasetAttack(FlexibleDistanceMinimizationAttack):
    """Draws randomly from the given dataset until adversarial examples for all
    inputs have been found.

    To pass data form the dataset to this attack, call :meth:`feed()`.
    :meth:`feed()` can be called several times and should only be called with
    batches that are small enough that they can be passed through the model.

    Args:
        distance : Distance measure for which minimal adversarial examples are searched.
    """

    def __init__(self, *, distance: Optional[Distance] = None):
        super().__init__(distance=distance)
        self.raw_inputs: List[ep.Tensor] = []
        self.raw_outputs: List[ep.Tensor] = []
        self.inputs: Optional[ep.Tensor] = None
        self.outputs: Optional[ep.Tensor] = None

    def feed(self, model: Model, inputs: Any) -> None:
        x = ep.astensor(inputs)
        del inputs

        self.raw_inputs.append(x)
        self.raw_outputs.append(model(x))

    def process_raw(self) -> None:
        raw_inputs = self.raw_inputs
        raw_outputs = self.raw_outputs
        assert len(raw_inputs) == len(raw_outputs)
        assert (self.inputs is None) == (self.outputs is None)

        if self.inputs is None:
            if len(raw_inputs) == 0:
                raise ValueError(
                    "DatasetAttack can only be called after data has been provided using 'feed()'"
                )
        elif self.inputs is not None:
            assert self.outputs is not None
            raw_inputs = [self.inputs] + raw_inputs
            raw_outputs = [self.outputs] + raw_outputs

        self.inputs = ep.concatenate(raw_inputs, axis=0)
        self.outputs = ep.concatenate(raw_outputs, axis=0)
        self.raw_inputs = []
        self.raw_outputs = []

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Criterion, T],
        *,
        early_stop: Optional[float] = None,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        self.process_raw()
        assert self.inputs is not None
        assert self.outputs is not None
        x, restore_type = ep.astensor_(inputs)
        del inputs, kwargs

        criterion = get_criterion(criterion)

        result = x
        found = criterion(x, model(x))

        dataset_size = len(self.inputs)
        batch_size = len(x)

        while not found.all():
            indices = np.random.randint(0, dataset_size, size=(batch_size,))

            xp = self.inputs[indices]
            yp = self.outputs[indices]
            is_adv = criterion(xp, yp)

            new_found = ep.logical_and(is_adv, found.logical_not())
            result = ep.where(atleast_kd(new_found, result.ndim), xp, result)
            found = ep.logical_or(found, new_found)

        return restore_type(result)


class DatasetMinimizationAttack(DatasetAttack):
    
    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Criterion, T],
        *,
        early_stop: Optional[float] = None,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        self.process_raw()
        assert self.inputs is not None
        assert self.outputs is not None
        x, restore_type = ep.astensor_(inputs)
        del inputs, kwargs

        criterion = get_criterion(criterion)

        dataset_size = len(self.inputs)
        batch_size = len(x)
        
        all_same_indices = [[i for _ in range(batch_size)] for i in range(dataset_size)]
        # this array gives at position [dataset_index][input_index] whether the given dataset value is adversarial
        possible_indices_array = [criterion(self.inputs[indices], self.outputs[indices]) for indices in all_same_indices]

        possible_indices = ep.stack(possible_indices_array)
    
        inputs_numpy = self.inputs.numpy()
        x_numpy = x.numpy()

        # distances = [
        #     [np.linalg.norm(x_numpy[j] - inputs_numpy[i], ord=1) for j in range(batch_size)] for i in range(dataset_size)    
        # ] 
        # print(distances)

        outputs = []
        for batch_index in range(batch_size):
            min_index = 0
            min_distance = 2000000
            for dataset_index in range(dataset_size):
                if possible_indices[dataset_index][batch_index]:
                    distance = np.linalg.norm(x_numpy[batch_index] - inputs_numpy[dataset_index], ord=self.distance.p)
                    if distance < min_distance:
                        min_distance = distance
                        min_index = dataset_index
            
            print(min_index)
            outputs.append(inputs_numpy[min_index])

        output_array = np.stack(outputs, axis=0)
        output_tensor = tf.constant(output_array)
        output_ep = ep.astensor(output_tensor)

        return restore_type(output_ep)




