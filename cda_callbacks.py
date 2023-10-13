from typing import Dict
import torch.nn as nn
from ray.rllib.policy.policy import Policy
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.ppo.ppo import PPO
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.logger import pretty_print


class CdaCallbacks (DefaultCallbacks):
    """This class provides utility callbacks that RLlib training algorithms invoke at various points.
        This class currently only handles a one-policy
        situation (could be used by multiple agents), with the policy named "default_policy".
    """

    def __init__(self,
                 legacy_callbacks_dict: Dict[str, callable]      = None,    #required by RLlib
                 use_perturbation_controller :              bool = False,   #deprecated - no longer supported
                ):

        super().__init__(legacy_callbacks_dict)
        if use_perturbation_controller:
            raise NotImplementedError("CdaCallbacks created with use_perturbation_controller on, but it is not supported.")

        self._use_perturbation_controller = False
        self._checkpoint_path = None
        """
        if use_perturbation_controller:
            self.info = PerturbationController()
            self._checkpoint_path = self.info.get_checkpoint_path()
        """


    def on_algorithm_init(self, *,
                          algorithm,
                          **kwargs,
                         ) -> None:

        """Called when a new algorithm instance has finished its setup() but before training begins.
            We will use it to initialize NN weights with Xavier normal distro.
        """

        # Update the initialize counter
        if self._use_perturbation_controller:
            self.info.increment_algo_init()

        # Get the initial weights from the newly created NN
        policy_dict = algorithm.get_weights(["default_policy"])["default_policy"]

        # Re-initialize the weights using torch's Xavier normal function. The dict contains many Tensors, some with 1 dimension, some with 2.
        # The 2D Tensors are weights, which are the only ones we want to modify. Some of the 1D are biases, which should remain 0, and some
        # others serve other purposes.
        for i, key in enumerate(policy_dict):
            w = policy_dict[key]
            if w.shape[0] == 2:
                nn.init.xavier_normal_(w, 1.41) #modify in place; torch docs recommends gain of sqrt(2) for relu activation

        # Stuff the modified weights into the newly created NN
        to_algo = {"default_policy": policy_dict} #should be of type Dict[PolicyId, ModelWeights]; PolicyID = str, ModelWeights = dict
        algorithm.set_weights(to_algo)
        print("///// CdaCallbacks: re-initialized model weights using Xavier normal.")
