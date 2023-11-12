from typing import Dict, Tuple, Optional, Union
import torch.nn as nn
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy.policy import Policy
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.ppo.ppo import PPO
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.logger import pretty_print
from ray.rllib.evaluation.episode import Episode
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import AgentID, PolicyID
from ray.rllib.evaluation import RolloutWorker


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


    def on_episode_step(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        episode: Union[Episode, EpisodeV2],
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:

        pass
        #print("///// Entered CdaCallbacks.on_episode_step.")


    def on_postprocess_trajectory(self, *,
                                  worker: "RolloutWorker",          #ray.rllib.evaluation
                                  episode: Episode,                 #ray.rllib.evaluation.episode
                                  agent_id: AgentID,                #ray.rllib.utils.typing
                                  policy_id: PolicyID,              #ray.rllib.utils.typing
                                  policies: Dict[PolicyID, Policy], #ray.rllib.policy
                                  postprocessed_batch: SampleBatch, #ray.rllib.policy.sample_batch
                                  original_batches: Dict[AgentID, Tuple[Policy, SampleBatch]],
                                  **kwargs,
                                 ) -> None:

        """Called immediately after a policy's postprocess_fn is called.
            Each element of the postprocessed_bacth is a numpy array.
            NOTE:  apparently SampleBatches don't contain "seq_lens" entries.
        """

        pass

        """
        print("///// Entered CdaCallbacks.on_postprocess_trajectory: agent_id = ", agent_id, ", policy_id = ", policy_id)
        print("      processed_batch obs len = {}, actions len = {}, rewards len = {}, infos len = {}, eps_id = {}"
              .format(len(postprocessed_batch["obs"]), len(postprocessed_batch["actions"]), len(postprocessed_batch["rewards"]), \
                        len(postprocessed_batch["infos"]), postprocessed_batch["eps_id"]))
        print("      action[1] = ", type(postprocessed_batch["actions"][1]), postprocessed_batch["actions"][1])
        print("      new_obs[1] = ", type(postprocessed_batch["new_obs"][1]), postprocessed_batch["new_obs"][1, 0:20])
        """
