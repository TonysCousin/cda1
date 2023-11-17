import sys
import copy
from statistics import mean
from typing import Tuple, Dict, List
import math
from datetime import datetime
import yaml
from yaml import load
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
from gymnasium.spaces import Box
import numpy as np
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv
from ray.tune.logger import pretty_print

from constants import Constants
from obs_vec import ObsVec
from roadway_b import Roadway
from vehicle import Vehicle
from hp_prng import HpPrng
from lane_change import LaneChange
from target_destination import TargetDestination
# Need to import every derived class that a user might choose to use, so that the config will be recognized:
from bot_type1_model import BotType1Model
from bot_type1a_guidance import BotType1aGuidance
from bot_type1b_guidance import BotType1bGuidance
from bridgit_model import BridgitModel
from bridgit_guidance import BridgitGuidance


class HighwayEnv(TaskSettableEnv):  #based on OpenAI gymnasium API; TaskSettableEnv can be used for curriculum learning

    """This environment is a segment of highway with a possible variety of on-ramps, off-ramps, lane creations,
        lane drops, and speed limit changes (including lane-specific speed limits).  The specific roadway geometry in
        use is defined in the supporting Roadway class.  This top level class is designed to be agnostic to the
        specifics of the geometry.  However, it is understood that any Roadway identifies its lanes with integer IDs
        in [0, N], such that lane 0 is the left-most lane and IDs increase going from left to right.


        THIS VERSION ONLY SUPPORTS A SINGLE LEARNING AGENT VEHICLE.  Once it is proven successful, the env will be
        enhanced to train multiple agents.


        This class also manages a configrable number of vehicles.  All vehicles follow basic environment "dynamics" of
        any roadway, such as lane boundaries & connectivity, speed limits, and target destinations, where the rewards
        are influenced by how the vehicle interacts with these elements.  It provides severe negative rewards for 3
        failure conditions:  crashing into another vehicle, running off-road, and stopping in the roadway.  This class
        also provides basic physics of motion, by propagating each vehicle's forward motion according to specified
        limits of acceleration, jerk and lane change speed.  Each vehicle in the scenario must specify its own physical
        properties (accel & jerk limits, lane change speed) and must provide its own tactical guidance algorith, which
        takes observations from this environment and produces an action vector.  Vehicles provide these capabilities by
        inheriting the basic structure of the abstract Vehicle class.  Therefore, any number of vehicle instances
        managed by this class may use the same vehicle model (including the same guidance policy), or each of them
        could use a different model.

        This simple enviroment assumes perfect traction and steering response, so that the physics of driving can
        essentially be ignored.  We will assume that all cars follow their chosen lane's centerlines at all times, except
        when changing lanes.  So there is no steering action required, per se, but there may be opportunity to change
        lanes left or right whenever a target lane is reachable in the chosen direction.  When a vehice decides to begin
        a lane change maneuver, it must spend several time steps executing that maneuver (no instantaneous lateral
        motion), the exact number being configrable in the specific vehicle model.  Once a lane change has begun, it must
        finish; lane change commands are ignored while such a maneuver is underway.

        If a lane change is commanded when there is no reachable lane in the chosen direction, the vehicle will be
        considered to have driven off-road into the grass.  Likewise, if the vehicle's current lane ends and the
        vehicle does not change lanes out of it, and runs straight off the end of the original lane, it is considered
        an off-road event.

        The environment is a continuous flat planar space in the map coordinate frame, with the X-axis origin at the
        left end of the farthest left lane. The location of any vehicle is represented by its X value and lane ID
        (which constrains Y), so the Y origin is arbitrary, and only used for graphical output.

        In order to support data transforms to/from a neural network, the geometry is easier modeled schematically, where
        all lanes are parallel and "adjacent". But any given lane may not always be reachable (via lane change) from an
        adjacent lane (e.g. due to a jersey wall or grass in between the physical lanes). We call this schematic
        representation the parametric coordinate frame, which uses coordinates P and Q as analogies to X and Y in the map
        frame. In practice, the Q coordinate is never needed, since it is implied by the lane ID, which is more readily
        available and important. It is possible to use this parametric frame because we ASSUME that the vehicles are
        generally docile and not operating at their physical performance limits (acceleration, traction), and therefore
        can perform any requested action regardless of the shape of the lane at that point. The parametric frame preserves
        the length of each lane segment, even though it "folds" the angled lanes back so that all lanes in the parametric
        frame are parallel to the P/X axis.

        There is no communication among the vehicles, only (perfect) observations from their own onboard sensors.

        OBSERVATION SPACE:  The learning agent's observations are limited to what it can gather from its "sensors" about
        the roadway and other vehicles in its vicinity, as well as some attributes about itself.  Its observation space
        is described in the __init__() method (all floats).  Any non-learning agents (which may be simplistic procedural
        bots) may have different observation spaces.

        ACTION SPACE:  all vehicle agents use the same action space, which is continuous, with the following elements
        (real world values, unscaled):
            target_speed        - the desired forward speed, m/s. Values are in [0, MAX_SPEED].
            lane_chg_cmd        - indicator of the type of lateral motion desired. Since it is represented as a float
                                    in [-1, 1], we interpret it as follows:
                                    [-1, -0.5)      = change left
                                    [-0.5, +0.5]    = stay in current lane
                                    (0.5, 1]        = change right

        Simulation notes:
        + The simulation is run at a configurable time step size, but we recognize that future expansion of this code to
          handle v2v comms will require steps of ~0.1 sec, so good to try to get close to that.
        + Vehicles are only allowed to go forward; they may exceed the posted speed limits, but have a max physical limit,
          so speeds are in the range [0, MAX_SPEED], which applies to all vehicles.
        + The desired accelerations may not achieve equivalent actual accelerations, due to inertia and jerk constraints.
        + If a lane change is commanded it will take multiple time steps to complete.
        + Vehicles are modeled as simple rectangular boxes.  Each vehicle's width fits within one lane, but when it is in a
          lane change state, it is considered to fully occupy both the lane it is departing and the lane it is moving toward.
        + If two (or more) vehicles' bounding boxes touch or overlap, they will be considered to have crashed.

        Agent rewards are provided by a separate reward function.  The reward logic is documented there.
    """

    metadata = {"render_modes": None} #required by Gymnasium
    #metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}


    def __init__(self,
                 config:        EnvContext,             #dict of config params
                 seed:          int             = None, #seed for PRNG
                 render_mode:   int             = None  #Ray rendering info, unused in this version
                ):

        """Wrapper for the real initialization in order to trap stray exceptions."""

        super().__init__()
        try:
            self._init(config, seed, render_mode)
        except Exception as e:
            print("\n///// Exception trapped in HighwayEnv.__init__: ", e)
            raise e


    def _init(self,
                 config:        EnvContext,             #dict of config params
                 seed:          int             = None, #seed for PRNG
                 render_mode:   int             = None  #Ray rendering info, unused in this version
                ):

        """Initialize an object of this class.  Config options are documented in _set_initial_conditions()."""

        # Store the arguments
        #self.seed = seed #Ray 2.0.0 chokes on the seed() method if this is defined (it checks for this attribute also)
        #TODO: try calling self.seed() without storing it as an instance attribute
        if seed is None:
            seed = datetime.now().microsecond
            print("///// init seed = {}".format(seed))
        self.prng = HpPrng(seed = seed)
        self.render_mode = render_mode

        self._set_initial_conditions(config)
        if self.debug > 0:
            print("\n///// HighwayEnv init: config = ", config)
            print("/////                  OBS_SIZE = ", ObsVec.OBS_SIZE)

        #
        #..........Roadway
        #

        # Create the roadway geometry
        self.roadway = Roadway(self.debug)

        # Define the target destinations for the ego vehicle (T targets) and for the bot vehicles (B targets)
        self.t_targets = []
        self.t_targets.append(TargetDestination(self.roadway, 1, 2900.0))
        self.t_targets.append(TargetDestination(self.roadway, 2, 2900.0))
        self.b_targets = copy.deepcopy(self.t_targets) #bots can seek T targets also
        self.b_targets.append(TargetDestination(self.roadway, 0, 2500.0))
        self.b_targets.append(TargetDestination(self.roadway, 4, 1600.0))

        #
        #..........Vehicles
        #

        # Get config data for the vehicles used in this scenario - the ego vehicle (where the agent lives) is index 0.
        # Normally, this would be wrapped in a try-except block, but Ray makes it very difficult to see the exception
        # in that case. So we let it fail ugly and the problem is much easier to spot.
        vc = self.vehicle_config
        v_data = vc["vehicles"]
        self.num_vehicles = len(v_data)

        # Instantiate model and guidance objects for each vehicle, then use them to construct the vehicle object
        self.vehicles = []
        for i in range(self.num_vehicles):
            # Mark this vehicle as ego if it is index 0 and it is going to be learning (i.e. not in embed collection mode)
            is_ego =  i == 0  and  (self.scenario < 20  or  self.scenario > 29)
            v = None
            spec = v_data[i]
            targets = self.t_targets if is_ego  else  self.b_targets #list of possible targets to navigate to
            try:
                model = getattr(sys.modules[__name__], spec["model"])(self.roadway,
                                    max_jerk      = spec["max_jerk"],
                                    max_accel     = spec["max_accel"],
                                    length        = spec["length"],
                                    lc_duration   = spec["lc_duration"],
                                    time_step     = self.time_step_size)
                guidance = getattr(sys.modules[__name__], spec["guidance"])(self.prng, self.roadway, targets)
                v = Vehicle(model, guidance, self.prng, self.roadway, is_ego, self.time_step_size, self.debug)
            except AttributeError as e:
                print("///// HighwayEnv.__init__: problem with config for vehicle ", i, " model or guidance: ", e)
                raise e
            except Exception as e:
                print("///// HighwayEnv.__init__: problem creating vehicle model, guidance, or the vehicle itself: ", e)
                print("Exception type is ", type(e))
                raise e

            self.vehicles.append(v)
            guidance.set_vehicle(v) #let the new guidance object know about the vehicle it is driving
        if self.debug > 1:
            print("///// HighwayEnv.__init__: {} vehicles constructed.".format(len(self.vehicles)))

        #
        #..........Define the observation space
        #

        # See the ObsVec class for detailed explanation of the observation space and how it is indexed.

        # Gymnasium requires a member variable named observation_space. Since we are dealing with world scale values here, we
        # will need a wrapper to scale the observations for NN input. That wrapper will also need to use self.observation_space.
        # So here we must anticipate that scaling and leave the limits open enough to accommodate it.
        lower_obs = np.full((ObsVec.OBS_SIZE), -1.0) #most values are -1, so only the others are explicitly set below
        lower_obs[ObsVec.SPEED_CMD]             = 0.0
        lower_obs[ObsVec.SPEED_CMD_PREV]        = 0.0
        lower_obs[ObsVec.STEPS_SINCE_LN_CHG]    = 0.0
        lower_obs[ObsVec.SPEED_CUR]             = 0.0
        lower_obs[ObsVec.SPEED_PREV]            = 0.0
        lower_obs[ObsVec.FWD_DIST]              = 0.0
        lower_obs[ObsVec.FWD_SPEED]             = -Constants.MAX_SPEED

        upper_obs = np.ones(ObsVec.OBS_SIZE) #most values are 1 (all values in sensor zones are limited to 1)
        upper_obs[ObsVec.SPEED_CMD]             = Constants.MAX_SPEED
        upper_obs[ObsVec.SPEED_CMD_PREV]        = Constants.MAX_SPEED
        upper_obs[ObsVec.LC_CMD]                = LaneChange.CHANGE_RIGHT
        upper_obs[ObsVec.LC_CMD_PREV]           = LaneChange.CHANGE_RIGHT
        upper_obs[ObsVec.STEPS_SINCE_LN_CHG]    = Constants.MAX_STEPS_SINCE_LC
        upper_obs[ObsVec.SPEED_CUR]             = Constants.MAX_SPEED
        upper_obs[ObsVec.SPEED_PREV]            = Constants.MAX_SPEED
        upper_obs[ObsVec.FWD_DIST]              = Constants.REFERENCE_DIST
        upper_obs[ObsVec.FWD_SPEED]             = Constants.MAX_SPEED

        self.observation_space = Box(low = lower_obs, high = upper_obs, dtype = float)
        if self.debug > 1:
            print("///// observation_space = ", self.observation_space)

        self.all_obs = np.zeros((self.num_vehicles, ObsVec.OBS_SIZE)) #one row for each vehicle
        self._verify_obs_limits("init after space defined")

        #
        #..........Define the action space
        #

        # Specify these for what the NN will deliver, not world scale
        lower_action = np.array([-1.0, -1.0])
        upper_action = np.array([ 1.0,  1.0])
        self.action_space = Box(low=lower_action, high = upper_action, dtype = float)
        if self.debug > 1:
            print("///// action_space = ", self.action_space)

        #
        #..........Remaining initializations
        #

        # Other persistent data
        self.total_steps = 0        #num time steps for this trial (worker), across all episodes; NOTE that this is different from the
                                    # total steps reported by Ray tune, which is accumulated over all rollout workers
        self.steps_since_reset = 0  #length of the current episode in time steps
        self.stopped_count = 0      #num consecutive time steps in an episode where vehicle speed is almost zero
        self.episode_count = 0      #number of training episodes (number of calls to reset())
        self.rollout_id = hex(int(self.prng.random() * 65536))[2:].zfill(4) #random int to ID this env object in debug logging
        print("///// Initializing env environment ID {} with configuration: {}".format(self.rollout_id, vc["title"]))

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        if self.debug == 2:
            print("///// init complete.")


    def seed(self, seed = None):
        """A required method that is apparently not yet supported in Ray 2.0.0."""
        pass
        #print("///// In environment seed - incoming seed value = ", seed)
        #self.seed = seed
        #super().seed(seed)


    def set_task(self,
                 task:      int         #ID of the task (lesson difficulty) to simulate, in [0, n)
                ) -> None:
        """Defines the difficulty level of the environment, which can be used for curriculum learning."""
        pass
        #raise NotImplementedError("HighwayB.set_task() has been deprecated.") #TODO - get rid of this & getter when sure no longer needed


    def get_task(self) -> int:
        """Returns the environment difficulty level currently in use."""

        return 0


    def reset(self, *,
              seed:         int             = None,    #reserved for a future version of gym.Environment
              options:      dict            = None
             ) -> Tuple[np.array, dict]:

        """Wrapper around the real reset method to trap for unhandled exceptions."""

        try:
            return self._reset(seed = seed, options = options)
        except Exception as e:
            print("\n///// Exception trapped in HighwayEnv.reset: ", e)
            raise e


    def _reset(self, *,
              seed:         int             = None,    #reserved for a future version of gym.Environment
              options:      dict            = None
             ) -> Tuple[np.array, dict]:

        """Reinitializes the environment to prepare for a new episode.  This must be called before
            making any calls to step().

            Return tuple includes an array of observation values, plus a dict of additional info key-value
            pairs.

            CAUTION: the returned observation vector is at actual world scale and needs to be
                     preprocessed before going into a NN!
        """

        EARLY_EPISODES = 10000 #num episodes to apply early curriculum to

        if self.debug > 0:
            print("\n///// Entering reset")

        # We need the following line to seed self.np_random
        #super().reset(seed=seed) #apparently gym 0.26.1 doesn't implement this method in base class!
        #self.seed = seed #okay to pass it to the parent class, but don't store a local member copy!

        # options may be a dict that can specify additional configurations - unique to each particular env
        if options is not None and len(options) > 0:
            print("\n///// HighwayEnv.reset: incoming options is: ", options)
            raise ValueError("reset() called with options, but options are not used in this environment.")

        # Copy the user-desired scenario to an effective value that can be changed just for this episode
        self.effective_scenario = self.scenario

        #
        #..........Set the initial conditions for each vehicle, depending on the scenario config
        #

        # Clear any lingering observations from the previous episode
        self.all_obs = np.zeros((self.num_vehicles, ObsVec.OBS_SIZE))

        # Clear the locations of all vehicles to get them out of the way for the random placement below
        for i in range(self.num_vehicles):
            self.vehicles[i].reset()

        # Solo bot vehicle that runs a single lane at its speed limit (useful for inference only)
        if self.effective_scenario >= 90:
            if self.effective_scenario - 90 >= self.roadway.NUM_LANES:
                raise ValueError("///// Attempting to reset to unknown scenario {}".format(self.effective_scenario))

            for i in range(self.num_vehicles):
                self.vehicles[i].active = False

            lane_id = self.effective_scenario - 90
            self.vehicles[1].reset(init_lane_id = lane_id, init_ddt = 0.0, init_speed = self.roadway.lanes[lane_id].segments[0][5])

        # Trainee ego vehicle involved with bot vehicles - randomize, depending on the specific scenario called for
        else:

            # Define the ego vehicle's location - since it's the first vehicle to be placed, anywhere will be acceptable,
            # as long as it has enough room to run at least half an episode before reaching end of lane (note that the two
            # ego targets are 100 m from the ends of their respective lanes). For scenarios 10-19 use it to specify the ego
            # vehicle's starting lane.
            ego_lane_id = 1
            if self.training  and  self.episode_count < EARLY_EPISODES  and  self.roadway.NUM_LANES == 6: #give preference to lanes 0, 4 & 5 in early episodes
                draw = self.prng.random()
                if draw < 0.25:
                    ego_lane_id = 0
                elif draw < 0.5:
                    ego_lane_id = 4
                elif draw < 0.75:
                    ego_lane_id = 5
                elif draw < 0.90:
                    ego_lane_id = 3
                elif draw < 0.95:
                    ego_lane_id = 2
            else:
                ego_lane_id = int(self.prng.random() * self.roadway.NUM_LANES)

            # Scenarios 10-19:  specify ego vehicle starting lane
            if 10 <= self.effective_scenario < 10 + Roadway.NUM_LANES:
                ego_lane_id = self.effective_scenario - 10

            # Define the starting P coordinate - for early training episodes, give preference toward the end of the lane, so it experiences failures
            lane_begin = self.roadway.get_lane_start_p(ego_lane_id)
            lane_length = self.roadway.get_total_lane_length(ego_lane_id)
            ego_p = lane_begin
            if self.episode_count < EARLY_EPISODES:
                if self.prng.random() < 0.3:
                    ego_p = self.prng.random() * max(lane_length - 150.0, 1.0) + lane_begin
                else:
                    ego_p = max(lane_begin + lane_length - self.prng.random()*500.0, lane_begin)

            if not self.training: #encourage it to start closer to beginning of the track for inference runs
                ego_p = self.prng.random() * 0.5*max(lane_length - 150.0, 1.0) + lane_begin

            # Randomly define the starting speed and initialize the vehicle data
            speed = self.prng.random() * Constants.MAX_SPEED
            self.vehicles[0].reset(init_lane_id = ego_lane_id, init_p = ego_p, init_speed = speed)
            if self.debug > 0:
                print("    * reset: ego lane = {}, p = {:.1f}, speed = {:4.1f}".format(ego_lane_id, ego_p, speed))

            # Choose how many vehicles will participate
            episode_vehicles = self._decide_num_vehicles()

            # If scenario is specified as 0 and we are in training mode, then randomly change to scenarios 1 or 2 to
            # properly train for various empty lane situations.
            if self.effective_scenario == 0  and  self.training:
                draw = self.prng.random()
                if draw < 0.05:
                    self.effective_scenario = 1 #all neighbors in ego's lane
                elif draw < 0.25:
                    self.effective_scenario = 2 #no neighbors in ego's lane

            # Randomize all participating vehicles within a box around the ego vehicle to maximize exercising its sensors
            deactivated_count = 0
            for i in range(1, self.num_vehicles):

                # Mark unused vehicles as inactive and skip over
                if i >= episode_vehicles:
                    self.vehicles[i].active = False
                    continue

                # Iterate until a suitable location is found for it
                space_found = False
                attempt = 0
                p = None
                min_p = ego_p - Constants.N_DISTRO_DIST_REAR
                max_p = ego_p + Constants.N_DISTRO_DIST_FRONT
                while not space_found  and  attempt < 20:
                    attempt += 1
                    # If too many vehicles packed in there, relax the boundaries on P to allow more freedom
                    if attempt == 6:
                        min_p -= 100.0
                        max_p += 100.0
                    elif attempt == 10:
                        min_p -= 200.0
                        max_p += 200.0
                    elif attempt == 14:
                        min_p -= 500.0
                        max_p += 500.0
                    elif attempt == 18:
                        min_p -= 1200.0
                        max_p += 1200.0

                    lane_id = self._select_bot_lane(ego_lane_id)
                    lane_begin = self.roadway.get_lane_start_p(lane_id)
                    lane_end = lane_begin + self.roadway.get_total_lane_length(lane_id)
                    p_lower = max(min_p, lane_begin)
                    p_upper = min(max_p, lane_end - Constants.CONSERVATIVE_LC_DIST)
                    if p_upper <= p_lower:
                        #print("///// WARNING: HighwayEnv.reset: difficulty positioning vehicle {} on lane {}, attempt {}, min_p = {:.1f}, "
                        #      .format(i, lane_id, attempt, min_p), \
                        #      "max_p = {:.1f}, p_lower = {:.1f}, p_upper = {:.1f}".format(max_p, p_lower, p_upper))
                        continue
                    p = self.prng.random()*(p_upper - p_lower) + p_lower
                    space_found = self._verify_safe_location(i, lane_id, p)
                    #print("***** vehicle {}, lane {}, p = {:.1f}, space_found = {}".format(i, lane_id, p, space_found))

                if not space_found:
                    self.vehicles[i].active = False
                    deactivated_count += 1
                    #print("///// reset: no space found for vehicle {}; deactivating it.".format(i))
                    continue

                # Pick a speed, then initialize this vehicle - if this vehicle is close behind ego then limit its speed to be similar
                # to avoid an immediate rear-ending.
                speed = self.prng.random() * (Constants.MAX_SPEED - 20.0) + 20.0
                vlen = self.vehicles[i].model.veh_length
                if i > 0  and  lane_id == self.vehicles[0].lane_id  and  3.0*vlen <= self.vehicles[0].p - p <= 8.0*vlen:
                    speed = min(speed, 1.1*self.vehicles[0].cur_speed)
                #print("***** reset: vehicle {}, lane = {}, p = {:.1f}, min_p = {:.1f}, max_p = {:.1f}".format(i, lane_id, p, min_p, max_p))
                self.vehicles[i].reset(init_lane_id = lane_id, init_p = p, init_speed = speed)

            """
            if deactivated_count == 0:
                print("***** reset: bots that don't fit = {}".format(deactivated_count))
            else:
                print("***** reset: bots that don't fit = {}; ego lane = {}, p = {:.1f}, effective scenario = {}"
                      .format(deactivated_count, ego_lane_id, ego_p, self.effective_scenario))
            """

        if self.debug > 0:
            print("///// HighwayEnv.reset: all vehicle starting configs defined.")

        #
        #..........Gather the observations from the appropriate vehicles & wrap up
        #

        # Initialize the num steps since previous lane change to maximum, since there is no history and we don't want to
        # discourage the initial LC.
        self.all_obs[0, ObsVec.STEPS_SINCE_LN_CHG] = Constants.MAX_STEPS_SINCE_LC

        # We must do this after all vehicles have been initialized, otherwise obs from the vehicles placed first won't
        # include sensing of vehicle placed later.

        # Get the obs from each vehicle
        dummy_actions = [0.5*Constants.MAX_SPEED, LaneChange.STAY_IN_LANE]
        for i in range(self.num_vehicles):
            self.all_obs[i, :] = self.vehicles[i].model.get_obs_vector(i, self.vehicles, dummy_actions, self.all_obs[i, :])

        # Other persistent data
        self.steps_since_reset = 0
        self.episode_count += 1

        if self.debug > 0:
            print("///// End of reset(). Returning obs ", self.all_obs[0, :])
        return self.all_obs[0, :], {} #only return the row for the ego vehicle


    def step(self,
                cmd     : list      #list of floats; 0 = speed command, 1 = desired lane, scaled
            ) -> Tuple[np.array, float, bool, bool, Dict]:

        """Wrapper around the real step method to trap unhandled exceptions."""

        try:
            return self._step(cmd)
        except Exception as e:
            print("\n///// Exception trapped in HighwayEnv.step: ", e)
            raise e


    def _step(self,
                cmd     : list      #list of floats; 0 = speed command, 1 = desired lane, scaled
            ) -> Tuple[np.array, float, bool, bool, Dict]:

        """Executes a single time step of the environment.  Determines how the input commands (actions) will alter the
            simulated world and returns the resulting observations.

            Return is array of new observations, new reward, done flag, truncated flag, and a dict of additional info.

            CAUTION: the returned observation vector is at actual world scale and needs to be
                     preprocessed before going into a NN!

            The process is:
                - gather tactical guidance commands based on existing observations (for the ego vehicle, these come in as
                  input args; for other vehicles their model needs to generate)
                - pass those commands to the dynamic models and move each vehicle to its new state
                - collect each entity's observations from that new state
        """

        if self.debug > 0:
            print("\n///// Entering step(): ego cmd = ", cmd)
            print("      vehicles array contains:")
            for i, v in enumerate(self.vehicles):
                v.print(i)

        self.total_steps += 1
        self.steps_since_reset += 1
        done = False
        truncated = False
        return_info = {"reason": "Unknown"}

        #
        #..........Update states of all vehicles
        #

        # Unscale the ego action inputs (both cmd values are in [-1, 1])
        ego_action = [None]*2
        ego_action[0] = (cmd[0] + 1.0)/2.0 * Constants.MAX_SPEED
        #raw_lc_cmd = min(max(cmd[1]*5.0/2.0, -1.0), 1.0) #allows threshold of +/- 0.2 for boundary between same lane and changing to adjacent
        raw_lc_cmd = min(max(cmd[1], -1.0), 1.0) #command threshold is +/- 0.5
        ego_action[1] = int(math.floor(raw_lc_cmd + 0.5)) #TODO: update doc/comment descriptions of cmd interpretation if this is a keeper.
        if self.steps_since_reset < 2: #force it to stay in lane for first time step
            ego_action[1] = 0.0
        #print("***** Entering step ", self.steps_since_reset, ": LC command = ", ego_action[1])

        # Loop through all active vehicles. Note that the ego vehicle is always at index 0.
        vehicle_actions = [None]*self.num_vehicles
        action = ego_action
        reached_tgt = False
        for i in range(self.num_vehicles):
            if not self.vehicles[i].active:
                continue

            # Exercise the tactical guidance algo to generate the next action commands for vehicles that aren't in training.
            if i > 0:
                #print("***   step: guiding vehicle {:2d} at lane {}, p {:.1f}, speed {:.1f}, LC count {}"
                #      .format(i, self.vehicles[i].lane_id, self.vehicles[i].p, self.vehicles[i].cur_speed, self.vehicles[i].lane_change_count))
                action = self.vehicles[i].guidance.step(self.all_obs[i, :]) #unscaled

            # Store the actions for future reference
            vehicle_actions[i] = action

            # Apply the appropriate dynamics model to each vehicle in the scenario to get its new state.
            new_speed, new_p, new_lane, reason = self.vehicles[i].advance_vehicle_spd(action[0], action[1]) #TODO: do we need these return values?
            if self.debug > 1:
                print("      Vehicle {} advanced with new_speed_cmd = {:.2f}. new_speed = {:.2f}, new_p = {:.2f}, new_lane = {}"
                        .format(i, action[0], new_speed, new_p, new_lane))

            # If the ego vehicle has reached one of its target destinations, it is a successful episode
            if i == 0:
                for t in self.t_targets:
                    if self.vehicles[0].lane_id == t.lane_id  and  new_p >= t.p:
                        reached_tgt = True
                        break

                if reached_tgt:
                    done = True
                    return_info["reason"] = "SUCCESS - reached target in lane {}!".format(self.vehicles[0].lane_id)
                    #print("/////+ step: {} step {}, success - completed the track".format(self.rollout_id, self.total_steps))

                # Else if it ran off the road or stopped, then end the episode in failure
                elif self.vehicles[0].off_road  or  self.vehicles[0].stopped:
                    done = True
                    return_info["reason"] = reason

                # Else if we have completed an episode (route fragment) consider it successfully done, but also truncated (hit step limit)
                elif self.steps_since_reset >= self.episode_length:
                    done = True
                    return_info["reason"] = "Reached max steps allowed."
                    truncated = True

            # Individual vehicle models must update the common observation elements as well as those specific to their vehicle type.
            # This may feel redundant, but it allows the top level env class to stay out of the observation business altogether.

        if self.debug > 1:
            print("      all vehicle dynamics updated.")

        #
        #..........Gather the results and observations and set the reward
        #

        # Get the sensor observations from each vehicle
        try:
            for i in range(self.num_vehicles):
                self.all_obs[i, :] = self.vehicles[i].model.get_obs_vector(i, self.vehicles, vehicle_actions[i], self.all_obs[i, :])
        except AssertionError as e:
            v = self.vehicles[i]
            print("///// AssertionError trapped in _step while getting obs vector for vehicle {} with LC stat = {}, count = {}, cur_speed = {:.1f}, "
                    .format(i, v.lane_change_status, v.lane_change_count, v.cur_speed), \
                  "prev_speed = {:.1f}, active = {}, off_road = {}, stopped = {}, stopped_count = {}"
                    .format(v.prev_speed, v.active, v.off_road, v.stopped, v.stopped_count))

        # For the ego vehicle, run its planning algo. This call doesn't fit well here, but is needed until the planner can be
        # replaced with a NN. This will replace a few elements in the obs vector.
        #TODO: eventually replace this call with a NN in the guidance class.
        if self.vehicles[0].active:
            self.all_obs[0, :] = self.vehicles[0].guidance.plan_route(self.all_obs[0, :])
        self._verify_obs_limits("step() before collision check on step {}".format(self.steps_since_reset))

        # Check that none of the vehicles has crashed into another, accounting for a lane change in progress taking up both lanes.
        crash = self._check_for_collisions()
        if crash:
            done = True
            return_info["reason"] = "Two (or more) vehicles crashed."

        # Determine the reward resulting from this time step's action
        reward, expl = self._get_reward(done, crash, self.vehicles[0].off_road, self.vehicles[0].stopped, reached_tgt)
        return_info["reward_detail"] = expl
        #print("***** step: {} step {}, returning reward of {:.4f}, {}".format(self.rollout_id, self.steps_since_reset, reward, expl)) #TODO debug
        #print("      partial obs = ", self.all_obs[0, 0:20])

        if self.debug > 0:
            print("///// step {} complete. Returning obs (only first row for ego vehicle).".format(self.steps_since_reset))
            print("      reward = ", reward, ", done = ", done)
            print("      final vehicles array =")
            for i, v in enumerate(self.vehicles):
                v.print(i)
            print("      reason = {}".format(return_info["reason"]))
            print("      reward_detail = {}\n".format(return_info["reward_detail"]))

        return self.all_obs[0, :], reward, done, truncated, return_info


    def get_stopper(self):
        """Returns the stopper object."""
        return self.stopper


    def get_burn_in_iters(self):
        """Returns the number of burn-in iterations configured."""
        return self.burn_in_iters


    def get_total_steps(self):
        """Returns the total number of time steps executed so far."""
        return self.total_steps


    def get_vehicle_data(self) -> List:
        """Returns a list of all the vehicles in the scenario, with the ego vehicle as the first item."""

        return self.vehicles


    def close(self):
        """Closes any resources that may have been used during the simulation."""
        pass #this method not needed for this version


    ##### internal methods #####


    def _set_initial_conditions(self,
                                config:     EnvContext
                               ):
        """Sets the initial conditions of the ego vehicle in member variables."""

        self.burn_in_iters = 0  #TODO: still needed?
        try:
            bi = config["burn_in_iters"]
            if bi > 0:
                self.burn_in_iters = int(bi)
        except KeyError as e:
            pass

        self.time_step_size = 0.1 #duration of a time step, seconds
        try:
            ts = config["time_step_size"]
        except KeyError as e:
            ts = None
        if ts is not None  and  ts != ""  and  float(ts) > 0.0:
            self.time_step_size = float(ts)

        self.episode_length = 999999 #mas number of time steps in an episode (large default supports inference)
        try:
            el = config["episode_length"]
            if el > 0:
                self.episode_length = int(el)
        except KeyError as e:
            pass

        self.debug = 0 #level of debug printing (0=none (default), 1=moderate, 2=full details)
        try:
            db = config["debug"]
        except KeyError as e:
            db = None
        if db is not None  and  db != ""  and  0 <= int(db) <= 2:
            self.debug = int(db)

        self.crash_report = False #should step() log the details of every crash experienced?
        try:
            cr = config["crash_report"]
            if cr:
                self.crash_report = True
        except KeyError as e:
            pass

        self.training = False #is the environment being used in a training job? (affects scaling of observations)
        try:
            tr = config["training"]
            if tr:
                self.training = True
        except KeyError as e:
            pass

        self.ignore_neighbor_crashes = False #should the collision detector ignore crashes between two non-ego vehicles?
        try:
            inc = config["ignore_neighbor_crashes"]
            if inc:
                self.ignore_neighbor_crashes = True
        except KeyError as e:
            pass

        self.verify_obs = False #verify that the obs vector values are all within the specified limits (runs slower)?
        try:
            vo = config["verify_obs"]
            self.verify_obs = vo
        except KeyError as e:
            pass

        self.vehicle_config = {} #dict of configs specific to the fleet of vehicles
        vcf = None
        try:
            vcf = config["vehicle_file"]
            with open(vcf, 'r') as stream:
                d = yaml.load(stream, Loader = Loader) #TODO: replace with call to safe_load() to plug security risk (https://pyyaml.org/wiki/PyYAMLDocumentation)
                self.vehicle_config = d
        except Exception as e:
            print("///// Exception in loading YAML file {}: {}".format(vcf, e))

        self.scenario = 0 #indicates the initial conditions of the scenario
        try:
            s = config["scenario"]
            if 0 <= s < 100:
                self.scenario = int(s)
        except KeyError as e:
            pass


    def _decide_num_vehicles(self) -> int:
        """Uses a weighted random draw to decide how many vehicles to use in the episode, favoring a small number in early
            training episodes, gradually favoring more as the number of epsisodes increases. For inference, we will push the
            favored band to the highest level.
        """

        MANY_EPISODES = 5000
        nv = self.num_vehicles
        assert nv > 1, "///// NEED AT LEAST 2 VEHICLES DEFINED."
        nv23 = max(2*nv//3, 1)
        nv3 = max(nv//3, 1)
        fav_low = float(nv23) #max out the range for inference runs
        fav_high = float(nv)
        fraction = min(self.episode_count/MANY_EPISODES, 1.0)
        if self.training:
            fav_low = min(float(nv23 - 1.0)*fraction + 1.0, float(nv23)) #lower bound for the favorite number range
            fav_high = min(float(nv - nv3)*fraction + nv3, float(nv)) #upper bound for the favorite number range
        #print("///// decide_num_vehicles: nv23 = {}, nv3 = {}, fav_low = {}, fav_high = {}".format(nv23, nv3, fav_low, fav_high)) #TODO debug
        assert fav_high >= fav_low, "///// ERROR in reset(): fav_high = {}, fav_low = {}".format(fav_high, fav_low)
        episode_vehicles = 1
        draw1 = self.prng.random() #determines if we are in the favored band or not
        draw2 = self.prng.random() #chooses the value from within the selected band
        if draw1 > 0.2: #we are in the favored band
            episode_vehicles = int(draw2*(fav_high - fav_low) + fav_low + 0.5)
        else: #below the favored band
            episode_vehicles = max(int(draw2*(fav_low - 1.0) + 1.0 + 0.5), 1)

        # Limit the number if we are running scenario 1, which places all vehicles in the same lane
        if self.effective_scenario == 1:
            episode_vehicles = min(episode_vehicles, 6)

        #print("      draw1 = {:.2f}, draw2 = {:.2f}, episode_vehicles = {}".format(draw1, draw2, episode_vehicles)) #TODO debug
        #print("*     episode {}, episode_vehicles = {}".format(self.episode_count, episode_vehicles))
        return episode_vehicles


    def _select_bot_lane(self,
                         ego_lane   : int,      #lane ID where the ego vehicle was placed
                        ) -> int:
        """Chooses the initial lane for a bot vehicle based on the specified scenario (only for scenarios < 10)."""

        # Case 1 - all bots in ego's lane
        if self.effective_scenario == 1:
            return ego_lane

        lane = int(self.prng.random() * self.roadway.NUM_LANES)

        # Case 2 - no bots in ego's lane
        if self.effective_scenario == 2:
            while lane == ego_lane:
                lane = int(self.prng.random() * self.roadway.NUM_LANES)

        # Case 0 - everything else
        return lane


    def _verify_safe_location(self,
                              n         : int,  #neighbor ID
                              lane_id   : int,  #desired lane ID for the neighbor
                              p         : float,#desired P coordinate for the neighbor (m in paremetric frame)
                             ) -> bool:         #returns true if the indicated location is safe
        """Determines if the candidate location (lane & P coordinate) is a safe place to put a vehicle at the beginning of a scenario.
            It needs to be sufficiently far from any other neighbors whose starting locations have already been defined so that there
            won't be an immediate crash after the sim starts. When this is called we won't necessarily know the candidate's speed, so
            need to consider that it could be way different from any of the other vehicles nearby.
        """

        assert 0 <= lane_id < self.roadway.NUM_LANES, "///// Attempting to place neighbor {} in invalid lane {}".format(n, lane_id)
        start = self.roadway.get_lane_start_p(lane_id)
        assert start <= p < start + self.roadway.get_total_lane_length(lane_id), \
                "///// Attempting to place neighbor {} in lane {} at invalid p = {:.1f}".format(n, lane_id, p)

        # This is the safe longitudinal distance between a slow vehicle in front, going 4 m/s, and a fast vehicle behind, going 30 m/s,
        # with a -3 m/s^2 decel capability and assuming the front vehicle holds a steady speed. It will take this distance plus the
        # distance covered by the front vehicle during that time for the rear vehicle to decelerate to 4 m/s. There could be worse
        # conditions, but they should be quite rare.
        SAFE_SEPARATION = 113.0
        safe = True

        # Loop through all active vehicles
        for o in range(self.num_vehicles):
            other = self.vehicles[o]
            if not other.active:
                continue

            # If the other vehicle is in candiate's lane then check if it is too close longitudinally. Note that if a neighbor has
            # not yet been placed, its lane ID is -1
            if other.lane_id == lane_id:
                if 0.0 <= abs(other.p - p) < SAFE_SEPARATION: #6.0*other.model.veh_length:
                    safe = False

        return safe


    def _check_for_collisions(self) -> bool:
        """Compares location and bounding box of each vehicle with all other vehicles to determine if there are
            any overlaps.  If any two vehicle bounding boxes overlap, then returns True, otherwise False.

            Return: has there been a collision we are interested in?
            Note that if the collision is between two neighbors (ego not involved) then return value depends on
            the config setting "ignore_neighbor_crashes".
        """

        if self.debug > 1:
            print("///// Entering _check_for_collisions")
        crash = False

        # Loop through all active vehicles but the final one to get vehicle A
        for i in range(self.num_vehicles - 1):
            va = self.vehicles[i]
            if not va.active:
                continue

            # Loop through the remaining active vehicles to get vehicle B
            for j in range(i + 1, self.num_vehicles):
                vb = self.vehicles[j]
                if not vb.active:
                    continue

                # If A and B are in the same lane, then
                if va.lane_id == vb.lane_id:

                    # If they are within one car length of each other, it's a crash
                    if abs(va.p - vb.p) <= 0.5*(va.model.veh_length + vb.model.veh_length):

                        # Mark the involved vehicles as out of service
                        va.active = False
                        vb.active = False
                        va.crashed = True
                        vb.crashed = True
                        if self.crash_report:
                            print("    / CRASH in same lane between vehicles {} and {} near {:.2f} m in lane {}"
                                        .format(i, j, va.p, va.lane_id))

                        # Mark it so only if it involves the ego vehicle or we are worried about all crashes
                        if i == 0  or  j == 0  or  not self.ignore_neighbor_crashes:
                            crash = True
                            if self.crash_report:
                                print("      v0 speed = {:.1f}, prev speed = {:.1f}, lc status = {}, lc count = {}"
                                        .format(self.all_obs[0, ObsVec.SPEED_CUR], self.all_obs[0, ObsVec.SPEED_PREV],
                                                self.vehicles[0].lane_change_status, self.vehicles[0].lane_change_count))
                                k = max(i, j)
                                print("      v{} speed = {:.1f}, lc_status = {}, lc count = {}".format(k, self.vehicles[k].cur_speed,
                                                                                                        self.vehicles[k].lane_change_status,
                                                                                                        self.vehicles[k].lane_change_count))
                            break

                # Else if they are in adjacent lanes, then
                elif abs(va.lane_id - vb.lane_id) == 1:

                    # If either vehicle is changing lanes at the moment, then
                    if va.lane_change_status != "none"  or  vb.lane_change_status != "none":

                        # If the two are within a vehicle length of each other, then
                        if abs(va.p - vb.p) <= 0.5*(va.model.veh_length + vb.model.veh_length):

                            # Check if one or both of them is changing lanes into the other's space
                            if self._conflicting_space(va, vb):

                                # Mark the involved vehicles as out of service
                                va.active = False
                                vb.active = False
                                va.crashed = True
                                vb.crashed = True
                                if self.crash_report:
                                    print("    / CRASH in adjacent lanes between vehicles {} and {} near {:.2f} m in lane {}"
                                                .format(i, j, vb.p, va.lane_id))

                                # Mark it so only if it involves the ego vehicle or we are worried about all crashes
                                if i == 0  or  j == 0  or  not self.ignore_neighbor_crashes:
                                    crash = True
                                    if self.crash_report:
                                        print("      v0 speed = {:.1f}, prev speed = {:.1f}, lc status = {}, lc count = {}"
                                              .format(self.all_obs[0, ObsVec.SPEED_CUR], self.all_obs[0, ObsVec.SPEED_PREV],
                                                      self.vehicles[0].lane_change_status, self.vehicles[0].lane_change_count))
                                        k = max(i, j)
                                        print("      v{} speed = {:.1f}, lc_status = {}, lc count = {}".format(k, self.vehicles[k].cur_speed,
                                                                                                               self.vehicles[k].lane_change_status,
                                                                                                               self.vehicles[k].lane_change_count))
                                    break

            if crash: #the previous break stmts only break out of the inner loop, so we need to break again
                break

        if self.debug > 0:
            print("///// _check_for_collisions complete. Returning ", crash)
        return crash


    def _conflicting_space(self,
                           va       : Vehicle,  #the first vehicle to investigate
                           vb       : Vehicle   #the second vehicle to investigate
                          ) -> bool:
        """Returns true if the two vehicles are likely occupying the same space, false otherwise. It is looking at lateral spacing of two
            side-by-side vehicles, either of which may be in the process of a lane change in either direction. When in a lane change maneuver,
            we ASSUME that the full width of both the origination lane and the destination lane are fully occupied by the vehicle executing
            this maneuver. Therefore, this will flag a crash if either (or both) of the adjacent vehicles is changing lanes where its
            destination lane is the one occupied by the other vehicle, even if the other vehicle is in the process of vacating that lane for
            the next one farther away.
        """

        # If vehicle A is changing lanes toward B then
        if (va.lane_change_status == "right"  and  vb.lane_id - va.lane_id == 1)  or  (va.lane_change_status == "left"  and  va.lane_id - vb.lane_id == 1):

            # Determine A's target lane (it could be B's lane or the lane it is currently in (i.e. its maneuver is almost complete))
            va_tgt = va.lane_id #initial guess that it is already mostly in its target lane
            if va.lane_change_count < va.model.lc_half_steps: #it is < 50% across the dividing line, so it is still registered in its originating lane
                if va.lane_change_status == "right":
                    va_tgt = va.lane_id + 1
                else:
                    va_tgt = va.lane_id - 1

            # If its target is B's lane, then it's a conflict
            if va_tgt == vb.lane_id:
                return True

        # If vehicle B is changing lanes toward A then
        if (vb.lane_change_status == "right"  and  va.lane_id - vb.lane_id == 1)  or  (vb.lane_change_status == "left"  and  vb.lane_id - va.lane_id == 1):

            # Determine B's target lane
            vb_tgt = vb.lane_id #initial guess that it is already mostly in its target lane
            if vb.lane_change_count < vb.model.lc_half_steps: #it is < 50% across the dividing line, so it is still registered in its originating lane
                if vb.lane_change_status == "right":
                    vb_tgt = vb.lane_id + 1
                else:
                    vb_tgt = vb.lane_id - 1

            # If its target is A's lane, then it's a conflict
            if vb_tgt == va.lane_id:
                return True

        # If we have gotten this far, then there is no conflict
        return False


    def _get_reward(self,
                    done    : bool,         #is this the final step in the episode?
                    crash   : bool,         #did one or more of the vehicles crash into each other?
                    off_road: bool,         #did the ego vehicle run off the road?
                    stopped : bool,         #has the vehicle come to a standstill?
                    tgt_reached : bool,     #has the vehicle reached an identified success target?
                   ):
        """Returns the reward for the current time step (float).  The reward should be near [-1, 1] for any situation.
            NOTE: for now we are only looking at the first vehicle's observations (the ego vehicle).
        """

        if self.debug > 1:
            print("///// Entering _get_reward rollout {}, step {}. done = {}, crash = {}, off_road = {}, tgt_reached = {}"
                    .format(self.rollout_id, self.total_steps, done, crash, off_road, tgt_reached))
        reward = 0.0
        explanation = ""

        # Handle inactive ego vehicle
        if not self.vehicles[0].active:
            reward = -1.0
            explanation = "Ego vehicle is not active!"
            return reward, explanation

        # If the episode is done then
        if done:

            # If there was a crash into another car then set a large penalty.
            if crash:
                reward = -1.5
                explanation = "Crashed into a vehicle. "

            # Else if ran off road, set a penalty
            elif off_road:
                reward = -1.0
                explanation = "Ran off road. "

            # Else if the vehicle just stopped in the middle of the road, set a penalty
            elif stopped:
                reward = -1.0
                explanation = "Vehicle stopped. "

            # Else if a target point has been achieved (mostly good for inference demo, but also special cases of complete episode)
            elif tgt_reached:
                reward = 1.0 #
                explanation = "Success: target point reached!"

            # Else (episode ended successfully)
            else:
                reward = 1.0
                explanation = "Successful episode!"

        # Else, episode still underway
        else:

            # Reward for following the route planner's recommendations for lane change activity, only if there is a non-zero recommendation that
            # is not "current lane". Staying in the current lane by default doesn't deserve a reward if it is the only possible choice.

            # CAUTION: this code block is tightly coupled to the design of the BridgitCtrl class.
            lc_desired = self.all_obs[0, ObsVec.DESIRABILITY_LEFT : ObsVec.DESIRABILITY_RIGHT+1]
            des_max = max(lc_desired)
            if des_max < 0.001:
                des_max = 1.0
            lc_cmd = int(self.all_obs[0, ObsVec.LC_CMD]) #CAUTION! this is quantized in [-1, 1]; add 1 to use it as an index
            bonus = 0.0

            # If a lane change has been commanded, give a bonus if it is going in a desirable direction. But don't consider if it's overly
            # redundant (more than twice while the maneuver is underway).
            if lc_cmd != LaneChange.STAY_IN_LANE  and  self.vehicles[0].lane_change_status != "none"  and  self.vehicles[0].lane_change_count < 3:
                cmd_desirability = lc_desired[lc_cmd+1]
                same_lane_desirability = lc_desired[1]
                factor = 0.3
                if cmd_desirability > same_lane_desirability: #command is better than staying put
                    bonus = factor #bonus needs to be rather large, since this will be a rare event
                    explanation += "LC des bonus {:.4f}. ".format(bonus)
                elif cmd_desirability < 0.1: #command was an especially poor choice
                    bonus = -factor
                    explanation += "LC des terrible {:.4f}. ".format(bonus)
                else: #otherwise not desirable
                    bonus = -factor
                    explanation += "LC des poor {:.4f}. ".format(bonus)
            """ previously used; may come back:
            if lc_desired[0] > 0.0  or  lc_desired[2] > 0.0: #left or right are reasonable choices
                bonus = 0.008 * lc_desired[lc_cmd+1] / des_max
                explanation += "LC des bonus {:.4f}. ".format(bonus)
            """
            reward += bonus

            # If a lane change was initiated, apply a penalty depending on how soon after the previous lane change
            if self.vehicles[0].lane_change_count == 1:
                penalty = 0.002*(Constants.MAX_STEPS_SINCE_LC - self.all_obs[0, ObsVec.STEPS_SINCE_LN_CHG]) + 0.002
                reward -= penalty
                explanation += "Ln chg pen {:.4f}. ".format(penalty)

            # Small penalty for widely varying lane commands (these obs are unscaled, so will be integers)
            """
            cmd_diff = abs(self.all_obs[0, ObsVec.LC_CMD] - self.all_obs[0, ObsVec.LC_CMD_PREV])
            penalty = 0.002 * cmd_diff * cmd_diff
            reward -= penalty
            if penalty > 0.0001:
                #print("///// get_reward: LC_CMD = {:.4f}, LC_CMD_PREV = {:.4f}".format(self.all_obs[0, ObsVec.LC_CMD], self.all_obs[0, ObsVec.LC_CMD_PREV])) #TODO debug
                explanation += "Ln cmd pen {:.4f}. ".format(penalty)
            """

            # Small penalty for widely varying speed commands
            cmd_diff = abs(self.all_obs[0, ObsVec.SPEED_CMD] - self.all_obs[0, ObsVec.SPEED_CMD_PREV]) / Constants.MAX_SPEED
            penalty = 0.07 * cmd_diff * cmd_diff
            reward -= penalty
            if penalty > 0.0001:
                explanation += "Spd var pen {:.4f}. ".format(penalty)

            # Penalty for deviating from roadway speed limit only if there isn't a slow vehicle nearby in front
            speed_mult = 0.06
            speed_limit = self.roadway.get_speed_limit(self.vehicles[0].lane_id, self.vehicles[0].p)
            fwd_vehicle_speed = self._get_fwd_vehicle_speed() #large value if no fwd vehicle
            cur_speed = self.all_obs[0, ObsVec.SPEED_CUR]
            penalty = 0.0
            if fwd_vehicle_speed >= speed_limit  or  cur_speed < 0.9*fwd_vehicle_speed:
                norm_speed = cur_speed / speed_limit #1.0 = speed limit
                diff = abs(norm_speed - 1.0)
                if diff > 0.02:
                    penalty = speed_mult*(diff - 0.02)
                    explanation += "spd pen {:.4f}. ".format(penalty)
            reward -= penalty

        if self.debug > 0:
            print("///// reward returning {:.4f} due to crash = {}, off_road = {}, stopped = {}. {}"
                    .format(reward, crash, off_road, stopped, explanation))

        return reward, explanation


    def _get_fwd_vehicle_speed(self) -> float:
        """If there is a vehicle forward of the ego vehicle and in the same lane, within 6 sensor zones of ego,
            then return the speed of that vehicle (the closest one in this range). If no vehicle is in these
            zones, then return MAX_SPEED.
        """

        fwd_speed = Constants.MAX_SPEED

        # Loop through obs zones forward of the ego vehicle to find the first one occupied and get its speed
        for z in range(6):
            z_idx = ObsVec.BASE_CTR_FRONT + z*ObsVec.CTR_ELEMENTS
            occupied = self.all_obs[0, z_idx + ObsVec.OFFSET_OCCUPIED]
            if occupied > 0.5:
                fwd_speed = self.all_obs[0, z_idx + ObsVec.OFFSET_SPEED]*Constants.MAX_SPEED + self.all_obs[0, ObsVec.SPEED_CUR]
                #print("*     get_fwd_vehicle_speed: z = {}, z_idx = {}, obs = {:.4f}, ego cur = {:.1f}, fwd_speed = {:.1f}"
                #      .format(z, z_idx, self.all_obs[0, z_idx+ObsVec.OFFSET_SPEED], self.all_obs[0, ObsVec.SPEED_CUR], fwd_speed)) #TODO debug
                break

        return fwd_speed


    def _verify_obs_limits(self,
                           tag      : str = ""  #optional explanation of where in the code this was called
                          ):
        """Checks that each element of the observation vector is within the limits of the observation space."""

        if not self.verify_obs:
            return

        lo = self.observation_space.low
        hi = self.observation_space.high

        try:
            for v in range(self.num_vehicles):
                for i in range(ObsVec.OBS_SIZE):
                    assert lo[i] <= self.all_obs[v, i] <= hi[i], "\n///// obs[{}, {}] value ({}) is outside bounds {} and {}" \
                                                            .format(v, i, self.all_obs[v, i], lo[i], hi[i])

        except AssertionError as e:
            print(e)
            print("///// Sample obs vector content at: {}:".format(tag))
            for v in range(min(self.num_vehicles, 2)):
                print("----- Vehicle {}:".format(v))
                for j in range(100):
                    j1 = j + 100
                    j2 = j + 200
                    j3 = j + 300
                    j4 = j + 400
                    j5 = j + 500
                    if j5 < ObsVec.OBS_SIZE:
                        print("      {:3d}:{:5.2f}  {:3d}:{:5.2f}  {:3d}:{:5.2f}  {:3d}:{:5.2f}  {:3d}:{:5.2f}  {:3d}:{:5.2f}"
                              .format(j, self.all_obs[v, j], j1, self.all_obs[v, j1], j2, self.all_obs[v, j2],
                                      j3, self.all_obs[v, j3], j4, self.all_obs[v, j4], j5, self.all_obs[v, j5]))
                    else:
                        print("      {:3d}:{:5.2f}  {:3d}:{:5.2f}  {:3d}:{:5.2f}  {:3d}:{:5.2f}  {:3d}:{:5.2f}"
                              .format(j, self.all_obs[v, j], j1, self.all_obs[v, j1], j2, self.all_obs[v, j2],
                                      j3, self.all_obs[v, j3], j4, self.all_obs[v, j4]))
