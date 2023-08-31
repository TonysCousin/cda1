import sys
from collections import deque
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
from bot_type1_model import BotType1Model
from bot_type1_ctrl import BotType1Ctrl


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
        properties (accel & jerk limits, lane change speed) and must provide its own control algorith, which takes
        observations from this environment and produces an action vector.  Vehicles provide these capabilities by
        inheriting the basic structure of the abstract Vehicle class.  Therefore, any number of vehicle instances
        managed by this class may use the same vehicle model (including the same control policy), or each of them
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

        There is no communication among the vehicles, only (perfect) observations of their own onboard sensors.

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

        try:
            self._init(config, seed, render_mode)
        except Exception as e:
            print("\n///// Exception trapped in HighwayEnv.__init__: ", e)
            sys.exit(1)


    def _init(self,
                 config:        EnvContext,             #dict of config params
                 seed:          int             = None, #seed for PRNG
                 render_mode:   int             = None  #Ray rendering info, unused in this version
                ):

        """Initialize an object of this class.  Config options are documented in _set_initial_conditions()."""

        super().__init__()

        # Store the arguments
        #self.seed = seed #Ray 2.0.0 chokes on the seed() method if this is defined (it checks for this attribute also)
        #TODO: try calling self.seed() without storing it as an instance attribute
        if seed is None:
            seed = datetime.now().microsecond
        self.prng = HpPrng(seed = seed)
        self.render_mode = render_mode

        self._set_initial_conditions(config)
        if self.debug > 0:
            print("\n///// HighwayEnv init: config = ", config)

        # Create the roadway geometry
        self.roadway = Roadway(self.debug)

        # Get config data for the vehicles used in this scenario - the ego vehicle (where the agent lives) is index 0
        vc = self.vehicle_config
        v_data = vc["vehicles"]
        self._num_vehicles = len(v_data)

        # Instantiate model and controller objects for each vehicle, then use them to construct the vehicle object
        self.vehicles = []
        for i in range(self._num_vehicles):
            is_ego = i == 0 #need to identify the ego vehicle as the only one that will be learning
            spec = v_data[i]
            model = getattr(sys.modules[__name__], spec["model"])(
                                  max_jerk      = spec["max_jerk"],
                                  max_accel     = spec["max_accel"],
                                  length        = spec["length"],
                                  lc_duration   = spec["lc_duration"],
                                  time_step     = self.time_step_size)
            controller = getattr(sys.modules[__name__], spec["controller"])() #TODO - need to figure out args
            v = Vehicle(model, controller, self.prng, self.roadway, is_ego, self.time_step_size, self.debug)
            self.vehicles.append(v)
            controller.set_vehicle(v) #let the new controller know about the vehicle it is driving

        #TODO: do we need this?
        # Propagate the full vehicle list
        #for i in range(self._num_vehicles):
        #    self.vehicles[i].controller.set_vehicle_list(self.vehicles)







        #
        #..........Define the observation space TODO - rewrite this whole section
        #
        # A key portion of the obs space is a representation of spatial zones around the ego vehicle. These zones
        # move with the vehicle, and are a schematic representation of the nearby roadway situation. That is, they
        # don't include any lane shape geometry, representing every lane as straight, and physically adjacent to its
        # next lane. This is possible because we ASSUME that the vehicles are operating well inside their performance
        # limits, so that road geometry won't affect their ability to change lanes, accelerate or decelerate at any
        # desired rate. The zones are arranged as in the following diagram. Given this assumption, the observation
        # space can be defined in the parametric coordinate frame.
        #
        #              +----------+----------+----------+
        #              |  zone 7  |  zone 4  |  zone 1  |
        #   +----------+----------+----------+----------+
        #   |  zone 8  |  ego veh |  zone 5  |  zone 2  | >>>>> direction of travel
        #   +----------+----------+----------+----------+
        #              |  zone 9  |  zone 6  |  zone 3  |
        #              +----------+----------+----------+
        #
        # All zones are the same size, with their width equal to the lane width. Zone length is nominally 2 sec of
        # travel distance (at the posted speed limit). The ego vehicle is always centered in its zone, longitudinally.
        # If a neighbor vehicle is in the process of changing lanes, it will be observed to be in both adjacent zones through
        # the full lane change process. However, since the grid is centered on the ego vehicle, that vehicle will be
        # handled specially when it comes to lane change maneuvers, looking into either zone 7 or zone 9 concerning a
        # possible crash.
        #
        # Each zone will provide the following set of information:
        #   Is it drivable? E.g. is there a lane in that relative position
        #   Is the zone reachable from ego lane? I.e. lateral motion is possible and legal along the full length of the zone.
        #   Is there a neighbor vehicle in the zone? No more than one vehicle will be allowed in any given zone.
        #   Occupant's P location within the zone, if occupant exists ((P - Prear) / zone length), in [0, 1]
        #   Occupant's speed relative to ego vehicle, if occupant exists (delta-S / speed limit), in approx [-1.2, 1.2]

        # Gymnasium requires a member variable named observation_space. Since we are dealing with world scale values here, we
        # will need a wrapper to scale the observations for NN input. That wrapper will also need to use self.observation_space.
        # So here we must anticipate that scaling and leave the limits open enough to accommodate it.
        lower_obs = np.zeros((ObsVec.OBS_SIZE)) #most values are 0, so only the others are explicitly set below
        lower_obs[ObsVec.LC_CMD]              = LaneChange.CHANGE_LEFT
        lower_obs[ObsVec.LC_CMD_PREV]         = LaneChange.CHANGE_LEFT

        upper_obs = np.ones(ObsVec.OBS_SIZE) #most values are 1
        upper_obs[ObsVec.EGO_DES_SPEED]       = Constants.MAX_SPEED
        upper_obs[ObsVec.EGO_DES_SPEED_PREV]  = Constants.MAX_SPEED
        upper_obs[ObsVec.LC_CMD]              = LaneChange.CHANGE_RIGHT
        upper_obs[ObsVec.LC_CMD_PREV]         = LaneChange.CHANGE_RIGHT
        upper_obs[ObsVec.STEPS_SINCE_LN_CHG]  = Constants.MAX_STEPS_SINCE_LC
        upper_obs[ObsVec.EGO_SPEED]           = Constants.MAX_SPEED
        upper_obs[ObsVec.EGO_SPEED_PREV]      = Constants.MAX_SPEED
        upper_obs[ObsVec.FWD_DIST]            = Constants.REFERENCE_DIST
        upper_obs[ObsVec.FWD_SPEED]           = Constants.MAX_SPEED

        self.observation_space = Box(low = lower_obs, high = upper_obs, dtype = float)
        if self.debug == 2:
            print("///// observation_space = ", self.observation_space)

        self.all_obs = np.zeros(self._num_vehicles, ObsVec.OBS_SIZE) #one row for each vehicle
        self._verify_obs_limits("init after space defined") #TODO: move this to ???

        #
        #..........Define the action space
        #

        # Specify these for what the NN will deliver, not world scale
        lower_action = np.array([-1.0, -1.0])
        upper_action = np.array([ 1.0,  1.0])
        self.action_space = Box(low=lower_action, high = upper_action, dtype = float)
        if self.debug == 2:
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
        print("///// Initializing env environment ID {}".format(self.rollout_id))

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

        raise NotImplementedError("HighwayB.set_task() has been deprecated.") #TODO - get rid of this & getter when sure no longer needed


    def get_task(self) -> int:
        """Returns the environment difficulty level currently in use."""

        raise NotImplementedError("HighwayB.get_task() has been deprecated.")


    def reset(self, *,
              seed:         int             = None,    #reserved for a future version of gym.Environment
              options:      dict            = None
             ) -> Tuple[np.array, dict]:

        """Wrapper around the real reset method to trap for unhandled exceptions."""

        try:
            return self._reset(seed = seed, options = options)
        except Exception as e:
            print("\n///// Exception trapped in HighwayEnv.reset: ", e)
            sys.exit(1)


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

        if self.debug > 0:
            print("\n///// Entering reset")

        # We need the following line to seed self.np_random
        #super().reset(seed=seed) #apparently gym 0.26.1 doesn't implement this method in base class!
        #self.seed = seed #okay to pass it to the parent class, but don't store a local member copy!

        # options may be a dict that can specify additional configurations - unique to each particular env
        if options is not None and len(options) > 0:
            print("\n///// HighwayEnv.reset: incoming options is: ", options)
            raise ValueError("reset() called with options, but options are not used in this environment.")

        #
        #..........Set the initial conditions for each vehicle, depending on the scenario config
        #

        # Clear any lingering observations from the previous episode
        self.all_obs = np.zeros(self._num_vehicles, ObsVec.OBS_SIZE)

        # Solo bot vehicle that runs each lane in sequence at its speed limit (useful for inference only)
        if self.scenario >= 90:
            if self.scenario - 90 >= self.roadway.NUM_LANES:
                raise ValueError("///// Attempting to reset to unknown scenario {}".format(self.scenario))

            for i in range(self._num_vehicles):
                self.vehicles[i].active = False

            lane_id = self.scenario - 90
            self.vehicles[1].reset(lane_id, init_ddt = 0.0, init_speed = self.roadway.lanes[lane_id].segments[0][5])

        # No starting configuration specified - randomize everything
        else:
            for i in range(self._num_vehicles):
                space_found = False
                while not space_found:
                    lane_id = int(self.prng.random() * self.roadway.NUM_LANES)
                    lane_begin = self.roadway.get_lane_start_p(lane_id)
                    ddt = self.prng.random() * (self.roadway.get_total_lane_length(lane_id) - 50.0)
                    loc = ddt + lane_begin
                    space_found = self._verify_safe_location(i, lane_id, loc)
                speed = self.prng.random() * Constants.MAX_SPEED
                self.vehicles[i].reset(lane_id = lane_id, init_ddt = ddt, init_speed = speed)

        #
        #..........Gather the observations from the appropriate vehicles & wrap up
        #

        # We must do this after all vehicles have been initialized, otherwise obs from the vehicles placed first won't
        # include sensing of vehicle placed later.

        # Get the obs from each vehicle
        for i in range(self._num_vehicles):
            self.full_obs[i, :] = self.vehicles[i].model.get_obs_vector(i, self.vehicles)

        # Other persistent data
        self.steps_since_reset = 0
        self.episode_count += 1

        if self.debug > 0:
            print("///// End of reset().")
        return self.full_obs[0, :], {} #only return the row for the ego vehicle


    def step(self,
                cmd     : list      #list of floats; 0 = speed command, 1 = desired lane, scaled
            ) -> Tuple[np.array, float, bool, bool, Dict]:

        """Wrapper around the real step method to trap unhandled exceptions."""

        try:
            return self._step(cmd)
        except Exception as e:
            print("\n///// Exception trapped in HighwayEnv.step: ", e)


    def _step(self,
                cmd     : list      #list of floats; 0 = speed command, 1 = desired lane, scaled
            ) -> Tuple[np.array, float, bool, bool, Dict]:

        """Executes a single time step of the environment.  Determines how the input commands (actions) will alter the
            simulated world and returns the resulting observations.

            Return is array of new observations, new reward, done flag, truncated flag, and a dict of additional info.

            CAUTION: the returned observation vector is at actual world scale and needs to be
                     preprocessed before going into a NN!

            The process is:
                - gather control commands based on existing observations (for the ego vehicle, these come in as
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
        return_info = {"reason": "Unknown"}

        #
        #..........Update states of all vehicles
        #

        # Unscale the ego action inputs (both actions are in [-1, 1])
        ego_action = []*2
        ego_action[0] = (cmd[0] + 1.0)/2.0 * Constants.MAX_SPEED
        ego_action[1] = int(math.floor(cmd[1] + 0.5))
        if self.steps_since_reset < 2: #force it to stay in lane for first time step
            ego_action[1] = 0.0

        # Loop through all active vehicles. Note that the ego vehicle is always at index 0.
        vehicle_actions = []*self._num_vehicles
        action = ego_action
        for i in range(self._num_vehicles):
            if not self.vehicles[i].active:
                continue

            # Exercise the control algo to generate the next action commands.
            if i > 0:
                action = self.vehicles[i].controller.step(self.full_obs[i, :])

            # Store the actions for future reference
            vehicle_actions[i] = action

            # Apply the appropriate dynamics model to each vehicle in the scenario to get its new state.
            new_speed, new_p, new_lane, reason = self.vehicles[i].advance_vehicle_spd(action[0], action[1]) #TODO: do we need these return values?
            if self.debug > 1:
                print("      Vehicle {} advanced with new_speed_cmd = {:.2f}. new_speed = {:.2f}, new_p = {:.2f}, new_lane = {}"
                        .format(i, action[0], new_speed, new_p, new_lane))

            # If the ego vehicle has reached one of its target destinations, it is a successful episode
            if i == 0:
                if self.vehicles[0].lane_id in [1, 2]  and  new_p >= TARGET_P:
                    done = True
                    return_info["reason"] = "SUCCESS - end of scenario!"
                    #print("/////+ step: {} step {}, success - completed the track".format(self.rollout_id, self.total_steps))

                # Else if it ran off the road or stopped, then end the episode in failure
                elif self.vehicles[0].off_road  or  self.vehicles[0].stopped:
                    done = True
                    return_info["reason"] = reason

        # Check that none of the vehicles has crashed into another, accounting for a lane change in progress taking up both lanes.
        crash = self._check_for_collisions()
        if crash:
            done = True
            return_info["reason"] = "Two (or more) vehicles crashed."
            #print("/////+ step: {} step {}, crash!".format(self.rollout_id, self.total_steps))








        # Determine the reward resulting from this time step's action
        reward, expl = self._get_reward(done, crash, self.vehicles[0].off_road, self.vehicles[0].stopped)
        return_info["reward_detail"] = expl
        #print("/////+ step: {} step {}, returning reward of {}, {}".format(self.rollout_id, self.total_steps, reward, expl))

        # Verify that the obs are within design limits
        self._verify_obs_limits("step after reward calc")

        if self.debug > 0:
            print("///// step complete. Returning obs = ")
            print(self.all_obs)
            print("      reward = ", reward, ", done = ", done)
            print("      final vehicles array =")
            for i, v in enumerate(self.vehicles):
                v.print(i)
            print("      reason = {}".format(return_info["reason"]))
            print("      reward_detail = {}\n".format(return_info["reward_detail"]))

        truncated = False #indicates if the episode ended due to step/time limit rather than success or failure
        return self.obs, reward, done, truncated, return_info


    def get_stopper(self):
        """Returns the stopper object."""
        return self.stopper


    def get_burn_in_iters(self):
        """Returns the number of burn-in iterations configured."""
        return self.burn_in_iters


    def get_total_steps(self):
        """Returns the total number of time steps executed so far."""
        return self.total_steps


    def get_vehicle_dist_downtrack(self,
                                   vehicle_id   : int   #index of the vehicle of interest
                                  ) -> float:
        """Returns the indicated vehicle's distance downtrack from its lane beginning, in m.
            Used for inference, which needs real DDT, not X location.
        """

        assert 0 <= vehicle_id < self._num_vehicles, \
                "///// HighwayEnv.get_vehicle_dist_downtrack: illegal vehicle_id entered: {}".format(vehicle_id)

        ddt = self.vehicles[vehicle_id].p
        lane_id = self.vehicles[vehicle_id].lane_id
        if lane_id == 2:
            ddt -= self.roadway.get_lane_start_p(lane_id)
        return ddt


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

        self.debug = 0 #level of debug printing (0=none (default), 1=moderate, 2=full details)
        try:
            db = config["debug"]
        except KeyError as e:
            db = None
        if db is not None  and  db != ""  and  0 <= int(db) <= 2:
            self.debug = int(db)

        self.training = False #is the environment being used in a training job? (affects scaling of observations)
        try:
            tr = config["training"]
            if tr == "True":
                self.training = True
        except KeyError as e:
            pass

        self.verify_obs = False #verify that the obs vector values are all within the specified limits (runs slower)?
        try:
            vo = config["verify_obs"]
            self.verify_obs = vo
        except KeyError as e:
            pass

        self.init_ego_lane = None #lane ID that the agent vehicle begins in (if not specified, then randomized)
        try:
            el = config["init_ego_lane"]
            if 0 <= el <= Roadway.NUM_LANES:
                self.init_ego_lane = el
        except KeyError as e:
            pass

        self.init_ego_speed = None #initial speed of the agent vehicle, m/s (if not specified, then randomized)
        try:
            es = config["init_ego_speed"]
            if 0 <= es <= Constants.MAX_SPEED:
                self.init_ego_speed = es
        except KeyError as e:
            pass

        self.init_ego_dist = None #initial downtrack location of the agent vehicle from lane begin, m (if not specified, then randomized)
        try:
            ed = config["init_ego_dist"]
            if 0 <= ed < Constants.SCENARIO_LENGTH:
                self.init_ego_dist = ed
        except KeyError as e:
            pass

        self.vehicle_config = {} #dict of configs specific to the fleet of vehicles
        vcf = None
        try:
            vcf = config["vehicle_file"]
            stream = open(vcf, 'r')
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


    def _update_obs_zones(self):
        """Updates the observation vector data for each of the roadway zones, based on ego state and current neighbor vehicle states.

            CAUTION: requires that each zone is represented the same in the obs vector and all zones are contiguous there.
        """

        # Determine offsets in the obs vector for zone columns and rows
        base = self.Z1_DRIVEABLE
        num_zone_fields = self.Z2_DRIVEABLE - base

        # Clear all zone info from previous time step
        for z in range(9):
            self.obs[base + num_zone_fields*z + 0] = 0.0 #drivable
            self.obs[base + num_zone_fields*z + 1] = 0.0 #reachable
            self.obs[base + num_zone_fields*z + 2] = 0.0 #occupied
            self.obs[base + num_zone_fields*z + 3] = 0.0 #p
            self.obs[base + num_zone_fields*z + 4] = 0.0 #speed
        self.obs[self.NEIGHBOR_IN_EGO_ZONE] = 0.0

       # Get the current roadway geometry
        ego_lane_id = self.vehicles[0].lane_id
        ego_p = self.vehicles[0].p
        if self.debug > 1:
            print("///// Entering update_obs_zones: ego_lane_id = {}, ego_p = {:.1f}, base = {}"
                  .format(ego_lane_id, ego_p, base))

        # Determine pavement existence and reachability in each zone
        # CAUTION: this block is dependent on the specific roadway geometry for this experiment, and is not generalized
        for row in range(1, 4):
            zone_front = ((3 - row) + 0.5)*Constants.OBS_ZONE_LENGTH #distance downtrack from ego vehicle, m
            zone_rear = zone_front - Constants.OBS_ZONE_LENGTH
            zone_mid_p = ego_p + 0.5*(zone_front + zone_rear) #absolute coordinate in p-frame
            # Get updated roadway geometry; NB all distances returned are relative to current ego location
            ego_rem, lid, la, lb, l_rem, rid, ra, rb, r_rem = self.roadway.get_current_lane_geom(ego_lane_id, ego_p)
            if self.debug > 1:
                print("///// _update_obs_zones: row = {}, ego_p = {:.1f}, zone_front = {:.1f}, zone_rear = {:.1f}, zone_mid = {:.1f}, la = {:.1f}, lb = {:.1f}"
                        .format(row, ego_p, zone_front, zone_rear, zone_mid_p, la, lb))

            # Determine if there is pavement in the left-hand zone and it's reachable
            if lid >= 0: #the lane exists somewhere along the route

                # Determine if the left lane exists next to the middle of this zone
                start_p = self.roadway.get_lane_start_p(lid)
                if start_p <= zone_mid_p <= start_p + self.roadway.get_total_lane_length(lid):
                    l_zone = 3*(row - 1) + 1
                    l_offset = base + (l_zone - 1)*num_zone_fields
                    self.obs[l_offset + 0] = 1.0 #drivable
                    if la <= zone_front  and  lb >= zone_rear:
                        self.obs[l_offset + 1] = 1.0 #reachable

            # Determine if there is pavement in the right-hand zone and it's reachable
            if rid >= 0: #there's a lane to the right somewhere along this route

                # Determine if the right lane exists next to the middle of this zone
                start_p = self.roadway.get_lane_start_p(rid)
                if start_p <= zone_mid_p <= start_p + self.roadway.get_total_lane_length(rid):
                    r_zone = 3*(row - 1) + 3
                    r_offset = base + (r_zone - 1)*num_zone_fields
                    self.obs[r_offset + 0] = 1.0 #drivable
                    if ra <= zone_front  and  rb >= zone_rear:
                        self.obs[r_offset + 1] = 1.0 #reachable

        # We know there's a lane in the center, but not how far it extends in either direction so look at each zone in this column.
        # Note that the "reachable" determination here is different from those above. Since the above represent adjacent lanes, they
        # can afford to be more liberal in that they say reachable = True if any part of the adjacent pavement borders any part of the
        # zone in question (i.e. the entire zone edge does not need to touch adjacent pavement). Whereas, for the ego's own lane, it
        # is more important to know as soon as the pavement disappears in _any_ part of its forward zone, so it can prepare to change
        # lanes.
        for row in range(1, 5):
            zone = 3*(row - 1) + 2
            if row == 3: #ego zone
                continue
            elif row == 4:
                zone = 8
            offset = base + (zone - 1)*num_zone_fields
            zone_front = ((3 - row) + 0.5)*Constants.OBS_ZONE_LENGTH #distance downtrack from ego vehicle, m
            zone_rear = zone_front - Constants.OBS_ZONE_LENGTH
            if ego_rem >= zone_front: #don't worry about lane existence behind ego vehicle; assume it's there
                self.obs[offset + 0] = 1.0 #drivable
                self.obs[offset + 1] = 1.0 #reachable is guaranteed if it is driveable, since it's the same lane

        # Loop through the neighbor vehicles
        for neighbor_idx in range(1, self._num_vehicles):
            nv = self.vehicles[neighbor_idx]

            # Find which zone column it is in (relative lane), if any (could be 2 lanes away) (ego is in column 1, lanes are 0-indexed, left-to-right)
            column = nv.lane_id - ego_lane_id + 1
            if self.debug > 1:
                print("///// update_obs_zones: considering neighbor {} in column {}".format(neighbor_idx, column))

            # Find which zone row it is in, if any (could be too far away)
            row = 0
            dist_ahead_of_ego = nv.p - ego_p
            if dist_ahead_of_ego > 2.5 * Constants.OBS_ZONE_LENGTH: #way out front somewhere
                row = 0
            elif dist_ahead_of_ego > 1.5 * Constants.OBS_ZONE_LENGTH:
                row = 1
            elif dist_ahead_of_ego > 0.5 * Constants.OBS_ZONE_LENGTH:
                row = 2
            elif dist_ahead_of_ego > -0.5 * Constants.OBS_ZONE_LENGTH:
                row = 3
            elif dist_ahead_of_ego > -1.5 * Constants.OBS_ZONE_LENGTH:
                row = 4
            # Else too far behind to consider, so allow row value of 0 for this case also

            # If the neighbor is too far away, no further consideration needed
            if column < 0  or  column > 2  or  row == 0:
                if self.debug > 1:
                    print("///// update_obs_zones: found neighbor {} too far away, with column {}, row {}, p = {:.1f}, dist_ahead_of_ego = {:.1f}"
                          .format(neighbor_idx, column, row, nv.p, dist_ahead_of_ego))
                continue

            # Neighbor is within our obs zone grid.
            # If it is also within the ego zone then
            if column == 1  and  row == 3:

                # Set the flag - nothing else needed
                self.obs[self.NEIGHBOR_IN_EGO_ZONE] = 1.0
                if dist_ahead_of_ego < 0.0:
                    self.obs[self.NEIGHBOR_IN_EGO_ZONE] = -1.0

                if self.debug > 1:
                    print("///// update_obs_zones: neighbor {} is in ego zone!".format(neighbor_idx))

            # Else get its offset into the obs vector
            else:
                zone = 0
                if row < 4:
                    zone = 3*(row - 1) + column + 1
                else:
                    if column == 1:
                        zone = 8
                    else:
                        if self.debug > 1:
                            print("///// update_obs_zones: found a neighbor beside zone 8: column = {}, dist_ahead_of_ego = {:.1f}"
                                  .format(column, dist_ahead_of_ego))
                        continue

                offset = base + (zone - 1)*num_zone_fields
                if self.debug > 1:
                    print("///// update_obs_zones: neighbor offset = {} for zone {}".format(offset, zone))

                # Since we've identified a neighbor vehicle in this zone, flag it as occupied
                self.obs[offset + 2] = 1.0

                # Set the neighbor's relative location within the zone
                zone_rear_p = ego_p + ((2.0 - row) + 0.5)*Constants.OBS_ZONE_LENGTH
                rel_p = (nv.p - zone_rear_p) / Constants.OBS_ZONE_LENGTH
                self.obs[offset + 3] = rel_p

                # Set the neighbor's relative speed
                self.obs[offset + 4] = (nv.cur_speed - self.vehicles[0].cur_speed) / Constants.ROAD_SPEED_LIMIT

                if self.debug > 1:
                    print("///// update_obs_zones: neighbor {} has column = {}, row = {}, zone = {}, zone_rear = {:.1f}, rel_p = {:.2f}, ego speed = {:.1f}"
                          .format(neighbor_idx, column, row, zone, zone_rear_p, rel_p, self.vehicles[0].cur_speed))

        if self.debug > 1:
            print("///// update_obs_zones complete.")
            for zone in range(1, 10):
                offset = base + (zone - 1)*num_zone_fields
                print("      Zone {}: drivable = {:.1f}, reachable = {:.1f}, occupied = {:.1f}, rel p = {:.2f}, rel speed = {:.2f}"
                      .format(zone, self.obs[offset+0], self.obs[offset+1], self.obs[offset+2], self.obs[offset+3], self.obs[offset+4]))


    def _verify_safe_location(self,
                              n         : int,  #neighbor ID
                              lane_id   : int,  #desired lane ID for the neighbor
                              p         : float,#desired P coordinate for the neighbor (m in paremetric frame)
                             ) -> bool:         #returns true if the indicated location is safe
        """Determines if the candidate location (lane & P coordinate) is a safe place to put a vehicle at the beginning of a scenario.
            It needs to be sufficiently far from any other neighbors whose starting locations have already been defined.
        """

        assert 0 <= lane_id < Constants.NUM_LANES, "///// Attempting to place neighbor {} in invalid lane {}".format(n, lane_id)
        start = self.roadway.get_lane_start_p(lane_id)
        assert start <= p < start + self.roadway.get_total_lane_length(lane_id), \
                "///// Attempting to place neighbor {} in lane {} at invalid p = {:.1f}".format(n, lane_id, p)

        safe = True

        # Loop through all active vehicles
        for o in range(self._num_vehicles):
            other = self.vehicles[o]
            if not other.active:
                continue

            # If the other vehicle is in candiate's lane then check if it is too close longitudinally. Note that if a neighbor has
            # not yet been placed, its lane ID is -1
            if other.lane_id == lane_id:
                if 0.0 <= p - other.p < 5.0*Constants.VEHICLE_LENGTH  or \
                   0.0 <= other.p - p < 3.0*Constants.VEHICLE_LENGTH:
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
        for i in range(self._num_vehicles - 1):
            va = self.vehicles[i]
            if not va.active:
                continue

            # Loop through the remaining active vehicles to get vehicle B
            for j in range(i + 1, self._num_vehicles):
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

                        # Mark it so only if it involves the ego vehicle or we are worried about all crashes
                        if i == 0  or  j == 0  or  not self.ignore_neighbor_crashes:
                            crash = True
                            if self.debug > 1:
                                print("      CRASH in same lane between vehicles {} and {} near {:.2f} m in lane {}"
                                        .format(i, j, va.p, va.lane_id))
                            break

                # Else if they are in adjacent lanes, then
                elif abs(va.lane_id - vb.lane_id) == 1:

                    # If either vehicle is changing lanes at the moment, then
                    if va.lane_change_status != "none"  or  vb.lane_change_status != "none":

                        # If the lane changer's target lane is occupied by the other vehicle, then
                        va_tgt = self.roadway.get_target_lane(va.lane_id, va.lane_change_status, va.p)
                        vb_tgt = self.roadway.get_target_lane(vb.lane_id, vb.lane_change_status, vb.p)
                        if va_tgt == vb.lane_id  or  vb_tgt == va.lane_id:

                            # If the two are within a vehicle length of each other, then it's a crash
                            if abs(va.p - vb.p) <= 0.5*(va.model.veh_length + vb.model.veh_length):

                                # Mark the involved vehicles as out of service
                                va.active = False
                                vb.active = False
                                va.crashed = True
                                vb.crashed = True

                                # Mark it so only if it involves the ego vehicle or we are worried about all crashes
                                if i == 0  or  j == 0  or  not self.ignore_neighbor_crashes:
                                    crash = True
                                    if self.debug > 1:
                                        print("      CRASH in adjacent lanes between vehicles {} and {} near {:.2f} m in lane {}"
                                                .format(i, j, vb.p, va.lane_id))
                                    break

            if crash: #the previous break stmts only break out of the inner loop, so we need to break again
                break

        if self.debug > 0:
            print("///// _check_for_collisions complete. Returning ", crash)
        return crash


    def _get_reward(self,
                    done    : bool,         #is this the final step in the episode?
                    crash   : bool,         #did one or more of the vehicles crash into each other?
                    off_road: bool,         #did the ego vehicle run off the road?
                    stopped : bool          #has the vehicle come to a standstill?
                   ):
        """Returns the reward for the current time step (float).  The reward should be in [-1, 1] for any situation."""

        if self.debug > 1:
            print("///// Entering _get_reward rollout {}, step {}. done = {}, crash = {}, off_road = {}"
                    .format(self.rollout_id, self.total_steps, done, crash, off_road))
        reward = 0.0
        explanation = ""

        # If the episode is done then
        if done:

            # If there was a multi-car crash or off-roading (single-car crash) then set a penalty, larger for multi-car crash
            if crash:
                reward = -15.0
                explanation = "Crashed into a vehicle. "

            elif off_road:
                reward = -10.0
                # Assign a different value if agent is in lane 2 so we can see in the logs which episodes are in this lane
                if self.vehicles[0].lane_id == 2:
                    reward = -10.0
                explanation = "Ran off road. "

            # Else if the vehicle just stopped in the middle of the road then
            elif stopped:

                # Subtract a penalty for no movement (needs to be as severe as off-road)
                reward = -12.0
                explanation = "Vehicle stopped. "

            # Else (episode ended successfully)
            else:

                # If we are allowed to reward the completion bonus then add amount inversely proportional
                # to the length of the episode.
                if self.difficulty_level == 0:
                    reward = 10.0
                    explanation = "Successful episode!"

                else:
                    if self.reward_for_completion:
                        dist = abs(self.steps_since_reset - 130)
                        reward = min(max(10.0 - 0.2*dist, 0.0), 10.0)
                        explanation = "Successful episode! {} steps".format(self.steps_since_reset)
                    else:
                        explanation = "Completed episode, but no bonus due to rule violation."

        # Else, episode still underway
        else:

            # Reward for staying alive
            INITIAL_BONUS = 0.01
            bonus = INITIAL_BONUS
            tune_steps = 1
            reward += bonus

            # Small penalty for widely varying lane commands
            cmd_diff = abs(self.obs[ObsVec.LC_CMD] - self.obs[ObsVec.LC_CMD_PREV])
            penalty = 0.1 * cmd_diff * cmd_diff
            reward -= penalty
            if penalty > 0.0001:
                explanation += "Ln cmd pen {:.4f}. ".format(penalty)

            # Small penalty for widely varying speed commands
            if self.difficulty_level > 0:
                cmd_diff = abs(self.obs[ObsVec.EGO_DES_SPEED] - self.obs[ObsVec.EGO_DES_SPEED_PREV]) / Constants.MAX_SPEED
                penalty = 0.4 * cmd_diff * cmd_diff
                reward -= penalty
                if penalty > 0.0001:
                    explanation += "Spd cmd pen {:.4f}. ".format(penalty)

            # Penalty for deviating from roadway speed limit
            speed_mult = 0.3
            if self.difficulty_level == 1  or  self.difficulty_level == 2:
                speed_mult *= 2.0

            norm_speed = self.obs[ObsVec.EGO_SPEED] / Constants.ROAD_SPEED_LIMIT #1.0 = speed limit
            diff = abs(norm_speed - 1.0)
            penalty = 0.0
            if diff > 0.02:
                penalty = speed_mult*(diff - 0.02)
                explanation += "spd pen {:.4f}. ".format(penalty)
            reward -= penalty

            # If a lane change was initiated, apply a penalty depending on how soon after the previous lane change
            if self.lane_change_count == 1:
                penalty = 0.05 + 0.005*(Constants.MAX_STEPS_SINCE_LC - self.obs[ObsVec.STEPS_SINCE_LN_CHG])
                reward -= penalty
                explanation += "Ln chg pen {:.4f}. ".format(penalty)

        if self.debug > 0:
            print("///// reward returning {:.4f} due to crash = {}, off_road = {}, stopped = {}"
                    .format(reward, crash, off_road, stopped))

        return reward, explanation


    def _verify_obs_limits(self,
                           tag      : str = ""  #optional explanation of where in the code this was called
                          ):
        """Checks that each element of the observation vector is within the limits of the observation space."""

        if not self.verify_obs:
            return

        lo = self.observation_space.low
        hi = self.observation_space.high

        try:
            for i in range(ObsVec.OBS_SIZE):
                assert lo[i] <= self.obs[i] <= hi[i], "\n///// obs[{}] value ({}) is outside bounds {} and {}" \
                                                        .format(i, self.obs[i], lo[i], hi[i])

        except AssertionError as e:
            print(e)
            print("///// Full obs vector content at: {}:".format(tag))
            for j in range(ObsVec.OBS_SIZE):
                print("      {:2d}: {}".format(j, self.obs[j]))
