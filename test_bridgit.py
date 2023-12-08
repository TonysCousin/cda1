import sys
from obs_vec import ObsVec
from hp_prng import HpPrng
from roadway_b import Roadway
from vehicle import Vehicle
from bridgit_model import BridgitModel
from bridgit_guidance import BridgitGuidance
from bot_type1_model import BotType1Model
from bot_type1a_guidance import BotType1aGuidance
from target_destination import TargetDestination


def main(argv):
    """Unit tester for the BridgitModel."""

    prng = HpPrng(17)
    debug = 0

    roadway = Roadway(debug)
    t_targets = []
    t_targets.append(TargetDestination(roadway, 1, 2900.0))
    t_targets.append(TargetDestination(roadway, 2, 2900.0))

    # Instantiate model and controller objects for each vehicle, then use them to construct the vehicle objects
    num_vehicles = 11
    vehicles = []
    for i in range(num_vehicles):
        is_ego = i == 0 #need to identify the ego vehicle as the only one that will be learning
        v = None
        model = None
        controller = None
        try:
            if is_ego:
                print("Building ego vehicle")
                model = BridgitModel(roadway)
                controller = BridgitGuidance(prng, roadway, t_targets)
            else:
                print("Building a simple bot vehicle.")
                model = BotType1Model(roadway, length = 5.0)
                controller = BotType1aGuidance(prng, roadway, t_targets)
            v = Vehicle(model, controller, prng, roadway, learning = is_ego, debug = debug)
        except AttributeError as e:
            print("///// Problem with config for vehicle ", i, " model or controller: ", e)
            raise e
        except Exception as e:
            print("///// Problem creating vehicle model, controller, or the vehicle itself: ", e)
            print("Exception type is ", type(e))
            raise e

        vehicles.append(v)
        controller.set_vehicle(v) #let the new controller know about the vehicle it is driving

    # Case set A (neighbor vehicle config)
    vehicles[1].lane_id = 3
    vehicles[1].p       = 842.0
    vehicles[2].lane_id = 3
    vehicles[2].p       = 888.0
    vehicles[3].lane_id = 5
    vehicles[3].p       = 909.0
    vehicles[4].lane_id = 4
    vehicles[4].p       = 966.0
    vehicles[5].lane_id = 1
    vehicles[5].p       = 791.0
    vehicles[6].lane_id = 1
    vehicles[6].p       = 881.0
    vehicles[7].lane_id = 3
    vehicles[7].p       = 799.0
    vehicles[7].model.veh_length = 2.1
    vehicles[8].lane_id = 3
    vehicles[8].p       = 788.0
    vehicles[9].lane_id = 2
    vehicles[9].p       = 813.0
    vehicles[10].lane_id= 2
    vehicles[10].p      = 839.0
    vehicles[10].model.veh_length = 18.0
    for vi in range(1, len(vehicles)):
        vehicles[vi].cur_speed = 28.34
    vehicles[4].cur_speed = 9.02
    vehicles[5].cur_speed = 33.34
    vehicles[6].cur_speed = 33.37

    # Need an empty vector to pass in - normally this will hold some historical elements that need to be saved
    obs = [0.0]*ObsVec.OBS_SIZE

    # Case A1
    vehicles[0].lane_id = 3
    vehicles[0].p       = 805.0
    vehicles[0].cur_speed = 25.5
    actions = [0.75, 0.8]
    obs = vehicles[0].model.get_obs_vector(0, vehicles, actions, obs)
    print_obs(obs, "Case A1 - ego lane = {}, p = {:.1f}".format(vehicles[0].lane_id, vehicles[0].p))

    # Case A2
    vehicles[0].lane_id = 1
    vehicles[0].p       = 812.2
    actions = [0.75, 0.8]
    obs = vehicles[0].model.get_obs_vector(0, vehicles, actions, obs)
    print_obs(obs, "Case A2 - ego lane = {}, p = {:.1f}".format(vehicles[0].lane_id, vehicles[0].p))

    # Case A3
    vehicles[0].lane_id = 2
    vehicles[0].p       = 511.6
    actions = [0.75, 0.8]
    obs = vehicles[0].model.get_obs_vector(0, vehicles, actions, obs)
    print_obs(obs, "Case A3 - ego lane = {}, p = {:.1f}".format(vehicles[0].lane_id, vehicles[0].p))

    # Case A4
    vehicles[0].lane_id = 1
    vehicles[0].p = 2406.9
    actions = [0.75, 0.8]
    obs = vehicles[0].model.get_obs_vector(0, vehicles, actions, obs)
    print_obs(obs, "Case A4 - ego lane = {}, p = {:.1f}".format(vehicles[0].lane_id, vehicles[0].p))

    # Case A5
    vehicles[0].lane_id = 5
    vehicles[0].p       = 1260.0
    actions = [0.75, 0.8]
    obs = vehicles[0].model.get_obs_vector(0, vehicles, actions, obs)
    print_obs(obs, "Case A5 - ego lane = {}, p = {:.1f}".format(vehicles[0].lane_id, vehicles[0].p))

    # Case A6
    vehicles[0].lane_id = 4
    vehicles[0].p       = 906.0
    actions = [0.75, 0.8]
    obs = vehicles[0].model.get_obs_vector(0, vehicles, actions, obs)
    print_obs(obs, "Case A6 - ego lane = {}, p = {:.1f}".format(vehicles[0].lane_id, vehicles[0].p))

    # Case A7
    vehicles[0].lane_id = 0
    vehicles[0].p       = 2459.7
    actions = [0.75, 0.8]
    obs = vehicles[0].model.get_obs_vector(0, vehicles, actions, obs)
    print_obs(obs, "Case A7 - ego lane = {}, p = {:.1f}".format(vehicles[0].lane_id, vehicles[0].p))

    # Case A8
    vehicles[0].lane_id = 2
    vehicles[0].p       = 896.0
    vehicles[0].cur_speed = 31.9
    actions = [0.75, 0.8]
    obs = vehicles[0].model.get_obs_vector(0, vehicles, actions, obs)
    print_obs(obs, "Case A8 - ego lane = {}, p = {:.1f}".format(vehicles[0].lane_id, vehicles[0].p))


def print_obs(obs, label):

    print("\nTest case: {}\n".format(label))

    # Visual layers
    print("            Pavement (SL)                       Vehicles (spd)\n")
    for row in range(ObsVec.NUM_ROWS):
        z = ObsVec.NUM_ROWS - 1 - row
        c0 = z
        c1 = z +   ObsVec.NUM_ROWS
        c2 = z + 2*ObsVec.NUM_ROWS
        c3 = z + 3*ObsVec.NUM_ROWS
        c4 = z + 4*ObsVec.NUM_ROWS

        p_row = ["  .  ", "  .  ", "  .  ", "  .  ", "  .  "]
        v_row = ["  .  ", "  .  ", "  .  ", "  .  ", "  .  "]
        if obs[ObsVec.BASE_PVMT_TYPE + c0] >= 0:
            p_row[0] = "{:5.2f}".format(obs[ObsVec.BASE_SPD_LIMIT + c0])
        if obs[ObsVec.BASE_PVMT_TYPE + c1] >= 0:
            p_row[1] = "{:5.2f}".format(obs[ObsVec.BASE_SPD_LIMIT + c1])
        if obs[ObsVec.BASE_PVMT_TYPE + c2] >= 0:
            p_row[2] = "{:5.2f}".format(obs[ObsVec.BASE_SPD_LIMIT + c2])
        if obs[ObsVec.BASE_PVMT_TYPE + c3] >= 0:
            p_row[3] = "{:5.2f}".format(obs[ObsVec.BASE_SPD_LIMIT + c3])
        if obs[ObsVec.BASE_PVMT_TYPE + c4] >= 0:
            p_row[4] = "{:5.2f}".format(obs[ObsVec.BASE_SPD_LIMIT + c4])

        if obs[ObsVec.BASE_OCCUPANCY + c0] > 0:
            v_row[0] = "{:5.2f}".format(obs[ObsVec.BASE_REL_SPEED + c0])
        if obs[ObsVec.BASE_OCCUPANCY + c1] > 0:
            v_row[1] = "{:5.2f}".format(obs[ObsVec.BASE_REL_SPEED + c1])
        if obs[ObsVec.BASE_OCCUPANCY + c2] > 0:
            v_row[2] = "{:5.2f}".format(obs[ObsVec.BASE_REL_SPEED + c2])
        if obs[ObsVec.BASE_OCCUPANCY + c3] > 0:
            v_row[3] = "{:5.2f}".format(obs[ObsVec.BASE_REL_SPEED + c3])
        if obs[ObsVec.BASE_OCCUPANCY + c4] > 0:
            v_row[4] = "{:5.2f}".format(obs[ObsVec.BASE_REL_SPEED + c4])

        if row == 20:
            p_row[2] = "*Ego*"
            v_row[2] = "*Ego*"

        print("{:2d} [{} {} {} {} {}]     [{} {} {} {} {}]"
              .format(z, p_row[0], p_row[1], p_row[2], p_row[3], p_row[4], v_row[0], v_row[1], v_row[2], v_row[3], v_row[4]))

    # Numerical details
    print("\n\n")
    for row in range(ObsVec.NUM_ROWS):
        z = ObsVec.NUM_ROWS - 1 - row
        c0 = z
        c1 = z +   ObsVec.NUM_ROWS
        c2 = z + 2*ObsVec.NUM_ROWS
        c3 = z + 3*ObsVec.NUM_ROWS
        c4 = z + 4*ObsVec.NUM_ROWS
        bp = ObsVec.BASE_PVMT_TYPE
        bl = ObsVec.BASE_SPD_LIMIT
        bo = ObsVec.BASE_OCCUPANCY
        bs = ObsVec.BASE_REL_SPEED

        br = 0
        zm5 = z - 5
        if 2 < zm5 < 10:
            br = 1
        elif 9 < zm5 < 20:
            br = 2
        l_bdry = ObsVec.BASE_LEFT_CTR_BDRY + br
        r_bdry = ObsVec.BASE_RIGHT_CTR_BDRY + br

        if row < 20:
            print("{:2d} Type: {:6.3f}   {:6.3f}   {:6.3f}   {:6.3f}   {:6.3f}".format(z, obs[c0+bp], obs[c1+bp], obs[c2+bp], obs[c3+bp], obs[c4+bp]))
            print("     SL: {:6.3f}   {:6.3f}   {:6.3f}   {:6.3f}   {:6.3f}".format(obs[c0+bl], obs[c1+bl], obs[c2+bl], obs[c3+bl], obs[c4+bl]))
            print("    Occ: {:6.3f}   {:6.3f}   {:6.3f}   {:6.3f}   {:6.3f}".format(obs[c0+bo], obs[c1+bo], obs[c2+bo], obs[c3+bo], obs[c4+bo]))
            print("    Spd: {:6.3f}   {:6.3f}   {:6.3f}   {:6.3f}   {:6.3f}".format(obs[c0+bs], obs[c1+bs], obs[c2+bs], obs[c3+bs], obs[c4+bs]))
            print("   LBdr:                   {:6.3f}".format(obs[l_bdry]))
            print("   RBdr:                   {:6.3f}".format(obs[r_bdry]))

        elif row == 20:
            print("{:2d} Type: {:6.3f}   {:6.3f}   *Ego *   {:6.3f}   {:6.3f}".format(z, obs[c0+bp], obs[c1+bp], obs[c3+bp], obs[c4+bp]))
            print("     SL: {:6.3f}   {:6.3f}   *Ego *   {:6.3f}   {:6.3f}".format(obs[c0+bl], obs[c1+bl], obs[c3+bl], obs[c4+bl]))
            print("    Occ: {:6.3f}   {:6.3f}   *Ego *   {:6.3f}   {:6.3f}".format(obs[c0+bo], obs[c1+bo], obs[c3+bo], obs[c4+bo]))
            print("    Spd: {:6.3f}   {:6.3f}   *Ego *   {:6.3f}   {:6.3f}".format(obs[c0+bs], obs[c1+bs], obs[c3+bs], obs[c4+bs]))
            print("   LBdr:                   {:6.3f}".format(obs[l_bdry]))
            print("   RBdr:                   {:6.3f}".format(obs[r_bdry]))

        else:
            print("{:2d} Type: {:6.3f}   {:6.3f}   {:6.3f}   {:6.3f}   {:6.3f}".format(z, obs[c0+bp], obs[c1+bp], obs[c2+bp], obs[c3+bp], obs[c4+0]))
            print("     SL: {:6.3f}   {:6.3f}   {:6.3f}   {:6.3f}   {:6.3f}".format(obs[c0+bl], obs[c1+bl], obs[c2+bl], obs[c3+bl], obs[c4+bl]))
            print("    Occ: {:6.3f}   {:6.3f}   {:6.3f}   {:6.3f}   {:6.3f}".format(obs[c0+bo], obs[c1+bo], obs[c2+bo], obs[c3+bo], obs[c4+bo]))
            print("    Spd: {:6.3f}   {:6.3f}   {:6.3f}   {:6.3f}   {:6.3f}".format(obs[c0+bs], obs[c1+bs], obs[c2+bs], obs[c3+bs], obs[c4+bs]))

        print(" ")



######################################################################################################
######################################################################################################

if __name__ == "__main__":
   main(sys.argv)
