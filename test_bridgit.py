import sys
from obs_vec import ObsVec
from hp_prng import HpPrng
from roadway_b import Roadway
from vehicle import Vehicle
from bridgit_model import BridgitModel
from bridgit_guidance import BridgitGuidance
from bot_type1_model import BotType1Model
from bot_type1a_guidance import BotType1aGuidance


def main(argv):
    """Unit tester for the BridgitModel."""

    prng = HpPrng(17)
    debug = 0

    roadway = Roadway(debug)

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
                controller = BridgitGuidance(prng, roadway)
            else:
                print("Building a simple bot vehicle.")
                model = BotType1Model(roadway, length = 5.0)
                controller = BotType1aGuidance(prng, roadway)
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
    vehicles[5].cur_speed = 33.34
    vehicles[6].cur_speed = 33.37

    # Case A1
    vehicles[0].lane_id = 3
    vehicles[0].p       = 805.0
    actions = [0.75, 0.8]
    obs = vehicles[0].model.get_obs_vector(0, vehicles, actions)
    print_obs(obs, "Case A1 - ego lane = {}, p = {:.1f}".format(vehicles[0].lane_id, vehicles[0].p))

    # Case A2
    vehicles[0].lane_id = 1
    vehicles[0].p       = 812.2
    actions = [0.75, 0.8]
    obs = vehicles[0].model.get_obs_vector(0, vehicles, actions)
    print_obs(obs, "Case A2 - ego lane = {}, p = {:.1f}".format(vehicles[0].lane_id, vehicles[0].p))

    # Case A3
    vehicles[0].lane_id = 2
    vehicles[0].p       = 511.6
    actions = [0.75, 0.8]
    obs = vehicles[0].model.get_obs_vector(0, vehicles, actions)
    print_obs(obs, "Case A3 - ego lane = {}, p = {:.1f}".format(vehicles[0].lane_id, vehicles[0].p))

    # Case A4
    vehicles[0].lane_id = 1
    vehicles[0].p = 2406.9
    actions = [0.75, 0.8]
    obs = vehicles[0].model.get_obs_vector(0, vehicles, actions)
    print_obs(obs, "Case A4 - ego lane = {}, p = {:.1f}".format(vehicles[0].lane_id, vehicles[0].p))

    # Case A5
    vehicles[0].lane_id = 5
    vehicles[0].p       = 1260.0
    actions = [0.75, 0.8]
    obs = vehicles[0].model.get_obs_vector(0, vehicles, actions)
    print_obs(obs, "Case A5 - ego lane = {}, p = {:.1f}".format(vehicles[0].lane_id, vehicles[0].p))

    # Case A6
    vehicles[0].lane_id = 4
    vehicles[0].p       = 906.0
    actions = [0.75, 0.8]
    obs = vehicles[0].model.get_obs_vector(0, vehicles, actions)
    print_obs(obs, "Case A6 - ego lane = {}, p = {:.1f}".format(vehicles[0].lane_id, vehicles[0].p))

    # Case A7
    vehicles[0].lane_id = 0
    vehicles[0].p       = 2459.7
    actions = [0.75, 0.8]
    obs = vehicles[0].model.get_obs_vector(0, vehicles, actions)
    print_obs(obs, "Case A7 - ego lane = {}, p = {:.1f}".format(vehicles[0].lane_id, vehicles[0].p))



def print_obs(obs, label):

    print("\nTest case: {}\n".format(label))

    # Visual layers
    print("            Pavement (SL)                       Vehicles (spd)\n")
    for row in range(25):
        side_z = 24 - row
        c0 = ObsVec.BASE_LL + side_z*ObsVec.NORM_ELEMENTS
        c1 = ObsVec.BASE_L + side_z*ObsVec.NORM_ELEMENTS
        c3 = ObsVec.BASE_R + side_z*ObsVec.NORM_ELEMENTS
        c4 = ObsVec.BASE_RR + side_z*ObsVec.NORM_ELEMENTS

        p_row = ["  .  ", "  .  ", "  .  ", "  .  ", "  .  "]
        v_row = ["  .  ", "  .  ", "  .  ", "  .  ", "  .  "]
        if obs[c0+0] >= 0:
            p_row[0] = "{:5.2f}".format(obs[c0+1])
        if obs[c1+0] >= 0:
            p_row[1] = "{:5.2f}".format(obs[c1+1])
        if obs[c3+0] >= 0:
            p_row[3] = "{:5.2f}".format(obs[c3+1])
        if obs[c4+0] >= 0:
            p_row[4] = "{:5.2f}".format(obs[c4+1])

        if obs[c0+2] > 0:
            v_row[0] = "{:5.2f}".format(obs[c0+3])
        if obs[c1+2] > 0:
            v_row[1] = "{:5.2f}".format(obs[c1+3])
        if obs[c3+2] > 0:
            v_row[3] = "{:5.2f}".format(obs[c3+3])
        if obs[c4+2] > 0:
            v_row[4] = "{:5.2f}".format(obs[c4+3])

        if row < 20:
            ctr_z = 19 - row
            c2 = ObsVec.BASE_CTR_FRONT + ctr_z*ObsVec.CTR_ELEMENTS
            if obs[c2+0] >= 0:
                p_row[2] = "{:5.2f}".format(obs[c2+1])
            if obs[c2+2] > 0:
                v_row[2] = "{:5.2f}".format(obs[c2+3])

        elif row == 20:
            p_row[2] = "*Ego*"
            v_row[2] = "*Ego*"

        else:
            ctr_z = 24 - row
            c2 = ObsVec.BASE_CTR_REAR + ctr_z*ObsVec.NORM_ELEMENTS
            if obs[c2+0] >= 0:
                p_row[2] = "{:5.2f}".format(obs[c2+1])
            if obs[c2+2] > 0:
                v_row[2] = "{:5.2f}".format(obs[c2+3])

        print("{:2d} [{} {} {} {} {}]     [{} {} {} {} {}]"
              .format(side_z, p_row[0], p_row[1], p_row[2], p_row[3], p_row[4], v_row[0], v_row[1], v_row[2], v_row[3], v_row[4]))

    # Numerical details
    print("\n\n")
    for row in range(25):
        side_z = 24 - row
        c0 = ObsVec.BASE_LL + side_z*ObsVec.NORM_ELEMENTS
        c1 = ObsVec.BASE_L + side_z*ObsVec.NORM_ELEMENTS
        c3 = ObsVec.BASE_R + side_z*ObsVec.NORM_ELEMENTS
        c4 = ObsVec.BASE_RR + side_z*ObsVec.NORM_ELEMENTS

        if row < 20:
            ctr_z = 19 - row
            c2 = ObsVec.BASE_CTR_FRONT + ctr_z*ObsVec.CTR_ELEMENTS
            print("{:2d} Type: {:6.3f}   {:6.3f}   {:6.3f}   {:6.3f}   {:6.3f}".format(side_z, obs[c0+0], obs[c1+0], obs[c2+0], obs[c3+0], obs[c4+0]))
            print("     SL: {:6.3f}   {:6.3f}   {:6.3f}   {:6.3f}   {:6.3f}".format(obs[c0+1], obs[c1+1], obs[c2+1], obs[c3+1], obs[c4+1]))
            print("    Occ: {:6.3f}   {:6.3f}   {:6.3f}   {:6.3f}   {:6.3f}".format(obs[c0+2], obs[c1+2], obs[c2+2], obs[c3+2], obs[c4+2]))
            print("    Spd: {:6.3f}   {:6.3f}   {:6.3f}   {:6.3f}   {:6.3f}".format(obs[c0+3], obs[c1+3], obs[c2+3], obs[c3+3], obs[c4+3]))
            print("   LBdr:                   {:6.3f}".format(obs[c2+4]))
            print("   RBdr:                   {:6.3f}".format(obs[c2+5]))

        elif row == 20:
            print("{:2d} Type: {:6.3f}   {:6.3f}   *Ego *   {:6.3f}   {:6.3f}".format(side_z, obs[c0+0], obs[c1+0], obs[c3+0], obs[c4+0]))
            print("     SL: {:6.3f}   {:6.3f}   *Ego *   {:6.3f}   {:6.3f}".format(obs[c0+1], obs[c1+1], obs[c3+1], obs[c4+1]))
            print("    Occ: {:6.3f}   {:6.3f}   *Ego *   {:6.3f}   {:6.3f}".format(obs[c0+2], obs[c1+2], obs[c3+2], obs[c4+2]))
            print("    Spd: {:6.3f}   {:6.3f}   *Ego *   {:6.3f}   {:6.3f}".format(obs[c0+3], obs[c1+3], obs[c3+3], obs[c4+3]))

        else:
            ctr_z = 24 - row
            c2 = ObsVec.BASE_CTR_REAR + ctr_z*ObsVec.NORM_ELEMENTS
            print("{:2d} Type: {:6.3f}   {:6.3f}   {:6.3f}   {:6.3f}   {:6.3f}".format(side_z, obs[c0+0], obs[c1+0], obs[c2+0], obs[c3+0], obs[c4+0]))
            print("     SL: {:6.3f}   {:6.3f}   {:6.3f}   {:6.3f}   {:6.3f}".format(obs[c0+1], obs[c1+1], obs[c2+1], obs[c3+1], obs[c4+1]))
            print("    Occ: {:6.3f}   {:6.3f}   {:6.3f}   {:6.3f}   {:6.3f}".format(obs[c0+2], obs[c1+2], obs[c2+2], obs[c3+2], obs[c4+2]))
            print("    Spd: {:6.3f}   {:6.3f}   {:6.3f}   {:6.3f}   {:6.3f}".format(obs[c0+3], obs[c1+3], obs[c2+3], obs[c3+3], obs[c4+3]))

        print(" ")



######################################################################################################
######################################################################################################

if __name__ == "__main__":
   main(sys.argv)
