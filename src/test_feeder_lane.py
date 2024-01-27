import sys
import copy
from roadway_b import RoadwayB
from target_destination import TargetDestination

def main(argv):

    # Create the roadway geometry
    roadway = RoadwayB(0)

    # Define the target destinations for the ego vehicle (T targets) and for the bot vehicles (B targets)
    t_targets = roadway.targets

    assert      t_targets[0].is_reachable_from(0, 1733.0), "***** Case 1 failed."
    assert      t_targets[0].is_reachable_from(1, 602.3),  "***** Case 2 failed."
    assert not  t_targets[0].is_reachable_from(4, 1555.5), "***** Case 3 failed."
    assert not  t_targets[0].is_reachable_from(5, 1349.6), "***** Case 4 failed."
    assert      t_targets[0].is_reachable_from(5, 1129.1), "***** Case 5 failed."
    assert      t_targets[1].is_reachable_from(3, 888.2),  "***** Case 6 failed."
    assert not  t_targets[1].is_reachable_from(0, 2399.0), "***** Case 7 failed."
    assert not  t_targets[2].is_reachable_from(3, 2188.8), "***** Case 8 failed."
    assert      t_targets[2].is_reachable_from(0, 2411.0), "***** Case 9 failed."
    assert      t_targets[2].is_reachable_from(4, 900.0),  "***** Case 10 failed."
    assert not  t_targets[2].is_reachable_from(4, 1500.0), "***** Case 11 failed."
    assert      t_targets[3].is_reachable_from(4, 900.0),  "***** Case 12 failed."
    assert      t_targets[3].is_reachable_from(4, 1500.0), "***** Case 13 failed."
    assert      t_targets[3].is_reachable_from(2, 1000.0), "***** Case 14 failed."
    assert not  t_targets[3].is_reachable_from(2, 1355.9), "***** Case 15 failed."
    assert not  t_targets[3].is_reachable_from(0, 1650.0), "***** Case 16 failed."

if __name__ == "__main__":
   main(sys.argv)
