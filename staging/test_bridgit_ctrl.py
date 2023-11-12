import sys
from typing import Dict
from bridgit_ctrl import BridgitCtrl, PosInfo

def main(argv):

    bridgit = BridgitCtrl(None, None, None)

    # Case 1 - distinct sets
    a = {1: 14.0, 2: 16.0}
    b = {3: 33.3, 4: 44.4}
    res = bridgit._dict_union(a, b)
    assert res[1] == 14.0, "Case 1 failed on item 1"
    assert res[2] == 16.0, "Case 1 falied on item 2"
    assert res[3] == 33.3, "Case 1 failed on item 3"
    assert res[4] == 44.4, "Case 1 failed on item 4"
    assert len(res) == 4, "Case 1 failed due to extraneous elements: {}".format(res)

    # Case 2 - full overlap
    a = {1: 14.0, 2: 16.0}
    b = {1: 15.0, 2: 15.0}
    res = bridgit._dict_union(a, b)
    assert res[1] == 15.0, "Case 2 failed on item 1"
    assert res[2] == 16.0, "Case 2 failed on item 2"
    assert len(res) == 2, "Case 2 failed due to extraneous elements in the dict: {}".format(res)

    # Case 2 - some overlap
    a = {1: 14.0, 2: 16.0, 3: 17.0}
    b = {5: 27.7, 2: 10.5, 3: 23.3}
    res = bridgit._dict_union(a, b)
    assert res[1] == 14.0, "Case 3 failed on item 1"
    assert res[2] == 16.0, "Case 3 failed on item 2"
    assert res[3] == 23.3, "Case 3 failed on item 3"
    assert res[5] == 27.7, "Case 3 failed on item 5"
    assert len(res) == 4, "Case 3 failed due to extraneous elements: {}".format(res)

    # Case 4 - helper class
    info = PosInfo()
    print(info.pri())

if __name__ == "__main__":
   main(sys.argv)
