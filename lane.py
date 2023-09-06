import math

class Lane:
    """Defines the geometry of a single lane and its relationship to other lanes in the map frame.
        Note: an adjoining lane must join this one exactly once (possibly at distance 0), and
                may or may not separate from it farther downtrack. Once it separates, it cannot
                rejoin.  If it does not separate, then separation location will be same as this
                lane's length.
    """

    def __init__(self,
                    my_id       : int,                  #ID of this lane
                    start_x     : float,                #X coordinate of the start of the lane, m
                    length      : float,                #total length of this lane, m (includes any downtrack buffer)
                    segments    : list,                 #list of straight segments that make up this lane; each item is
                                                        # a tuple containing: (x0, y0, x1, y1, length, speed limit, pavement type), where
                                                        # x0, y0 are map coordinates of the starting point, in m
                                                        # x1, y1 are map coordinates of the ending point, in m
                                                        # length is the length of the segment, in m
                                                        # speed limit is the regulatory limit in m/s
                                                        # pavement type is an enumeration using PavementType
                                                        #Each lane must have at least one segment, and segment lengths
                                                        # need to add up to total lane length
                    left_id     : int       = -1,       #ID of an adjoining lane to its left, or -1 for none
                    left_join   : float     = 0.0,      #P location where left lane first joins, m (param frame)
                    left_sep    : float     = math.inf,
                                                        #P location where left lane separates from this one, m (param frame)
                    right_id    : int       = -1,       #ID of an ajoining lane to its right, or -1 for none
                    right_join  : float     = 0.0,      #P location where right lane first joins, m (param frame)
                    right_sep   : float     = math.inf
                                                        #P location where right lane separates from this one, m (param frame)
                ):

        self.my_id = my_id
        self.start_x = start_x
        self.length = length
        self.left_id = left_id
        self.left_join = left_join
        self.left_sep = left_sep
        self.right_id = right_id
        self.right_join = right_join
        self.right_sep = right_sep
        self.segments = segments

        assert start_x >= 0.0, "Lane {} start_x {} is invalid.".format(my_id, start_x)
        assert length > 0.0, "Lane {} length of {} is invalid.".format(my_id, length)
        assert left_id >= 0  or  right_id >= 0, "Lane {} has no adjoining lanes.".format(my_id)
        assert len(segments) > 0, "Lane {} has no segments defined.".format(my_id)
        seg_len = 0.0
        for si, s in enumerate(segments):
            seg_len += s[4]
            assert s[0] != s[2]  or  s[1] != s[3], "Lane {}, segment {} both ends have same coords.".format(my_id, si)
            assert len(s) == 7, "Lane {}, segment {} malformed - incorrect number of elements.".format(my_id, si)
            assert s[5] > 0.0, "Lane {}, segment {} illegal speed limit {}".format(my_id, si, s[5])
        assert abs(seg_len - length) < 1.0, "Lane {} sum of segment lengths {} don't match total lane length {}.".format(my_id, seg_len, length)
        if left_id >= 0:
            assert left_id != my_id, "Lane {} left adjoining lane has same ID".format(my_id)
            assert left_sep > left_join, "Lane {} left_sep {} not larger than left_join {}".format(my_id, left_sep, left_join)
        if right_id >= 0:
            assert right_id != my_id, "Lane {} right adjoining lane has same ID".format(my_id)
            assert right_sep > right_join, "Lane {} right_sep {} not larger than right_join {}".format(my_id, right_sep, right_join)
        if left_id >= 0  and  right_id >= 0:
            assert left_id != right_id, "Lane {}: both left and right adjoining lanes share ID {}".format(my_id, left_id)
