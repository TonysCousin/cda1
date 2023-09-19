class TargetDestination:

    """Stores the location of a target destination within the roadway network.
        Vehicles may compare their location to this for evaluating their success.
    """

    def __init__(self,
                 lane   : int,      #ID of the lane where the target lives
                 p      : float,    #P coordinate of the target, m
                ):

        self.lane_id = lane
        self.p = p
