from vehicle_controller import VehicleController

class BotType1Ctrl(VehicleController):

    """Defines the control algorithm for the Type 1 bot vehicle."""

    def __init__(self):
        super().__init__()
        print("///// BotType1Ctrl.__init__ entered.")
