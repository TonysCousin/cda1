import sys
import time
from graphics import Graphics
from highway_env_wrapper import HighwayEnvWrapper

"""This program draws the roadway geometry and other static imagery for the graphical background.
    Once it is displayed, take a screen shot of it and convert to a .bmp image for use by the
    Graphics package to display instead of drawing it live on each run (doing so allows grabbing
    of background pixels to refill places where animated elements have been drawn).
"""
def main(argv):

    # Create the environment model, which will include the roadway geometry
    env = HighwayEnvWrapper({}) #don't need any config params for this program

    # Set up the basic graphics geometry and display the window
    g = Graphics(env, background_only = True)

    # Wait for a long time; user will have to kill the program to finish.
    print("///// Press Ctrl-C to end program.")
    time.sleep(9999)
    exit(0)

if __name__ == "__main__":
   main(sys.argv)
