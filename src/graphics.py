from cmath import inf
import math
import pygame
from pygame.locals import *
from typing import List
from constants import Constants
from highway_env import HighwayEnv
from roadway_b import Roadway
from target_destination import TargetDestination

"""Provides all the graphics display for the inference program.

    Some notes about pixel geometry. These are subject to change, so should be taken with a grain of salt.
    Window size of 1800 (wide) x 800 (high) pixels works well.  With this size and the RoadwayB track, we
    get a scale factor of 0.588 pixels/meter.  A 30 m lane width display is pleasing to the eye and scales
    to 18 pixels across, which fits a 16x16 icon nicely, to represent a vehicle for visual purposes. Note
    that this width is only used for display purposes - for dynamic calculations, it is assumed the lane is
    a typical 3.7 m width.

    However, note that
        * a 5 m long vehicle covers only 3 pixels in the display
        * each 5 m sensor grid is 3x3 pixels across (despite the ridiculously large display of lane width)

    For the graphics, the R-S coordinate frame has its origin at the upper left of the window, with R
    values increasing to the right and S values increasing downward.  Meanwhile, the map coordinate frame
    (x, y) has its origin somewhere on the left edge of the map, so all X values are positive, increasing
    to the right, and y values increase going upward.
"""

class Graphics:

    # set up the colors & fonts
    BLACK                   = (  0,   0,   0)
    WHITE                   = (255, 255, 255)
    BACKGROUND_COLOR        = ( 50,  50,  50)
    LEGEND_COLOR            = WHITE
    LANE_EDGE_COLOR         = WHITE
    NEIGHBOR_COLOR          = ( 64, 128, 255)
    EGO_COLOR               = (168, 168, 0) #yellow
    PLOT_AXES_COLOR         = (200, 200,  50)
    DATA_COLOR              = WHITE
    REFERENCE_COLOR         = (120,  90,   0)
    BASIC_FONT_SIZE         = 14    #vertical pixels
    LARGE_FONT_SIZE         = 18    #vertical pixels
    AVG_PIX_PER_CHAR_BASIC  = 5.0 #TODO need to experiment with this
    AVG_PIX_PER_CHAR_LARGE  = 7.5

    # Other graphics constants
    LANE_WIDTH              = Roadway.WIDTH
    WINDOW_SIZE_R           = 1800      #window width, pixels
    WINDOW_SIZE_S           = 800       #window height, pixels
    REAL_TIME_RATIO         = 5.0       #Factor faster than real time
    FONT_PATH               = "docs/fonts"
    IMAGE_PATH              = "docs/images"
    BACKGROUND_IMAGE        = "/cda1_background_r2.bmp" #r2 has no target indicators, but is missing top 3 rows of pixels
    BACKGROUND_VERT_SHIFT   = 3 #num pixels it is shifted upward from where the plotting thinks it is

    # Geometry of data plots
    PLOT_H          = 80        #height of each plot, pixels
    PLOT_W          = 300       #width of each plot, pixels
    PLOT1_R         = 0.7*WINDOW_SIZE_R #upper-left corner of plot #1
    PLOT1_S         = 0.5*WINDOW_SIZE_S
    PLOT2_R         = PLOT1_R   #upper-left corner of plot #2
    PLOT2_S         = PLOT1_S + PLOT_H + BASIC_FONT_SIZE + 30
    LOW_SPEED       = 20.1      #m/s
    NOMINAL_SPEED   = 29.1      #m/s
    HIGH_SPEED      = 33.5      #m/s
    PLOT_STEPS      = 600       #max num time steps that can be plotted

    # Visual controls
    USE_VEHICLE_IMAGES  = True  #should bitmap images be used to represent vehicles? (if false, then circles)

    #TODO: revise this whole class to generalize the color & icon for each vehicle, and plot any data for any vehicle; there is no "ego" known here.

    def __init__(self,
                 env    : HighwayEnv,
                ):
        """Initializes the graphics and draws the roadway background display."""

        # Save the environment for future reference
        self.env = env

        # set up pygame
        pygame.init()
        self.pgclock = pygame.time.Clock()
        self.display_freq = Graphics.REAL_TIME_RATIO / env.time_step_size

        # set up the window
        self.window_surface = pygame.display.set_mode((Graphics.WINDOW_SIZE_R, Graphics.WINDOW_SIZE_S), flags = 0)
        pygame.display.set_caption('cda1')

        # draw the background onto the surface
        self.window_surface.fill(Graphics.BACKGROUND_COLOR)

        # Load the canned background image that shows the track, empty data plots and legend info
        self.background_image = pygame.image.load(Graphics.IMAGE_PATH + Graphics.BACKGROUND_IMAGE).convert_alpha()
        self.window_surface.blit(self.background_image, (0, 0))

        # Loop through all segments of all lanes and find the extreme coordinates to determine our bounding box
        x_min = inf
        y_min = inf
        x_max = -inf
        y_max = -inf
        for lane in env.roadway.lanes:
            for seg in lane.segments:
                x_min = min(x_min, seg[0], seg[2])
                y_min = min(y_min, seg[1], seg[3])
                x_max = max(x_max, seg[0], seg[2])
                y_max = max(y_max, seg[1], seg[3])

        # Add a buffer all around to ensure we have room to draw the edge lines, which are 1/2 lane width away
        x_min -= 0.5*Graphics.LANE_WIDTH
        y_min -= 0.5*Graphics.LANE_WIDTH
        x_max += 0.5*Graphics.LANE_WIDTH
        y_max += 0.5*Graphics.LANE_WIDTH

        # Define the transform between roadway coords (x, y) and display viewport pixels (r, s).  Note that
        # viewport origin is at upper left, with +s pointing downward.  Leave a few pixels of buffer on all sides
        # of the display so the lines don't bump the edge.
        buffer = 8 #pixels
        display_width = Graphics.WINDOW_SIZE_R - 2*buffer
        display_height = Graphics.WINDOW_SIZE_S - 2*buffer
        roadway_width = x_max - x_min
        roadway_height = y_max - y_min
        ar_display = display_width / display_height
        ar_roadway = roadway_width / roadway_height
        self.scale = display_height / roadway_height     #pixels/meter (on Tensorbook this is 0.588)
        if ar_roadway > ar_display:
            self.scale = display_width / roadway_width
        self.roadway_center_x = x_min + 0.5*roadway_width
        self.roadway_center_y = y_min + 0.5*roadway_height
        self.display_center_r = Graphics.WINDOW_SIZE_R // 2
        self.display_center_s = int(Graphics.WINDOW_SIZE_S - 0.5*roadway_height * self.scale) - 2*buffer #set the roadway as high in the window as possible
        print("      Graphics init: scale = {}, display center r,s = ({:4d}, {:4d}), roadway center x,y = ({:5.0f}, {:5.0f})"
                .format(self.scale, self.display_center_r, self.display_center_s, self.roadway_center_x, self.roadway_center_y))

        # set up fonts
        self.basic_font = pygame.font.Font(Graphics.FONT_PATH + "/FreeSans.ttf", Graphics.BASIC_FONT_SIZE)
        self.large_font = pygame.font.Font(Graphics.FONT_PATH + "/FreeSans.ttf", Graphics.LARGE_FONT_SIZE)

        """
        # Loop through the lane segments and draw the left and right edge lines of each
        for lane in env.roadway.lanes:
            for seg in lane.segments:
                self._draw_segment(seg[0], seg[1], seg[2], seg[3], Graphics.LANE_WIDTH)

        # Display the window legend & other footer text
        self._write_legend(0, Graphics.WINDOW_SIZE_S)

        pygame.display.update()
        #time.sleep(20) #debug only
        """

        # Initialize images - use convert_alpha to enable the transparent areas of the bitmaps
        self.crash_image = pygame.image.load(Graphics.IMAGE_PATH +              "/crash16.bmp").convert_alpha()
        self.off_road_image = pygame.image.load(Graphics.IMAGE_PATH +           "/off-road16.bmp").convert_alpha()
        self.vehicle_ego_image = pygame.image.load(Graphics.IMAGE_PATH +        "/Ego2_16.bmp").convert_alpha()
        self.vehicle_bridgit_image = pygame.image.load(Graphics.IMAGE_PATH +    "/Yellow_car16_simple.bmp").convert_alpha()
        self.vehicle_bot1a_image = pygame.image.load(Graphics.IMAGE_PATH +      "/Purple_car16_simple.bmp").convert_alpha()
        self.vehicle_bot1b_image = pygame.image.load(Graphics.IMAGE_PATH +      "/Blue_car16_simple.bmp").convert_alpha()
        self.target_primary_image = pygame.image.load(Graphics.IMAGE_PATH +     "/square_red_white_lg_16.bmp").convert_alpha()
        self.target_secondary_image = pygame.image.load(Graphics.IMAGE_PATH +   "/square_black_white_sm_16.bmp").convert_alpha()

        # Determine offsets for vehicle image centers - these needed to be subtracted from the (r, s) location in order to display an image
        # because the image display is keyed to the upper-left corner of the image. Assumes all vehicle images are the same size.
        image_rect = list(self.vehicle_ego_image.get_rect())
        self.veh_image_r_offset = (image_rect[2] - image_rect[0])//2
        self.veh_image_s_offset = (image_rect[3] - image_rect[1])//2 + Graphics.BACKGROUND_VERT_SHIFT

        # Display the training targets (primary) and bot targets (secondary)
        self._display_targets()

        # Set up lists of previous screen coords and display images for each vehicle
        vehicles = env.get_vehicle_data()
        self.prev_veh_r = [0] * len(vehicles)
        self.prev_veh_s = [0] * len(vehicles)
        self.veh_images = [self.vehicle_bot1b_image] * len(vehicles)
        self.veh_images[0] = self.vehicle_ego_image
        for v_idx in range(1, len(self.prev_veh_r)): #don't overwrite the ego vehicle
            gn = vehicles[v_idx].guidance.name
            if gn == "BridgitGuidance":
                self.veh_images[v_idx] = self.vehicle_bridgit_image
            elif gn == "BotType1aGuidance":
                self.veh_images[v_idx] = self.vehicle_bot1a_image

        # Initialize the previous vehicles' locations near the beginning of a lane (doesn't matter which lane for this step)
        for v_idx in range(len(self.prev_veh_r)):
            self.prev_veh_r[v_idx] = int(self.scale*(self.env.roadway.lanes[0].segments[0][0] - self.roadway_center_x)) + self.display_center_r
            self.prev_veh_s[v_idx] = Graphics.WINDOW_SIZE_S - \
                                     int(self.scale*(self.env.roadway.lanes[0].segments[0][1] - self.roadway_center_y)) - self.display_center_s
        #TODO: draw rectangles instead of circles, with length = vehicle length & width = 0.5*lane width
        self.veh_radius = int(0.25 * Graphics.LANE_WIDTH * self.scale) #radius of icon in pixels

        #
        #..........Add live data plots to the display
        #

        # Plot speed of a vehicle (ego vehicle if it is in the scenario, else vehicle 1)
        title = "Ego speed, m/s"
        if self.env.scenario >= 90:
            title = "Speed, m/s"
        self.plot_speed = Plot(self.window_surface, Graphics.PLOT1_R, Graphics.PLOT1_S, Graphics.PLOT_H, Graphics.PLOT_W, 0.0, \
                                   Constants.MAX_SPEED, max_steps = Graphics.PLOT_STEPS, title = title)
        self.plot_speed.add_reference_line(Graphics.LOW_SPEED, Graphics.REFERENCE_COLOR)
        self.plot_speed.add_reference_line(Graphics.NOMINAL_SPEED, Graphics.REFERENCE_COLOR)
        self.plot_speed.add_reference_line(Graphics.HIGH_SPEED, Graphics.REFERENCE_COLOR)

        # Plot the lane ID of the ego vehicle
        self.plot_lane = Plot(self.window_surface, Graphics.PLOT2_R, Graphics.PLOT2_S, Graphics.PLOT_H, Graphics.PLOT_W, 0, 5, \
                                max_steps = Graphics.PLOT_STEPS, title = "Ego lane ID", show_vert_axis_scale = False)
        self.plot_lane.add_reference_line(1, Graphics.REFERENCE_COLOR)
        self.plot_lane.add_reference_line(2, Graphics.REFERENCE_COLOR)
        self.plot_lane.add_reference_line(3, Graphics.REFERENCE_COLOR)
        self.plot_lane.add_reference_line(4, Graphics.REFERENCE_COLOR)
        self.plot_lane.add_reference_line(5, Graphics.REFERENCE_COLOR)


    def update(self,
               action  : list,      #vector of actions for the ego vehicle for the current time step
               obs     : list,      #vector of observations of the ego vehicle for the current time step
               vehicles: list,      #list of Vehicle objects, with item [0] as the ego vehicle
              ):
        """Paints all updates on the display screen, including the new motion of every vehicle and any data plots."""

        # Loop through each vehicle in the scenario
        for v_idx in range(len(vehicles)):

            # Grab the background under where we want the vehicle to appear & erase the old vehicle
            #pygame.draw.circle(self.window_surface, Graphics.BACKGROUND_COLOR, (self.prev_veh_r[v_idx], self.prev_veh_s[v_idx]), self.veh_radius, 0)

            # Replace the background under where the vehicle was previously located
            image_r = self.prev_veh_r[v_idx] - self.veh_image_r_offset
            image_s = self.prev_veh_s[v_idx] - self.veh_image_s_offset
            pos = self.veh_images[v_idx].get_rect().move(image_r, image_s)
            self.window_surface.blit(self.background_image, pos, pos)

            # Skip over vehicles that are inactive for reasons other than a crash or ego off-roading
            if not vehicles[v_idx].active  and  not (vehicles[v_idx].crashed  or  (v_idx == 0  and  vehicles[v_idx].off_road)):
                continue

            # Get the vehicle's new location on the surface
            new_x, new_y = self._get_vehicle_coords(vehicles, v_idx)
            new_r, new_s = self._map2screen(new_x, new_y)

            # If the vehicle has crashed, then display the crash symbol at its location
            if vehicles[v_idx].crashed:
                #print("***   Graphics.update: vehicle {} crashed.".format(v_idx))
                pos = self.crash_image.get_rect().move(new_r - self.veh_image_r_offset, new_s - self.veh_image_s_offset) #defines the upper-left corner of the image
                self.window_surface.blit(self.crash_image, pos)

            # Else if the ego vehicle ran off-road, display that symbol next to the lane where it happened. Normally off-roading is to the side
            # of the lane (due to illegal lane change attempt); if so, we show the image to that side.
            elif v_idx == 0  and  vehicles[v_idx].off_road:
                lateral_offset = 0 #as if it ran off the end of a lane
                if vehicles[v_idx].lane_change_status == "left":
                    lateral_offset = int(self.LANE_WIDTH*self.scale + 0.5)
                elif vehicles[v_idx].lane_change_status == "right":
                    lateral_offset = -int(self.LANE_WIDTH*self.scale + 0.5)
                pos = self.off_road_image.get_rect().move(new_r - self.veh_image_r_offset, new_s - self.veh_image_s_offset + lateral_offset) #defines the upper-left corner of the image
                self.window_surface.blit(self.off_road_image, pos)

            # else the vehicle is still active, so display the vehicle in its new location.  Note that the obs vector is not scaled at this point.
            else:
                #pygame.draw.circle(self.window_surface, self.veh_colors[v_idx], (new_r, new_s), self.veh_radius, 0)
                pos = self.veh_images[v_idx].get_rect().move(new_r - self.veh_image_r_offset, new_s - self.veh_image_s_offset)
                self.window_surface.blit(self.veh_images[v_idx], pos)

           # Update the previous location
            self.prev_veh_r[v_idx] = new_r
            self.prev_veh_s[v_idx] = new_s

        # Update data plots
        pv = 0
        if self.env.scenario >= 90:
            pv = 1
        self.plot_speed.update(vehicles[pv].cur_speed)
        self.plot_lane.update(5 - vehicles[pv].lane_id)

        # Repaint the surface
        pygame.display.update()
        #print("   // Graphics: moving vehicle {} from r,s = ({:4d}, {:4d}) to ({:4d}, {:4d}) and new x,y = ({:5.0f}, {:5.0f})"
        #        .format(v_idx, self.prev_veh_r[v_idx], self.prev_veh_s[v_idx], new_r, new_s, new_x, new_y))

        # Pause until the next time step
        self.pgclock.tick(self.display_freq)


    def close(self):
        pygame.quit()


    def key_press_event(self) -> int:
        """Determines if a keyboard key has been pressed since system event buffer was last flushed.
            Flushes the event buffer up to the point of the first detected key press, or until buffer is empty.
            Ignores key release events.
            Returns the value of the key if one was detected, or None if no key presses in the event buffer.
                Key value is expressed in pygame.locals keycodes (beginning with K_).
        """

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                #print("      Key pressed: ", event.key)
                return event.key

        return None


    def wait_for_key_press(self) -> int:
        """Suspends processing until a keyboard key is pressed. Flushes the event buffer up to the point of the
            first detected key press. Ignores key release events.
            Returns the value of the key pressed, as expressed in pygame.locals keycodes (beginning with K_).
        """

        while True:
            key = self.key_press_event()
            if key is not None:
                return key


    def _draw_segment(self,
                      x0        : float,
                      y0        : float,
                      x1        : float,
                      y1        : float,
                      w         : float
                     ):
        """Draws a single lane segment on the display, which consists of the left and right edge lines.
            ASSUMES that all segments are oriented with headings between 0 and 180 deg for simplicity.
        """

        # Find the scaled lane end-point pixel locations (these are centerline of the lane)
        r0, s0 = self._map2screen(x0, y0)
        r1, s1 = self._map2screen(x1, y1)

        # Find the scaled width of the lane
        ws = 0.5 * w * self.scale

        angle = math.atan2(y1-y0, x1-x0) #angle above horizontal (to the right), radians in [-pi, pi]
        sin_a = math.sin(angle)
        cos_a = math.cos(angle)

        # Find the screen coords of the left edge line
        left_r0 = r0 - ws*sin_a
        left_r1 = r1 - ws*sin_a
        left_s0 = s0 - ws*cos_a
        left_s1 = s1 - ws*cos_a

        # Find the screen coords of the right edge line
        right_r0 = r0 + ws*sin_a
        right_r1 = r1 + ws*sin_a
        right_s0 = s0 + ws*cos_a
        right_s1 = s1 + ws*cos_a

        # Draw the edge lines
        pygame.draw.line(self.window_surface, Graphics.LANE_EDGE_COLOR, (left_r0, left_s0), (left_r1, left_s1))
        pygame.draw.line(self.window_surface, Graphics.LANE_EDGE_COLOR, (right_r0, right_s0), (right_r1, right_s1))


    def _get_vehicle_coords(self,
                            vehicles    : List, #list of all Vehicles in the scenario
                            vehicle_id  : int   #ID of the vehicle; 0=ego, others=neighbor vehicles
                           ) -> tuple:
        """Returns the map frame coordinates of the indicated vehicle based on its lane ID and distance downtrack.

            CAUTION: this is a general solution for any geometry defined in the convention of the Roadway class,
            but ASSUMES that all lane segments are generally pointing toward the right (heading in (0, 180)).
        """

        assert 0 <= vehicle_id < len(vehicles), "///// _get_vehicle_coords: invalid vehicle_id = {}".format(vehicle_id)

        road = self.env.roadway
        lane = vehicles[vehicle_id].lane_id
        x, y = road.param_to_map_frame(vehicles[vehicle_id].p, lane)

        # Figure out which segment of the lane it is in
        found = False
        for seg in road.lanes[lane].segments:
            if x <= seg[2]: #right end x coord

                # Find how far down this segment we are, and use that to interpolate the y coordinate, in case it is angled.
                assert x >= seg[0], "///// _get_vehicle_coords: vehicle X coord {:.2f} is outside of segment with X bounds {:.2f} and {:.2f} on lane {}" \
                                    .format(x, seg[0], seg[2], lane)
                f = (x - seg[0]) / (seg[2] - seg[0])
                y = seg[1] + f*(seg[3] - seg[1])
                found = True
                break

        if not found:
            raise LookupError("///// _get_vehicle_coords: could not find segment in lane {} that contains vehicle X = {:.3f}".format(x))

        return x, y


    def _map2screen(self,
                     x      : float,        #X coordinate in map frame
                     y      : float,        #Y coordinate in map frame
                    ) -> tuple:             #Returns (r, s) coordinates in the screen frame (pixels)
        """Converts an (x, y) point in the map frame to the nearest (r, s) point in the screen coordinates."""

        r = int(self.scale*(x - self.roadway_center_x) + 0.5) + self.display_center_r
        s = Graphics.WINDOW_SIZE_S - int(self.scale*(y - self.roadway_center_y) + 0.5) - self.display_center_s
        return r, s


    def _display_targets(self):
        """Displays each of the primary and secondary target locations. All targets are displayed, since all are used by the bot vehicles.
            Only valid, active targets are used by the ego vehicle, and those are displayed differently.
        """

        # Display the secondary ones first, as this may be a superset of the primary (active) targets
        for tgt in self.env.b_targets:
            self._display_target(tgt, self.target_secondary_image)

        # Now display the primary targets, whose images will overlay any secondaries that coincide
        for tgt in self.env.t_targets:
            if tgt.active:
                self._display_target(tgt, self.target_primary_image)


    def _display_target(self,
                            tgt     : TargetDestination,
                            image   : pygame.image,
                        ):
        """Displays a single target destination, using the desired image, at its defined location."""

        x, y = self.env.roadway.param_to_map_frame(tgt.p, tgt.lane_id)
        r, s = self._map2screen(x, y)
        pos = image.get_rect().move(r - self.veh_image_r_offset, s - self.veh_image_s_offset)
        self.window_surface.blit(image, pos)



    def _write_legend(self,
                      r     : int,          #R coordinate of upper-left corner of the legend
                      s     : int,          #S coordinate of the upper-left corner of the legend
                     ):
        """Creates the legend display for all the info in the window."""

        # Create the text on a separate surface and copy it to the display surface
        title = "SPACE = Start/Pause/Resume,  ESC = Exit"
        width = len(title) * Graphics.AVG_PIX_PER_CHAR_LARGE
        text = self.basic_font.render(title, True, Graphics.LEGEND_COLOR, Graphics.BACKGROUND_COLOR)
        text_rect = text.get_rect()
        text_rect.center = (r + width//2, s - Graphics.LARGE_FONT_SIZE)
        self.window_surface.blit(text, text_rect)


######################################################################################################
######################################################################################################


class Plot:
    """Displays an x-y plot of time series data on the screen."""

    def __init__(self,
                 surface    : pygame.Surface,   #the Pygame surface to draw on
                 corner_r   : int,              #X coordinate of the upper-left corner, screen pixels
                 corner_s   : int,              #Y coordinate of the upper-left corner, screen pixels
                 height     : int,              #height of the plot, pixels
                 width      : int,              #width of the plot, pixels
                 min_y      : float,            #min value of data to be plotted on Y axis
                 max_y      : float,            #max value of data to be plotted on Y axis
                 max_steps  : int       = 180,  #max num time steps that will be plotted along X axis
                 axis_color : tuple     = Graphics.PLOT_AXES_COLOR, #color of the axes
                 data_color : tuple     = Graphics.DATA_COLOR, #color of the data curve being plotted
                 title      : str       = None,  #Title above the plot
                 show_vert_axis_scale: bool = True, #should the numerical scale on the vertical axis be displayed?
                ):
        """Defines and draws the empty plot on the screen, with axes and title."""

        assert max_y > min_y, "///// Plot defined with illegal min_y = {}, max_y = {}".format(min_y, max_y)
        assert max_steps > 0, "///// Plot defined with illegal max_steps = {}".format(max_steps)
        assert corner_r >= 0, "///// Plot defined with illegal corner_r = {}".format(corner_r)
        assert corner_s >= 0, "///// Plot defined with illegal corner_s = {}".format(corner_s)
        assert height > 0,    "///// Plot defined with illegal height = {}".format(height)
        assert width > 0,     "///// Plot defined with illegal width = {}".format(width)

        self.surface = surface
        self.cr = corner_r
        self.cs = corner_s - Graphics.BACKGROUND_VERT_SHIFT
        self.height = height
        self.width = width
        self.min_y = min_y
        self.max_y = max_y
        self.max_steps = max_steps
        self.axis_color = axis_color
        self.data_color = data_color

        # Determine scale factors for the data
        self.r_scale = self.width / max_steps #pixels per time step
        self.s_scale = self.height / (max_y - min_y) #pixels per unit of data value

        # Initialize drawing coordinates for the data curve (in (r, s) pixel location)
        self.prev_r = None
        self.prev_s = None

        self.basic_font = pygame.font.Font(Graphics.FONT_PATH + "/FreeSans.ttf", Graphics.BASIC_FONT_SIZE)

        # Draw the axes - for numbering, assume that the given min & max are "nice" numbers, so don't need to search
        # for nearest nice numbers.
        pygame.draw.line(surface, axis_color, (self.cr, self.cs + height), (self.cr + width, self.cs + height))
        pygame.draw.line(surface, axis_color, (self.cr, self.cs + height), (self.cr, self.cs))
        if show_vert_axis_scale:
            self._make_y_label(min_y, self.cs + height)
            self._make_y_label(max_y, self.cs)

        # Create the plot's text on a separate surface and copy it to the display surface
        if title is not None:
            text = self.basic_font.render(title, True, axis_color, Graphics.BACKGROUND_COLOR)
            text_rect = text.get_rect()
            text_rect.center = (self.cr + width//2, self.cs - Graphics.BASIC_FONT_SIZE)
            surface.blit(text, text_rect)

        pygame.display.update()


    def add_reference_line(self,
                           y_value  : float,
                           color    : tuple,
                          ):
        """Draws a horizontal line across the plot at the specified y value."""

        assert self.min_y <= y_value <= self.max_y, "///// Error: Plot.add_reference_line called with invalid y_value = {}".format(y_value)

        s = self.cs + Graphics.PLOT_H - y_value*self.s_scale
        pygame.draw.line(self.surface, color, (self.cr, s), (self.cr + Graphics.PLOT_W, s))


    def update(self,
               data     : float,    #the real-world data value to be plotted (Y value)
              ):
        """Adds the next sequential data point to the plot."""

        # If there has been no data plotted so far, then set the first point
        if self.prev_r is None:
            self.prev_r = self.cr
            self.prev_s = self.cs + Graphics.PLOT_H - data*self.s_scale

        # Else draw a line from the previous point to the current point
        else:
            new_r = self.prev_r + self.r_scale
            new_s = self.cs + Graphics.PLOT_H - data*self.s_scale
            if new_r <= self.cr + Graphics.PLOT_W:
                pygame.draw.line(self.surface, self.data_color, (self.prev_r, self.prev_s), (new_r, new_s))
                self.prev_r = new_r
                self.prev_s = new_s
                #pygame.display.update()


    def _make_y_label(self,
                      val       : float,    #value to be displayed
                      location  : int,      #vertical (s) coordinate for the center of the label, pixels
                     ):
        """Creates a numeric label for the Y axis, placing it at the specified vertical location. Displayed value
            will be truncated to the nearest integer from the given value.  Label will be displayed on the surface.
        """

        label = "{:.0f}".format(val)
        text = self.basic_font.render(label, True, self.axis_color, Graphics.BACKGROUND_COLOR)
        text_rect = text.get_rect()
        text_rect.center = (self.cr - (len(label) + 1)*Graphics.BASIC_FONT_SIZE//2, location)
        self.surface.blit(text, text_rect)
