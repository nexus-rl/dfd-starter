from rocketsim import Vec3, Angle
import math

# Floor: 0
# Center field: (0, 0)
# Side wall: x=±4096
# Side wall length: 7936
# Back wall: y=±5120
# Back wall length: 5888
# Ceiling: z=2044
# Goal height: z=642.775
# Goal center-to-post: 892.755
# Goal depth: 880
# Corner wall length: 1629.174
# The corner planes intersect the axes at ±8064 at a 45 degrees angle

ARENA_FLOOR = 0
ARENA_CENTER = Vec3(0, 0, ARENA_FLOOR)
ARENA_SIDE_WALL_X_POSITIVE = 4096
ARENA_SIDE_WALL_LENGTH = 7936
ARENA_BACK_WALL_Y_POSITIVE = 5120
ARENA_BACK_WALL_LENGTH = 5888
ARENA_SIDE_WALL_X_NEGATIVE = -ARENA_SIDE_WALL_X_POSITIVE
ARENA_BACK_WALL_Y_NEGATIVE = -ARENA_BACK_WALL_Y_POSITIVE
ARENA_CEILING = 2044
ARENA_GOAL_HEIGHT = 642.775 # The crossbar
ARENA_GOAL_CENTER_TO_POST = 892.755
ARENA_GOAL_FULL_WIDTH = 2 * ARENA_GOAL_CENTER_TO_POST
ARENA_GOAL_DEPTH = 880
ARENA_CORNER_WALL_LENGTH = 1629.174

# Orange is positive y, blue is negative y
# Left wall for the blue team is positive x, right wall for the blue team is negative x
# Consider adding back wall and side walls, as well as diagonal walls
# Probably worth defining volumes and planes for each of these

ARENA_BLUE_BACKBOARD_BOTTOM = Vec3(0, ARENA_BACK_WALL_Y_NEGATIVE, ARENA_GOAL_HEIGHT)
ARENA_BLUE_BACKBOARD_TOP = Vec3(0, ARENA_BACK_WALL_Y_NEGATIVE, ARENA_CEILING)
ARENA_BLUE_GOAL_LEFT_POST_FLOOR = Vec3(ARENA_GOAL_CENTER_TO_POST, ARENA_BACK_WALL_Y_NEGATIVE, 0)
ARENA_BLUE_GOAL_RIGHT_POST_FLOOR = Vec3(-ARENA_GOAL_CENTER_TO_POST, ARENA_BACK_WALL_Y_NEGATIVE, 0)
ARENA_BLUE_GOAL_CENTER_FLOOR = Vec3(0, ARENA_BACK_WALL_Y_NEGATIVE, 0)
ARENA_BLUE_GOAL_LEFT_POST_CROSSBAR= Vec3(ARENA_GOAL_CENTER_TO_POST, ARENA_BACK_WALL_Y_NEGATIVE, ARENA_GOAL_HEIGHT)
ARENA_BLUE_GOAL_RIGHT_POST_CROSSBAR = Vec3(-ARENA_GOAL_CENTER_TO_POST, ARENA_BACK_WALL_Y_NEGATIVE, ARENA_GOAL_HEIGHT)
ARENA_BLUE_GOAL_CENTER_CROSSBAR = Vec3(0, ARENA_BACK_WALL_Y_NEGATIVE, ARENA_GOAL_HEIGHT)
ARENA_BLUE_GOAL_LEFT_POST_CEILING = Vec3(ARENA_GOAL_CENTER_TO_POST, ARENA_BACK_WALL_Y_NEGATIVE, ARENA_CEILING)
ARENA_BLUE_GOAL_RIGHT_POST_CEILING = Vec3(-ARENA_GOAL_CENTER_TO_POST, ARENA_BACK_WALL_Y_NEGATIVE, ARENA_CEILING)
ARENA_BLUE_GOAL_CENTER_CEILING = Vec3(0, ARENA_BACK_WALL_Y_NEGATIVE, ARENA_CEILING)
ARENA_BLUE_GOAL_BACK_OF_NET = Vec3(0, ARENA_BACK_WALL_Y_NEGATIVE - ARENA_GOAL_DEPTH, 0)

ARENA_ORANGE_BACKBOARD_BOTTOM = Vec3(0, ARENA_BACK_WALL_Y_POSITIVE, ARENA_GOAL_HEIGHT)
ARENA_ORANGE_BACKBOARD_TOP = Vec3(0, ARENA_BACK_WALL_Y_POSITIVE, ARENA_CEILING)
ARENA_ORANGE_GOAL_LEFT_POST_FLOOR = Vec3(-ARENA_GOAL_CENTER_TO_POST, ARENA_BACK_WALL_Y_POSITIVE, 0)
ARENA_ORANGE_GOAL_RIGHT_POST_FLOOR = Vec3(ARENA_GOAL_CENTER_TO_POST, ARENA_BACK_WALL_Y_POSITIVE, 0)
ARENA_ORANGE_GOAL_CENTER_FLOOR = Vec3(0, ARENA_BACK_WALL_Y_POSITIVE, 0)
ARENA_ORANGE_GOAL_LEFT_POST_CROSSBAR= Vec3(-ARENA_GOAL_CENTER_TO_POST, ARENA_BACK_WALL_Y_POSITIVE, ARENA_GOAL_HEIGHT)
ARENA_ORANGE_GOAL_RIGHT_POST_CROSSBAR = Vec3(ARENA_GOAL_CENTER_TO_POST, ARENA_BACK_WALL_Y_POSITIVE, ARENA_GOAL_HEIGHT)
ARENA_ORANGE_GOAL_CENTER_CROSSBAR = Vec3(0, ARENA_BACK_WALL_Y_POSITIVE, ARENA_GOAL_HEIGHT)
ARENA_ORANGE_GOAL_LEFT_POST_CEILING = Vec3(-ARENA_GOAL_CENTER_TO_POST, ARENA_BACK_WALL_Y_POSITIVE, ARENA_CEILING)
ARENA_ORANGE_GOAL_RIGHT_POST_CEILING = Vec3(ARENA_GOAL_CENTER_TO_POST, ARENA_BACK_WALL_Y_POSITIVE, ARENA_CEILING)
ARENA_ORANGE_GOAL_CENTER_CEILING = Vec3(0, ARENA_BACK_WALL_Y_POSITIVE, ARENA_CEILING)
ARENA_ORANGE_GOAL_BACK_OF_NET = Vec3(0, ARENA_BACK_WALL_Y_POSITIVE + ARENA_GOAL_DEPTH, 0)

SMALL_BOOST_AMOUNT = 12
BIG_BOOST_AMOUNT = 100

SMALL_BOOSTS = [
    Vec3(0, -4240, 70),
    Vec3(-1792, -4184, 70),
    Vec3(1792, -4184, 70),
    Vec3(-940, -3308, 70),
    Vec3(940, -3308, 70),
    Vec3(0, -2816, 70),
    Vec3(-3584, -2484, 70),
    Vec3(3584, -2484, 70),
    Vec3(-1788, -2300, 70),
    Vec3(1788, -2300, 70),
    Vec3(-2048, -1036, 70),
    Vec3(0, -1024, 70),
    Vec3(2048, -1036, 70),
    Vec3(-1024, 0, 70),
    Vec3(1024, 0, 70),
    Vec3(-2048, 1036, 70),
    Vec3(0, 1024, 70),
    Vec3(2048, 1036, 70),
    Vec3(-1788, 2300, 70),
    Vec3(1788, 2300, 70),
    Vec3(-3584, 2484, 70),
    Vec3(3584, 2484, 70),
    Vec3(0, 2816, 70),
    Vec3(-940, 3310, 70),
    Vec3(940, 3308, 70),
    Vec3(-1792, 4184, 70),
    Vec3(1792, 4184, 70),
    Vec3(0, 4240, 70),
]

BIG_BOOSTS = [
    Vec3(-3072, -4096, 73),
    Vec3(3072, -4096, 73),
    Vec3(-3584, 0, 73),
    Vec3(3584, 0, 73),
    Vec3(-3072, 4096, 73),
    Vec3(3072, 4096, 73),
]

# Spawn Locations
# Kickoff 	Blue 	Orange
# Right corner 	loc: (-2048, -2560), yaw: 0.25 pi 	loc: (2048, 2560), yaw: -0.75 pi
# Left corner 	loc: (2048, -2560), yaw: 0.75 pi 	loc: (-2048, 2560), yaw: -0.25 pi
# Back right 	loc: (-256.0, -3840), yaw: 0.5 pi 	loc: (256.0, 3840), yaw: -0.5 pi
# Back left 	loc: (256.0, -3840), yaw: 0.5 pi 	loc: (-256.0, 3840), yaw: -0.5 pi
# Far back center 	loc: (0.0, -4608), yaw: 0.5 pi 	loc: (0.0, 4608), yaw: -0.5 pi
# Demolished 	Blue 	Orange
# Right inside 	loc: (-2304, -4608), yaw: 0.5 pi 	loc: (2304, 4608), yaw: -0.5 pi
# Right outside 	loc: (-2688, -4608), yaw: 0.5 pi 	loc: (2688, 4608), yaw: -0.5 pi
# Left inside 	loc: (2304, -4608), yaw: 0.5 pi 	loc: (-2304, 4608), yaw: -0.5 pi
# Left outside 	loc: (2688, -4608), yaw: 0.5 pi 	loc: (-2688, 4608), yaw: -0.5 pi

KICKOFF_BLUE = [
    (Vec3(-2048, -2560, 0), Angle().with_yaw(0.25 * math.pi)), # Right corner
    (Vec3(2048, -2560, 0), Angle().with_yaw(0.75 * math.pi)), # Left corner
    (Vec3(-256.0, -3840, 0), Angle().with_yaw(0.5 * math.pi)), # Back right
    (Vec3(256.0, -3840, 0), Angle().with_yaw(0.5 * math.pi)), # Back left
    (Vec3(0.0, -4608, 0), Angle().with_yaw(0.5 * math.pi)), # Far back center
]

KICKOFF_ORANGE = [
    (Vec3(-2048, 2560, 0), Angle().with_yaw(-0.25 * math.pi)), # Right corner
    (Vec3(2048, 2560, 0), Angle().with_yaw(-0.75 * math.pi)), # Left corner
    (Vec3(-256.0, 3840, 0), Angle().with_yaw(-0.5 * math.pi)), # Back right
    (Vec3(256.0, 3840, 0), Angle().with_yaw(-0.5 * math.pi)), # Back left
    (Vec3(0.0, 4608, 0), Angle().with_yaw(-0.5 * math.pi)), # Far back center
]

DEMOLISHED_BLUE = [
    (Vec3(-2304, -4608, 0), Angle().with_yaw(0.5 * math.pi)), # Right inside
    (Vec3(-2688, -4608, 0), Angle().with_yaw(0.5 * math.pi)), # Right outside
    (Vec3(2304, -4608, 0), Angle().with_yaw(0.5 * math.pi)), # Left inside
    (Vec3(2688, -4608, 0), Angle().with_yaw(0.5 * math.pi)), # Left outside
]

DEMOLISHED_ORANGE = [
    (Vec3(-2304, 4608, 0), Angle().with_yaw(-0.5 * math.pi)), # Right inside
    (Vec3(-2688, 4608, 0), Angle().with_yaw(-0.5 * math.pi)), # Right outside
    (Vec3(2304, 4608, 0), Angle().with_yaw(-0.5 * math.pi)), # Left inside
    (Vec3(2688, 4608, 0), Angle().with_yaw(-0.5 * math.pi)), # Left outside
]

# Elevation of Objects at Rest
#     Ball: 92.75 (its radius)
#     Hybrid: 17.00
#     Octane: 17.01
#     Dominus: 17.05
#     Breakout: 18.33
#     Batmobile/Plank: 18.65

ELEVATION_BALL = 92.75
ELEVATION_HYBRID = 17.00
ELEVATION_OCTANE = 17.01
ELEVATION_DOMINUS = 17.05
ELEVATION_BREAKOUT = 18.33
ELEVATION_BATMOBILE = 18.65
ELEVATION_PLANK = 18.65

# Physics
# Conversion: 1 uu = 1 cm (e.g. 2778 uu/s = 100 km/h)

CONVERT_UU_TO_CM = 1
CONVERT_UU_TO_M = 0.01
CONVERT_UU_TO_KM = 0.00001
CONVERT_UUS_TO_KPH = 0.036

# Ball
#     Radius: 92.75 uu
#     Max speed: 6000 uu/s

BALL_RADIUS = 92.75
BALL_MAX_SPEED = 6000

# Car
#     Max car speed (boosting): 2300 uu/s
#     Supersonic speed threshold: 2200 uu/s
#     Max driving speed (forward and backward) with no boost: 1410 uu/s
# Minimum and maximum rotation (in radians):

#     Pitch: [-pi/2, pi/2]
#     Yaw: [-pi, pi]
#     Roll: [-pi, pi]

# Maximum car angular acceleration:

#     Yaw: 9.11 radians/s^2
#     Pitch: 12.46 radians/s^2
#     Roll: 38.34 radians/s^2

# Maximum car angular velocity: 5.5 radians/s

CAR_MAX_SPEED = 2300
CAR_SUPERSONIC_SPEED = 2200
CAR_MAX_SPEED_NO_BOOST = 1410
CAR_MIN_PITCH = -0.5 * math.pi
CAR_MAX_PITCH = 0.5 * math.pi
CAR_MIN_YAW = -math.pi
CAR_MAX_YAW = math.pi
CAR_MIN_ROLL = -math.pi
CAR_MAX_ROLL = math.pi
CAR_MAX_ANGULAR_ACCELERATION_YAW = 9.11
CAR_MAX_ANGULAR_ACCELERATION_PITCH = 12.46
CAR_MAX_ANGULAR_ACCELERATION_ROLL = 38.34
CAR_MAX_ANGULAR_VELOCITY = 5.5