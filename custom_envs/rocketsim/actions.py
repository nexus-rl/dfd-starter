import numpy as np

def make_lookup_table():
    actions = []
    # Ground
    for throttle in (-1, 0, 0.5, 1):
        for steer in (-1, -0.5, 0, 0.5, 1):
            for boost in (0, 1):
                for handbrake in (0, 1):
                    if boost == 1 and throttle != 1:
                        continue
                    actions.append(
                        [throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
    # Aerial
    for pitch in (-1, -0.75, -0.5, 0, 0.5, 0.75, 1):
        for yaw in (-1, -0.75, -0.5, 0, 0.5, 0.75, 1):
            for roll in (-1, 0, 1):
                for jump in (0, 1):
                    for boost in (0, 1):
                        if jump == 1 and yaw != 0:  # Only need roll for sideflip
                            continue
                        if pitch == roll == jump == 0:  # Duplicate with ground
                            continue
                        # Enable handbrake for potential wavedashes
                        handbrake = jump == 1 and (
                                pitch != 0 or yaw != 0 or roll != 0)
                        actions.append(
                            [boost, yaw, pitch, yaw, roll, jump, boost, handbrake])
    # append stall
    actions.append([0, 1, 0, 0, -1, 1, 0, 0])
    actions = np.array(actions)
    return actions

actions = make_lookup_table()