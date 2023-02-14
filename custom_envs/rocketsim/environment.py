from rocketsim import Angle, Vec3
from rocketsim.sim import Arena, CarConfig, GameMode, Team, Car, CarControls, Ball
from .actions import actions

import pygame
import numpy as np
import os
import gym
import os

GOAL_THRESHOLD = 100

class Environment(gym.Env):
    def __init__(self, opt_id="fd_worker_{}".format(os.getpid()), render_mode=None, **kwargs):
        self.opt_id = opt_id
        self.rng = np.random.RandomState(0)
        self.arena = None
        self.agent_id = None
        self.last_action = None
        self.timeout_ticks = None
        self.goal_pos = None
        self.last_agent_pos = None
        self.last_ball_pos = None
        self.action_space = gym.spaces.Discrete(len(actions))

        obs = self.reset()[0]
        obs_shape = np.shape(self.reset()[0])
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, obs_shape)

        self._check_obs(obs)

        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((1000, 800))
        else:
            self.screen = None

    def _make_action(self, action):
        action = actions[action]
        throttle, steer, yaw, pitch, roll, jump, boost, handbrake = action
        return CarControls(
            throttle=throttle,
            steer=steer,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            jump=jump,
            boost=boost,
            handbrake=handbrake,
        )

    def step(self, action):
        if self.arena is None:
            raise Exception("Arena not initialized, you must call reset() before step()")

        controls = self._make_action(int(action))
        self.arena.set_car_controls(self.agent_id, controls)
        self.arena.step(8)

        reward = self._reward_fn()

        done = self._distance_car_from_target(self.goal_pos) < GOAL_THRESHOLD
        self.timeout_ticks -= 1
        timeout = self.timeout_ticks <= 0
        self.last_action = action
        if self.screen:
            self.render()

        if self._is_agent_upside_down():
            done = True
            reward = -10

        obs = self._form_obs()
        self.last_agent_pos = self._get_agent().get_pos()
        self.last_ball_pos = self._get_ball().get_pos()

        return obs, reward, done, timeout, {}

    def _is_agent_upside_down(self):
        agent_roll = self._get_agent().get_angles().roll
        return agent_roll > 0.8 or agent_roll < -0.8

    def _subtract_vec3(self, a: Vec3, b: Vec3):
        return np.array([a.x - b.x, a.y - b.y, a.z - b.z])

    def _vel_car_ball(self):
        if self.last_agent_pos is None or self.last_ball_pos is None:
            return 0.0

        agent_pos = self._get_agent().get_pos()
        ball_pos = self._get_ball().get_pos()

        last_distance = np.linalg.norm(self._subtract_vec3(self.last_ball_pos, self.last_agent_pos))
        current_distance = np.linalg.norm(self._subtract_vec3(ball_pos, agent_pos))

        return (last_distance - current_distance) / (6000.0 + 2300.0)

    def _vel_ball_goal(self):
        if self.last_ball_pos is None:
            return 0.0

        ball_pos = self._get_ball().get_pos()

        last_distance = np.linalg.norm(self._subtract_vec3(self.goal_pos, self.last_ball_pos))
        current_distance = np.linalg.norm(self._subtract_vec3(self.goal_pos, ball_pos))

        return (last_distance - current_distance) / 2300.0

    def _vel_car_goal(self):
        if self.last_agent_pos is None:
            return 0.0

        agent_pos = self._get_agent().get_pos()

        last_distance = np.linalg.norm(self._subtract_vec3(self.goal_pos, self.last_agent_pos))
        current_distance = np.linalg.norm(self._subtract_vec3(self.goal_pos, agent_pos))

        return (last_distance - current_distance) / 2300.0

    def _distance_ball_from_target(self, target: Vec3):
        ball = self._get_ball()
        return np.linalg.norm(self._subtract_vec3(ball.get_pos(), target))

    def _distance_car_from_target(self, target: Vec3):
        agent = self._get_agent()
        return np.linalg.norm(self._subtract_vec3(agent.get_pos(), target))

    def _reward_fn(self):
        goal_reward = 1 if self._distance_car_from_target(self.goal_pos) < GOAL_THRESHOLD else 0
        return self._vel_car_goal() * 0.1 + goal_reward * 0.9

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        if self.arena is None:
            self.arena = Arena(GameMode.Soccar)
            self.agent_id = self.arena.add_car(Team.Blue, CarConfig.Octane)

        self.timeout_ticks = 200
        self.last_action = None

        ball = self._get_ball()
        # self.last_ball_pos = ball.pos = Vec3(self.rng.uniform(-3000, 3000), self.rng.uniform(-3000, 3000), 17)
        self.last_ball_pos = ball.pos = Vec3(0, 0, 5000)
        # ball.vel = Vec3(0, 0, 0.5)
        self.arena.ball = ball

        agent = self._get_agent()
        # x = +- 3000
        # y = +- 3000
        self.last_agent_pos = agent.pos = Vec3(self.rng.uniform(-3000, 3000), self.rng.uniform(-3000, 3000), 17)
        agent.angles = Angle(0, 0, 0)
        agent.boost = 100

        self.goal_pos = Vec3(self.rng.uniform(-3000, 3000), self.rng.uniform(-3000, 3000), 17)
        #self.goal_pos = Vec3(0, 0, 17)
        self.arena.set_car(self.agent_id, agent)

        return self._form_obs(), {}

    def _get_agent(self) -> Car:
        return self.arena.get_car(self.agent_id)

    def _get_ball(self) -> Ball:
        return self.arena.get_ball()

    def _form_obs(self):
        # most of these values come from:
        # https://github.com/ZealanL/RocketSim/blob/main/src/RLConst.h

        POS_SCALE = 1/5000
        ANGLE_SCALE = 1/3.14159
        BOOST_SCALE = 1/100
        AGENT_VEL_SCALE = 1/2300
        BALL_VEL_SCALE = 1/6000
        BALL_ANGVEL_SCALE = 1/6
        AGENT_ANGVEL_SCALE = 1/5.5
        ACTION_SCALE = 1/len(actions)

        agent = self._get_agent()
        ball = self._get_ball()
        agent_pos = agent.get_pos()
        agent_vel = agent.get_vel()

        # Bullet expresses angular velocities as a 3d vector that points along
        # the axis of rotation. This is done to make it trivial to add angular
        # velocities together.
        agent_angvel = agent.get_angvel()

        agent_angles = agent.get_angles()
        ball_pos = ball.get_pos()
        ball_vel = ball.get_vel()
        ball_angvel = ball.get_angvel()

        # list of 3-tuples of friendly-name, root value, and scale value
        values = np.array([
            ("agent pos x", agent_pos.x, POS_SCALE),
            ("agent pos y", agent_pos.y, POS_SCALE),
            ("agent pos z", agent_pos.z, POS_SCALE),

            ("agent angle pitch", agent_angles.pitch, ANGLE_SCALE),
            ("agent angle yaw", agent_angles.yaw, ANGLE_SCALE),
            ("agent angle roll", agent_angles.roll, ANGLE_SCALE),

            ("agent vel x", agent_vel.x, AGENT_VEL_SCALE),
            ("agent vel z", agent_vel.y, AGENT_VEL_SCALE),
            ("agent vel y", agent_vel.z, AGENT_VEL_SCALE),

            ("agent angvel x", agent_angvel.x, AGENT_ANGVEL_SCALE),
            ("agent angvel y", agent_angvel.y, AGENT_ANGVEL_SCALE),
            ("agent angvel z", agent_angvel.z, AGENT_ANGVEL_SCALE),

            ("boost", agent.boost, BOOST_SCALE),

            ("ball pos x", ball_pos.x, POS_SCALE),
            ("ball pos y", ball_pos.y, POS_SCALE),
            ("ball pos z", ball_pos.z, POS_SCALE),

            ("ball vel x", ball_vel.x, BALL_VEL_SCALE),
            ("ball vel y", ball_vel.y, BALL_VEL_SCALE),
            ("ball vel z", ball_vel.z, BALL_VEL_SCALE),

            ("ball angvel x", ball_angvel.x, BALL_ANGVEL_SCALE),
            ("ball angvel y", ball_angvel.y, BALL_ANGVEL_SCALE),
            ("ball angvel z", ball_angvel.z, BALL_ANGVEL_SCALE),

            ("goal x", self.goal_pos.x, POS_SCALE),
            ("goal y", self.goal_pos.y, POS_SCALE),
            ("goal z", self.goal_pos.z, POS_SCALE),

            ("last action", self.last_action, ACTION_SCALE) if self.last_action is not None else ("last action", -1, 1),

            ("vel car ball", self._vel_car_ball(), 1),
            ("vel car goal", self._vel_car_goal(), 1),
            ("vel ball goal", self._vel_ball_goal(), 1),
        ], dtype=[("name", "U20"), ("root", np.float32), ("scale", np.float32)])

        #create obs by multiplying the root value by the scale value:
        obs = values["root"] * values["scale"]

        self._check_obs(obs, values)

        # self._check_obs(obs, values, scale)
        
        return obs
    
    def _check_obs(self, obs, values=None):
        if not hasattr(self, "observation_space"):
            return

        if obs not in self.observation_space:
            # enumerate the values here to give a good error message
            for i, val in enumerate(obs):
                if not (-1 <= val <= 1):
                    if values is not None:
                        print(f"WARN: Observation space error: {i}: {val} (derived from '{values['name'][i]}', value {values['root'][i]}, scaled by {values['scale'][i]}) not in range [-1, 1]")

                    print(f"WARN: Observation space error: {i}: {val} not in range [-1, 1]")

    def close(self):
        pass

    def render(self):
        if self.screen is None:
            return
        else:
            display = self.screen
        display.fill((255, 255, 255))

        # Get the agent and ball
        agent_pos = self._get_agent().get_pos()
        ball_pos = self._get_ball().get_pos()

        # Get obs so we can draw them
        obs = self._form_obs()

        # Draw the agent
        pygame.draw.circle(display, (255, 0, 0), (int(400 + agent_pos.x / 10), int(300 + agent_pos.y/10)), 10)
        # Draw text above the agent with "Agent: x,y,z"
        font = pygame.font.SysFont('Arial', 12)
        text = font.render(f"Agent: {agent_pos.x:.2f}, {agent_pos.y:.2f}, {agent_pos.z:.2f}", True, (0, 0, 0))
        display.blit(text, (-100 + int(400 + agent_pos.x / 10), -25 + int(300 + agent_pos.y / 10)))
        last_action_str = "None"
        if self.last_action == 0:
            last_action_str = "Left"
        elif self.last_action == 1:
            last_action_str = "Right"
        elif self.last_action == 2:
            last_action_str = "Forward"
        elif self.last_action == 3:
            last_action_str = "Stop"
        text = font.render(f"Last action: {last_action_str}", True, (0, 0, 0))
        display.blit(text, (-120 + int(400 + agent_pos.x / 10), -5 + int(300 + agent_pos.y / 10)))

        # Draw the ball
        pygame.draw.circle(display, (0, 0, 255), (int(400 + ball_pos.x / 10), int(300 + ball_pos.y / 10)), 5)
        # Draw text above the ball with "Ball: x,y,z"
        text = font.render(f"Ball: {ball_pos.x:.2f}, {ball_pos.y:.2f}, {ball_pos.z:.2f}", True, (0, 0, 0))
        display.blit(text, (-100 + int(400 + ball_pos.x / 10), -25 + int(300 + ball_pos.y / 10)))


        # Draw the goal
        pygame.draw.circle(display, (0, 255, 0), (int(400 + self.goal_pos.x / 10), int(300 + self.goal_pos.y / 10)), 10)
        # Draw text above the goal with "Goal: x,y,z"
        text = font.render(f"Goal: {self.goal_pos.x:.2f}, {self.goal_pos.y:.2f}, {self.goal_pos.z:.2f}", True, (0, 0, 0))
        display.blit(text, (-100 + int(400 + self.goal_pos.x / 10), -25 + int(300 + self.goal_pos.y / 10)))

        obs_names = [
            "Agent x position",
            "Agent y position",
            "Agent z position",
            "Agent pitch",
            "Agent yaw",
            "Agent roll",
            "Agent boost",
            "Agent x velocity",
            "Agent y velocity",
            "Agent z velocity",
            "Agent pitch rate",
            "Agent yaw rate",
            "Agent roll rate",
            "Goal x position",
            "Goal y position",
            "Goal z position",
            "Last action",
            "Car->goal velocity"
        ]

        # Enumerate zip of obs and obs_names and print them out
        for i, (obs_name, obs_val) in enumerate(zip(obs_names, obs)):
            text = font.render(f"{obs_name}: {obs_val:.2f}", True, (0, 0, 0))
            display.blit(text, (10, 10 + i * 20))

        pygame.display.update()

        # Sleep for 16ms
        pygame.time.wait(16)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                import sys
                sys.exit()
