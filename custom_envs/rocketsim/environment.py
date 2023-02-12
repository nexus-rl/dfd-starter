from rocketsim import Angle, Vec3
from rocketsim.sim import Arena, CarConfig, GameMode, Team, Car, CarControls, Ball

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
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(-1, 1, (14,))
        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((1000, 800))
        else:
            self.screen = None

    def step(self, action):
        if self.arena is None:
            raise Exception("Arena not initialized, you must call reset() before step()")

        # Controls are turn left, turn right, accelerate, and brake
        controls = CarControls(throttle=0)
        if action == 0:
            controls.throttle = 0.5
            controls.steer = -1
        elif action == 1:
            controls.throttle = 0.5
            controls.steer = 1
        elif action == 2:
            controls.throttle = 1
        elif action == 3:
            controls.throttle = -1

        self.arena.set_car_controls(self.agent_id, controls)
        self.arena.step(8)

        reward = self._reward_fn()
        done = self._distance_car_from_target(self.goal_pos) < GOAL_THRESHOLD
        self.timeout_ticks -= 1
        timeout = self.timeout_ticks <= 0
        self.last_action = action
        if self.screen:
            self.render()
        return self._form_obs(), reward, done, timeout, {}

    def _subtract_vec3(self, a: Vec3, b: Vec3):
        return np.array([a.x - b.x, a.y - b.y, a.z - b.z])

    def _distance_car_from_ball(self):
        agent = self._get_agent()
        ball = self._get_ball()
        return np.linalg.norm(self._subtract_vec3(agent.get_pos(), ball.get_pos()))

    def _distance_ball_from_target(self, target: Vec3):
        ball = self._get_ball()
        return np.linalg.norm(self._subtract_vec3(ball.get_pos(), target))

    def _distance_car_from_target(self, target: Vec3):
        agent = self._get_agent()
        return np.linalg.norm(self._subtract_vec3(agent.get_pos(), target))

    def _reward_fn(self):
        # car_ball_dist = self._distance_car_from_ball() / 10000
        # ball_target_dist = self._distance_ball_from_target(GOAL) / 10000
        car_target_dist = self._distance_car_from_target(self.goal_pos) / 10000
        return -car_target_dist #- ball_target_dist

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        if self.arena is None:
            self.arena = Arena(GameMode.Soccar)
            self.agent_id = self.arena.add_car(Team.Blue, CarConfig.Octane)

        self.timeout_ticks = 200
        self.last_action = None

        ball = self._get_ball()
        ball.pos = Vec3(0, 0, 5000)
        # ball.vel = Vec3(0, 0, 0.5)
        self.arena.ball = ball

        agent = self._get_agent()
        # x = +- 3000
        # y = +- 3000
        agent.pos = Vec3(self.rng.uniform(-3000, 3000), self.rng.uniform(-3000, 3000), 17)
        agent.angles = Angle(0, 0, 0)
        agent.boost = 100

        self.goal_pos = Vec3(self.rng.uniform(-3000, 3000), self.rng.uniform(-3000, 3000), 17)
        self.arena.set_car(self.agent_id, agent)

        return self._form_obs(), {}

    def _get_agent(self) -> Car:
        return self.arena.get_car(self.agent_id)

    def _get_ball(self) -> Ball:
        return self.arena.get_ball()

    def _form_obs(self):
        # OBS:
        # 1. Agent x position
        # 2. Agent y position
        # 3. Agent z position
        # 4. Agent pitch
        # 5. Agent yaw
        # 6. Agent roll
        # 7. Agent boost
        # 8. Agent x velocity
        # 9. Agent y velocity
        # 10. Agent z velocity
        # 11. Ball x position
        # 12. Ball y position
        # 13. Ball z position
        # 14. Ball x velocity
        # 15. Ball y velocity
        # 16. Ball z velocity
        # 17. Last action

        POS_SCALE = 1/5000
        ANGLE_SCALE = 1/3.14159
        BOOST_SCALE = 1/100
        VEL_SCALE = 1/5000
        ACTION_SCALE = 1/3

        agent = self._get_agent()
        # ball = self._get_ball()
        agent_pos = agent.get_pos()
        agent_vel = agent.get_vel()
        agent_angles = agent.get_angles()
        # ball_pos = ball.get_pos()
        # ball_vel = ball.get_vel()

        obs = [
            agent_pos.x * POS_SCALE,
            agent_pos.y * POS_SCALE,
            agent_pos.z * POS_SCALE,
            agent_angles.pitch * ANGLE_SCALE,
            agent_angles.yaw * ANGLE_SCALE,
            agent_angles.roll * ANGLE_SCALE,
            agent.boost * BOOST_SCALE,
            agent_vel.x * VEL_SCALE,
            agent_vel.y * VEL_SCALE,
            agent_vel.z * VEL_SCALE,
            self.goal_pos.x * POS_SCALE,
            self.goal_pos.y * POS_SCALE,
            self.goal_pos.z * POS_SCALE,
            -1 if self.last_action is None else (self.last_action * ACTION_SCALE)
        ]

        return np.asarray(obs, dtype=np.float32)

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
            "Goal x position",
            "Goal y position",
            "Goal z position",
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