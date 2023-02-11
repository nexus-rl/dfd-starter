from rocketsim import Angle, Vec3
from rocketsim.sim import Arena, CarConfig, GameMode, Team, Car, CarControls, Ball

import numpy as np
import os
import gym
import os

GOAL = Vec3(0, 5000, 93)
GOAL_THRESHOLD = 100

class Environment(gym.Env):
    def __init__(self, opt_id="fd_worker_{}".format(os.getpid()), render_mode=None, **kwargs):
        self.opt_id = opt_id
        self.rng = np.random.RandomState(0)
        self.arena = None
        self.agent_id = None
        self.last_action = None
        self.timeout_ticks = None
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(-1, 1, (17,))
        self.render_mode = render_mode

    def step(self, action):
        if self.arena is None:
            raise Exception("Arena not initialized, you must call reset() before step()")

        # Controls are turn left, turn right, accelerate, and brake
        controls = CarControls(throttle=0.5)
        if action == 0:
            controls.steer = -1
        elif action == 1:
            controls.steer = 1
        elif action == 2:
            controls.throttle = 1
        elif action == 3:
            controls.throttle = 0

        self.arena.set_car_controls(self.agent_id, controls)
        self.arena.step(8)

        reward = self._reward_fn()
        done = self._distance_ball_from_target(GOAL) < GOAL_THRESHOLD
        self.timeout_ticks -= 1
        timeout = self.timeout_ticks <= 0
        self.last_action = action
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

    def _reward_fn(self):
        car_ball_dist = self._distance_car_from_ball() / 10000
        ball_target_dist = self._distance_ball_from_target(GOAL) / 10000
        return -car_ball_dist - ball_target_dist

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        if self.arena is None:
            self.arena = Arena(GameMode.Soccar)
            self.agent_id = self.arena.add_car(Team.Blue, CarConfig.Octane)

        self.timeout_ticks = 3000
        self.last_action = None

        ball = self._get_ball()
        ball.pos = Vec3(0, 0, 93)
        self.arena.ball = ball

        agent = self._get_agent()
        agent.pos = Vec3(100, 100, 17)
        agent.angles = Angle(0, 0, 0)
        agent.boost = 100
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
        ball = self._get_ball()
        agent_pos = agent.get_pos()
        agent_vel = agent.get_vel()
        agent_angles = agent.get_angles()
        ball_pos = ball.get_pos()
        ball_vel = ball.get_vel()

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
            ball_pos.x * POS_SCALE,
            ball_pos.y * POS_SCALE,
            ball_pos.z * POS_SCALE,
            ball_vel.x * VEL_SCALE,
            ball_vel.y * VEL_SCALE,
            ball_vel.z * VEL_SCALE,
            -1 if self.last_action is None else (self.last_action * ACTION_SCALE)
        ]

        return np.asarray(obs, dtype=np.float32)

    def close(self):
        pass

    def render(self):
        pass
