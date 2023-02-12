from custom_envs import simple_trap_env
from run_sequential import SequentialRunner
from dsgd import DSGD
import torch


if __name__ == "__main__":
    cfg = {
        "seed": 124,
        "grad_opt": DSGD,
        "env_id": "Walker2d-v4",
        "wandb": {"log": True,
                  "run_name": "rng_test"},

        "optimizer": {"learning_rate": 0.01,
                      "noise_std": 0.02,
                      "batch_size": 40,
                      "ent_coef": 0,
                      "max_delayed_return": 10,
                      "vbn_buffer_size": 0,
                      "normalize_obs": True,
                      "eval_prob": 0.05},

        "strategy": {"zeta_size": 2,
                     "max_history_size": 2},

        "omega": {"max_value": 1,
                  "min_value": 0,
                  "default_value": 0,
                  "steps_to_min": 25,
                  "steps_to_max": 75,
                  "improvement_threshold": 1.035,
                  "window_size": 20}}

    runner = SequentialRunner(log_to_wandb=True,
                              opt_fn=DSGD,
                              env_id="Walker2d-v4",
                              wandb_run_name="rng_test",
                              normalize_obs=True,
                              learning_rate=0.01,
                              noise_std=0.02,
                              batch_size=40,
                              ent_coef=0.,
                              random_seed=124,
                              max_delayed_return=0,
                              vbn_buffer_size=1000,
                              zeta_size=2,
                              max_strategy_history_size=2,
                              eval_prob=0.05,
                              omega_improvement_threshold=1.035,
                              omega_reward_history_size=20,
                              omega_default_value=0.0,
                              omega_min_value=0.0,
                              omega_max_value=1,
                              omega_steps_to_min=25,
                              omega_steps_to_max=75,
                              )

    runner.train(n_epochs=1000000)
