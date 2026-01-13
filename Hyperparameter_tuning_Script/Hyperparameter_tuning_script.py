import copy
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from keras_tuner import RandomSearch

from Env_ref import Env
from agent_ref import Agent
import agent_ref  # to set the global `env` used inside Agent.get_action


# ---------- 1. Environment factory ----------

NETWORK_FILE = "Python/simple_network.txt"
TRAIN_FILE = "Python/simple_train.txt"
MAX_DELAY = 600
CONFLICT_DELAY_CAP = 1200
EARLY_CAP = 120


def make_env():
    return Env(NETWORK_FILE, TRAIN_FILE, MAX_DELAY, CONFLICT_DELAY_CAP, EARLY_CAP)


# ---------- 2. RL training loop used during tuning ----------

def run_training(env, agent, max_episodes=30):
    """
    Shortened training loop for Keras Tuner.
    Stops when either:
      - we reach max_episodes, OR
      - epsilon has decayed below epsilon_min.
    Returns the average episode return over the last few episodes.
    """
    # Make this env visible inside agent_ref (Agent.get_action uses global `env` there)
    agent_ref.env = env

    global_step = 0
    n_blocks = len(env.tc_list)
    n_max_trains = len(env.train_ids)
    ITERATION_PER_EPISODE = n_max_trains * n_blocks * n_blocks

    episode_returns = []

    episode_idx = 0
    while episode_idx < max_episodes and agent.epsilon >= agent.epsilon_min:
        episode_idx += 1
        env.episode_idx = episode_idx

        state = env.reset()
        initial_obj = env.current_lp_cost
        done = False
        current_best = initial_obj
        episode_return = 0.0

        for _ in range(ITERATION_PER_EPISODE):
            global_step += 1
            action = agent.get_action(state, env.current_postion, env.next_pos)

            next_state, reward, is_terminal = env.step(action)
            episode_return += reward

            if is_terminal != 'Pass':
                agent.append_sample(state, action, reward, next_state, is_terminal)

            if is_terminal:
                done = True

            if global_step >= agent.train_start and global_step % agent.train_freq == 0:
                agent.train_model(global_step, done)
                if global_step % agent.update_freq == 0:
                    agent.update_target_model()

            if env.current_lp_cost < current_best:
                current_best = env.current_lp_cost

            state = copy.deepcopy(next_state)

            if done:
                break

        episode_returns.append(episode_return)

    if not episode_returns:
        # if something went wrong, return very bad score
        return -1e12

    tail_k = min(5, len(episode_returns))
    return float(np.mean(episode_returns[-tail_k:]))


# ---------- 3. HyperModel: defines the search space ----------

class RLHyperModel(kt.HyperModel):
    """
    HyperModel that defines the hyperparameter search space.
    The returned Keras model is a dummy; we don't actually train it.
    """

    def build(self, hp):
        # Define the hyperparameters we want to tune
        hp.Float("learning_rate", 1e-5, 1e-3, sampling="log")
        hp.Float("discount_factor", 0.90, 0.99)
        hp.Float("epsilon_decay", 0.999, 0.99999)
        hp.Choice("epsilon_min", [0.01, 0.05])
        hp.Choice("batch_size", [128, 256, 512])
        hp.Float("l2_reg", 1e-4, 1e-2, sampling="log")
        hp.Choice("train_start", [5000, 10000])
        hp.Choice("train_freq", [8, 16])
        hp.Choice("update_freq", [64, 128])
        hp.Int("max_episodes", min_value=20, max_value=60, step=20)

        # Dummy model just to satisfy Keras Tuner's interface
        inputs = tf.keras.Input(shape=(1,))
        outputs = tf.keras.layers.Dense(1)(inputs)
        model = tf.keras.Model(inputs, outputs)
        return model


# ---------- 4. Custom tuner: uses the RL loop instead of model.fit ----------

class RLTuner(RandomSearch):
    """
    RandomSearch tuner where we override run_trial
    to execute our RL loop instead of model.fit().
    """

    def run_trial(self, trial, *fit_args, **fit_kwargs):
        # Build hyperparameters and dummy model via the HyperModel
        hp = trial.hyperparameters
        model = self.hypermodel.build(hp)

        # Retrieve chosen hyperparameter values
        learning_rate = hp.get("learning_rate")
        discount_factor = hp.get("discount_factor")
        epsilon_decay = hp.get("epsilon_decay")
        epsilon_min = hp.get("epsilon_min")
        batch_size = hp.get("batch_size")
        l2_reg = hp.get("l2_reg")
        train_start = hp.get("train_start")
        train_freq = hp.get("train_freq")
        update_freq = hp.get("update_freq")
        max_episodes = hp.get("max_episodes")

        # Build Env + Agent with these hyperparameters
        env = make_env()
        n_blocks = len(env.tc_list)
        n_max_trains = len(env.train_ids)
        n_node = len(env.nodes)
        n_alterArcSet = len(env.alterArcSet)

        agent = Agent(
            n_blocks,
            n_max_trains,
            n_node,
            n_alterArcSet,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon_start=1.0,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            batch_size=batch_size,
            train_start=train_start,
            train_freq=train_freq,
            update_freq=update_freq,
            memory_size=1_000_000,
            l2_reg=l2_reg,
        )

        # Run RL training and compute score
        score = run_training(env, agent, max_episodes=max_episodes)

        # Report the objective value ("score") to the oracle
        self.oracle.update_trial(trial.trial_id, {"score": score})
        self.oracle.end_trial(trial.trial_id, status="COMPLETED")
        self.oracle.save()


# ---------- 5. Run the search ----------

def main():
    hypermodel = RLHyperModel()

    tuner = RLTuner(
        hypermodel=hypermodel,
        objective=kt.Objective("score", direction="max"),
        max_trials=10,  # be careful, RL is expensive
        directory="kt_logs",
        project_name="rl_rescheduling",
        overwrite=True,
    )

    tuner.search()

    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
    print("Best score:", best_trial.score)
    print("Best hyperparameters:")
    for name, value in best_trial.hyperparameters.values.items():
        print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
