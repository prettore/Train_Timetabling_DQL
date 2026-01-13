# DQN_ag_test.py  -- evaluation + timetable dumping

import copy
import os
import csv

import agent_ref
import AG_ref as ag
from Env_ref import Env
from agent_ref import Agent


def snapshot_nodes(nodes):
   
    snap = {}

    for n in nodes:
        node_id = getattr(n, "node_id", None)
        if node_id is None:
            # if for some reason node_id doesn't exist, skip the node
            continue

        train_id = getattr(n, "train_id", getattr(n, "train", -1))
        block    = getattr(n, "block", -1)
        arr_t    = getattr(n, "arr_time", 0.0)
        dep_t    = getattr(n, "dep_time", 0.0)

        snap[node_id] = {
            "node_id":  node_id,
            "train_id": train_id,
            "block":    block,
            "arr_time": arr_t,
            "dep_time": dep_t,
        }

    return snap

def sec_to_hms_str(t):
    """Convert seconds-from-midnight to HH:MM:SS string."""
    t = int(round(t))
    h = t // 3600
    m = (t % 3600) // 60
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def dump_episode_timetable(episode_idx, tt_in, tt_out,
                           out_dir="episode_timetables"):
  
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"episode_{episode_idx:05d}.csv")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode",
                "node_id",
                "train_id",
                "block",
                "init_arr_time",
                "init_dep_time",
                "final_arr_time",
                "final_dep_time",
                "arr_delay_sec",
                "dep_delay_sec",
            ]
        )

        for node_id, init_row in tt_in.items():
            final_row = tt_out.get(node_id, init_row)

            init_arr = init_row["arr_time"]
            init_dep = init_row["dep_time"]
            fin_arr  = final_row["arr_time"]
            fin_dep  = final_row["dep_time"]

            arr_delay = fin_arr - init_arr
            dep_delay = fin_dep - init_dep

            writer.writerow(
                [
                    episode_idx,
                    node_id,
                    init_row["train_id"],
                    init_row["block"],
                    sec_to_hms_str(init_arr),
                    sec_to_hms_str(init_dep),
                    sec_to_hms_str(fin_arr),
                    sec_to_hms_str(fin_dep),
                    arr_delay,
                    dep_delay,
                ]
            )

    print(f"[dump] Wrote timetable for episode {episode_idx} to {out_path}")



if __name__ == "__main__":
    
    network_file = "Python/simple_network.txt"
    train_file   = "Python/simple_train_case5.txt"

    max_delay          = 100
    conflict_delay_cap = 120
    early_cap          = 120

    
    load_file = "./save_model_softearlycap_epsilon_strict/20220402_DQN_30_3_80000.weights.h5"

 
    global env
    env = Env(network_file, train_file, max_delay, conflict_delay_cap, early_cap, training=False)
    agent_ref.env = env  

    n_blocks      = len(env.tc_list)
    n_max_trains  = len(env.train_ids)
    n_node        = len(env.nodes)
    n_alterArcSet = len(env.alterArcSet)

  
    agent = agent_ref.Agent(
        n_blocks,
        n_max_trains,
        n_node,
        n_alterArcSet,
        load_file=load_file,
    )

    # Pure evaluation: no exploration, no learning
    agent.epsilon     = 0.0
    agent.epsilon_min = 0.0

   
    EPISODES = 1
    # A loose upper bound so we don’t loop forever
    ITERATION_PER_EPISODE = n_max_trains * n_blocks * n_blocks

    for e in range(1, EPISODES + 1):
        state = env.reset()

        timetable_in = snapshot_nodes(env.nodes)

        done = False
        total_reward = 0.0
        steps = 0

        while not done and steps < ITERATION_PER_EPISODE:
            steps += 1

            # Agent.get_action needs state, current_postion, next_pos
            action = agent.get_action(state, env.current_postion, env.next_pos)

            next_state, reward, done = env.step(action)
            total_reward += reward

            state = copy.deepcopy(next_state)

        # --- snapshot result timetable at end of episode ---
        ag.results(env.ag_graph, env.nodes)

        timetable_out = snapshot_nodes(env.nodes)

        dump_episode_timetable(e, timetable_in, timetable_out)

        # Use the metrics your Env tracks
        final_obj         = getattr(env, "current_lp_cost", 0.0)
        max_end_delay     = getattr(env, "last_max_train_end_delay", 0.0)
        max_train_end_early   = getattr(env, "last_max_train_end_early", 0.0)
        max_conflict_incr = getattr(env, "last_max_conflict_increment", 0.0)
        max_early_incr    = getattr(env, "last_max_early_increment", 0.0)
        episode_return    = getattr(env, "last_episode_return", total_reward)

        print(
            f"Episode {e:3d} | "
            f"steps={steps:4d} | "
            f"return={episode_return:10.3f} | "
            f"final_obj={final_obj:10.3f} | "
            f"maxEndDelay={max_end_delay:8.2f} | "
            f"maxTrainEndEarly={max_train_end_early:8.2f} | "
            f"maxConflictIncr={max_conflict_incr:8.2f} | "
            f"maxEarlyIncr={max_early_incr:8.2f}"
        )
