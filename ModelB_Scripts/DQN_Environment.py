import sys
import numpy as np
import random
import copy

import gen_data_ref as gen
import AG_ref as ag
import networkx as nx


class Env():
    cnt_in_episode = 0
    current_lp_cost = -1
    bigM = 99999

    def __init__(self, network_file, train_file, max_delay, conflict_delay_cap, early_cap):
        self.p_graph, self.tc_list = gen.gen_p_graph(network_file)
        self.n_blocks = nx.number_of_nodes(self.p_graph)

        self.nodes, self.nodeMap, self.train_ids, self.n_operations  = gen.gen_nodes(train_file)

        self.n_max_trains = len(self.train_ids)
        self.n_trains = len(self.train_ids)
        self.max_delay = max_delay

        self.num_nodes = len(self.nodes)
        self.train_nodes = gen.gen_train_nodes(self.n_trains, self.nodes)
        self.current_postion = self.cal_postion(self.train_nodes)
        self.next_pos = self.next_postion(self.train_nodes)
        self.train_file = train_file

        self.ag_graph = gen.gen_ag_graph(self.nodes)
        self.alterArcSet = gen.gen_alter_arcs2(self.p_graph, self.nodes)
        self.alterArcSet = ag.fixed_alter_arcs2(self.ag_graph, self.alterArcSet)
        ag.print_alter_arcs(self.alterArcSet)
        self.node_labels = {}
        self.pos = {}
        for i in range(len(self.nodes)):
            node = self.nodes[i]
            self.pos[i] = [node.pos_x, node.pos_y]
            self.node_labels[i] = node.name

        self.mip = 0
        self.n_Feasible = 0
        self.n_Fail = 0
        self.n_Optimal = 0
                # --- NEW: separate cap for conflict-resolution delay
        # Backward compatible: if not provided, fall back to max_delay
        self.conflict_delay_cap = conflict_delay_cap if conflict_delay_cap is not None else max_delay
        self.early_cap = early_cap if early_cap is not None else 60.0  # e.g., 60s of allowed earliness per block


        # --- reward shaping hyperparameters 
        self.w_obj      = 1.0   # improvement in -cost
        self.w_mono     = 10.0   # penalty for end_delay > start_delay
        self.w_cap      = 50.0   # penalty for exceeding conflict cap
        self.w_early    = 50.0
        self.step_pen   = 0.001 # tiny step penalty to finish faster
        self.norm_delay = max(1.0, float(self.conflict_delay_cap))  # scaling


    def cal_postion(self, train_nodes):
        #print(train_nodes)
        postion = []
        for i in range(self.n_trains):
            #print(self.nodes[train_nodes[i][0]].block)
            if len(train_nodes[i]) == 0:
                postion.append(0)
            else :
                postion.append(self.nodes[train_nodes[i][0]].block)
        return postion

    def total_postion(self, train_nodes):
        # print(train_nodes)
        total_pos = np.zeros(self.n_blocks)
        for i in range(self.n_trains):
            if len(train_nodes[i]) != 0:
                for j in range(len(train_nodes[i])):
                    tc_index = self.tc_list.index(self.nodes[train_nodes[i][j]].block)
                    total_pos[tc_index] += 1
        #print(total_pos)
        total_pos = total_pos / self.n_trains
        return total_pos

    def next_postion(self, train_nodes):
        #print(train_nodes)
        postion = []
        for i in range(self.n_trains):
            #print(self.nodes[train_nodes[i][0]].block)
            if len(train_nodes[i]) - 1 <= 0:
                postion.append(0)
            else : postion.append(self.nodes[train_nodes[i][1]].block)
        return postion

    def direction_postion(self, train_nodes):
        up_direction = np.zeros(self.n_blocks)
        dn_direction = np.zeros(self.n_blocks)
        for i in range(self.n_trains):
            if len(train_nodes[i]) != 0:
                tc_index = self.tc_list.index(self.nodes[train_nodes[i][0]].block)
                if self.nodes[train_nodes[i][0]].direction == 1:
                    up_direction[tc_index] = 1
                if self.nodes[train_nodes[i][0]].direction == 2:
                    dn_direction[tc_index] = 1
        #print(total_pos)
        return (up_direction, dn_direction)


    def reset(self):
        # self.n_trains = np.random.randint(2, self.n_max_trains+1)
        self.n_trains = self.n_max_trains
        self.train_list = random.sample(self.train_ids, self.n_trains)
        self.cnt_in_episode = 0

        self.nodes, self.nodeMap = gen.gen_nodes1(self.train_file, self.train_list)
        self.num_nodes = len(self.nodes)
        self.nodes = gen.add_entry_delay(self.nodes, self.max_delay, 100)
        self.train_nodes = gen.gen_train_nodes(self.n_trains, self.nodes)
        self.current_postion = self.cal_postion(self.train_nodes)
        self.next_pos = self.next_postion(self.train_nodes)
        self.ag_graph = gen.gen_ag_graph(self.nodes)
        self.alterArcSet = gen.gen_alter_arcs2(self.p_graph, self.nodes)
        self.alterArcSet = ag.fixed_alter_arcs2(self.ag_graph, self.alterArcSet)

        self.node_labels = {}
        self.pos = {}
        for i in range(len(self.nodes)):
            node = self.nodes[i]
            self.pos[i] = [node.pos_x, node.pos_y]
            self.node_labels[i] = node.name

        
        self.current_lp_cost = -1 * ag.cost(self.ag_graph)
        self.mip = 0
        self.prev_obj = self.current_lp_cost

        # caches for monotonicity baseline
        s0, e0 = ag.first_last_delays(self.ag_graph, self.nodes)
        self.prev_starts = s0 if s0 is not None else []
        self.prev_ends   = e0 if e0 is not None else []

        self.max_conflict_increment = 0.0         # episode max per-block increment over all trains
        self.last_max_conflict_increment = 0.0
        self.last_terminal_reward = 0.0
        self.max_train_end_delay = 0.0              # episode max of final (end) delay across trains
        self.last_max_train_end_delay = 0.0  # stash for logging on terminal
        self.episode_return = 0.0
        self.last_episode_return = 0.0
        self.max_early_increment = 0.0
        self.last_max_early_increment = 0.0



        self.current_state = self.ag_to_state()
        return self.current_state


    def ag_to_state(self):
        self.current_postion = self.cal_postion(self.train_nodes)
        self.next_pos = self.next_postion(self.train_nodes)

        #print(self.current_postion)
        states0 = nx.to_numpy_array(self.p_graph) 
        states1 = np.zeros(self.n_blocks) #Block
        for i in self.current_postion:
            if i == 0: continue  # current_postion
            tc_index = self.tc_list.index(i)
            states1[tc_index] = 1

        states3 = np.zeros(self.n_blocks) # Block 
        states4 = np.zeros(self.n_blocks)  # Block 
        states10 = np.zeros(self.n_blocks)  # AG 
        states11 = np.zeros(self.n_blocks)  # AG 

        entry_times = nx.to_numpy_array(self.ag_graph)[0]
        min_time = np.min(-1*np.delete(entry_times, np.where(entry_times == 0)))
        #print("entry_times: ",entry_times)
        #print("min_entry_time: ", min_time)

        node_indexs = []
        for i in range(len(self.current_postion)):
            if self.current_postion[i] == 0 :
                node_index = nx.number_of_nodes(self.ag_graph)-1
            else: node_index = self.train_nodes[i][0]
            node_indexs.append(node_index)

        #print("node_indexs: ",node_indexs)
        tc_indexs= []
        if nx.negative_edge_cycle(self.ag_graph) == False:
            longest_paths = nx.single_source_bellman_ford_path_length(self.ag_graph, source=0)

            for i in range(len(self.current_postion)):
                n = self.nodes[node_indexs[i]]
                if self.current_postion[i] == 0: continue # current_postion
                tc_index = self.tc_list.index(self.current_postion[i])
                if tc_index in tc_indexs: continue
                states10[tc_index] = self.ag_graph.in_degree(n.node_id)
                states11[tc_index] = self.ag_graph.out_degree(n.node_id)
                states3[tc_index] = -1 * longest_paths[n.node_id] - min_time
                if n.next_fixed_node != -1:
                    states4[tc_index] = -1 * longest_paths[n.next_fixed_node] - min_time
                else: states4[tc_index] = -1 * longest_paths[n.node_id] + (n.dep_time-n.arr_time) - min_time
                tc_indexs.append(tc_index)

        else:
            for i in range(len(self.current_postion)):
                if self.current_postion[i] == 0: continue
                tc_index = self.tc_list.index(self.current_postion[i])
                if tc_index in tc_indexs: continue
                states10[tc_index] = 0
                states11[tc_index] = 0
                states3[tc_index] = 0
                states4[tc_index] = 0
                tc_indexs.append(tc_index)

        states3 = states3 / max(1, np.max(states3), np.max(states4))
        states4 = states4 / max(1, np.max(states3), np.max(states4))
        states10 = states10 / max(1, np.max(states10), np.max(states11))
        states11 = states11 / max(1, np.max(states11), np.max(states11))

        states5 = np.zeros(self.n_blocks)  # Next Block
        for i in self.next_postion(self.train_nodes):
            if i == 0 : continue #next postion
            tc_index = self.tc_list.index(i)
            states5[tc_index] = 1

        states6 = np.zeros(self.n_blocks) 
        degree = nx.degree(self.p_graph)
        tc = nx.nodes(self.p_graph)
        for i in tc:
            if degree[i] > 4:
                states6[self.tc_list.index(i)] = 1

        states8, states9 = self.direction_postion(self.train_nodes)

        states = np.append(self.n_blocks, states0)
        states = np.append(states, states1)
        states = np.append(states, states3)
        states = np.append(states, states4)
        states = np.append(states, states5)
        states = np.append(states, states6)
        states = np.append(states, states8)
        states = np.append(states, states9)
        states = np.append(states, states10)
        states = np.append(states, states11)

        if self.current_lp_cost != self.bigM:
            states = np.append(states, self.current_lp_cost/self.bigM)
        else : states = np.append(states, 0)
        return states

    def step(self, action):
        # advance step counter
        self.cnt_in_episode += 1

        # ---------- already-failed episode guard (terminal)
        if self.current_lp_cost == self.bigM:
            self.n_Fail += 1
            print("Fail1!")
            terminal_reward = -1 * self.bigM
            # stash episode-end metrics
            if hasattr(self, "last_max_train_end_delay"):
                self.last_max_train_end_delay = getattr(self, "max_train_end_delay", 0.0)
            if hasattr(self, "last_max_conflict_increment"):
                self.last_max_conflict_increment = getattr(self, "max_conflict_increment", 0.0)
            if hasattr(self, "last_max_early_increment"):
                self.last_max_early_increment = getattr(self, "max_early_increment", 0.0)
            if hasattr(self, "episode_return"):
                self.last_episode_return = self.episode_return + terminal_reward
            return (self.current_state, terminal_reward, True)

        # ---------- Finish action (terminal)
        if action == self.n_blocks:
            print("Finish !!!")
            # Terminal reward = NEGATIVE of the cost (maximize -cost)
            terminal_reward = -1.0 * self.current_lp_cost  # == -ag.cost(self.ag_graph)

            if self.current_lp_cost == self.bigM:
                self.n_Fail += 1
                print("Fail2!")
                terminal_reward = -1 * self.bigM

            if self.current_lp_cost != self.bigM and ag.is_select_all_alter_arcs(self.alterArcSet) == True:
                self.n_Feasible += 1
                print("Feasible!", self.mip, self.current_lp_cost)
                if int(self.mip) == self.current_lp_cost:
                    self.n_Optimal += 1
                    print("Optimal!")

            # stash episode-end metrics
            if hasattr(self, "last_max_train_end_delay"):
                self.last_max_train_end_delay = getattr(self, "max_train_end_delay", 0.0)
            if hasattr(self, "last_max_conflict_increment"):
                self.last_max_conflict_increment = getattr(self, "max_conflict_increment", 0.0)
            if hasattr(self, "last_max_early_increment"):
                self.last_max_early_increment = getattr(self, "max_early_increment", 0.0)
            if hasattr(self, "episode_return"):
                self.last_episode_return = self.episode_return + terminal_reward
            return (self.current_state, terminal_reward, True)

        # ---------- Normal action path
        block = self.tc_list[action]
        avail_times, avail_trains = [], []
        train_index = -1

        for i in range(len(self.current_postion)):
            if block == self.current_postion[i]:
                avail_times.append(self.nodes[self.train_nodes[i][0]].arr_time)
                avail_trains.append(i)

        if len(avail_times) > 0:
            train_index = avail_trains[np.argmin(avail_times)]

        # No train available for this block -> small non-terminal penalty
        if train_index == -1:
            r = -1 * self.cnt_in_episode
            if hasattr(self, "episode_return"):
                self.episode_return += r
            return (self.current_state, r, False)

        node_index = self.train_nodes[train_index][0]
        node_alter_set = ag.node_alter_arcs(self.alterArcSet, node_index)

        # Ensure alter arcs are present once
        for alterArcInfo in node_alter_set:
            if alterArcInfo.node_j.node_id == node_index and alterArcInfo.isSet != True:
                self.ag_graph.add_edge(alterArcInfo.node_s_j.node_id, alterArcInfo.node_i.node_id)
                self.ag_graph[alterArcInfo.node_s_j.node_id][alterArcInfo.node_i.node_id]['weight'] = -120
                self.ag_graph[alterArcInfo.node_s_j.node_id][alterArcInfo.node_i.node_id]['type'] = 'Alter'
                alterArcInfo.isSet = True
            if alterArcInfo.node_i.node_id == node_index and alterArcInfo.isSet != True:
                self.ag_graph.add_edge(alterArcInfo.node_s_i.node_id, alterArcInfo.node_j.node_id)
                self.ag_graph[alterArcInfo.node_s_i.node_id][alterArcInfo.node_j.node_id]['weight'] = -120
                self.ag_graph[alterArcInfo.node_s_i.node_id][alterArcInfo.node_j.node_id]['type'] = 'Alter'
                alterArcInfo.isSet = True

        # consume this node for the chosen train
        del self.train_nodes[train_index][0]

        # ---------- Objective before/after (good sign: higher is better)
        old_obj = self.current_lp_cost
        self.current_lp_cost = -1 * ag.cost(self.ag_graph)
        new_obj = self.current_lp_cost

        # base dense reward = improvement (positive if delay decreased)
        r = self.w_obj * (old_obj - new_obj) / max(1.0, abs(old_obj))

        # ---------- Delays & constraints

        # (a) First/last delays -> monotonicity
        s_delays, e_delays = ag.first_last_delays(self.ag_graph, self.nodes)
        if s_delays is None:
            # negative cycle: terminate hard
            terminal_reward = -1 * self.bigM
            if hasattr(self, "last_max_train_end_delay"):
                self.last_max_train_end_delay = getattr(self, "max_train_end_delay", 0.0)
            if hasattr(self, "last_max_conflict_increment"):
                self.last_max_conflict_increment = getattr(self, "max_conflict_increment", 0.0)
            if hasattr(self, "last_max_early_increment"):
                self.last_max_early_increment = getattr(self, "max_early_increment", 0.0)
            if hasattr(self, "episode_return"):
                self.last_episode_return = self.episode_return + terminal_reward
            return (self.current_state, terminal_reward, True)

        # Track timetable-style Max delay (max final delay across trains, clamped ≥ 0)
        if hasattr(self, "max_train_end_delay"):
            current_max_end = max(e_delays) if e_delays else 0.0
            current_max_end = max(0.0, float(current_max_end))
            if current_max_end > self.max_train_end_delay:
                self.max_train_end_delay = current_max_end

        # (1) Monotonicity penalty: end must not exceed start
        mono_sum = sum(max(0.0, e - s) for s, e in zip(s_delays, e_delays))
        r -= self.w_mono * (mono_sum / self.norm_delay)

        # (b) Per-block increments -> lateness & earliness caps (SOFT)
        res = ag.per_train_delay_increments_both(self.ag_graph, self.nodes)
        if res[0] is None:
            # negative cycle: terminate hard
            terminal_reward = -1 * self.bigM
            if hasattr(self, "last_max_train_end_delay"):
                self.last_max_train_end_delay = getattr(self, "max_train_end_delay", 0.0)
            if hasattr(self, "last_max_conflict_increment"):
                self.last_max_conflict_increment = getattr(self, "max_conflict_increment", 0.0)
            if hasattr(self, "last_max_early_increment"):
                self.last_max_early_increment = getattr(self, "max_early_increment", 0.0)
            if hasattr(self, "episode_return"):
                self.last_episode_return = self.episode_return + terminal_reward
            return (self.current_state, terminal_reward, True)

        (starts, ends,
        max_incr_by_train, max_incr_global,
        max_early_by_train, max_early_global) = res

        # track episode maxima (optional, useful for plots)
        if hasattr(self, "max_conflict_increment"):
            if max_incr_global > self.max_conflict_increment:
                self.max_conflict_increment = max_incr_global
        if hasattr(self, "max_early_increment"):
            if max_early_global > self.max_early_increment:
                self.max_early_increment = max_early_global

        # lateness soft-cap penalty (unchanged)
        over_cap = max(0.0, max_incr_global - float(self.conflict_delay_cap))
        r -= self.w_cap * (over_cap / max(1.0, float(self.conflict_delay_cap)))

        # NEW: earliness soft-cap penalty (symmetric)
        cap_early = getattr(self, "early_cap", 60.0)   # default if not set in __init__
        over_early = max(0.0, max_early_global - float(cap_early))
        r -= self.w_early * (over_early / max(1.0, float(cap_early)))

        # small per-step penalty
        r -= self.step_pen

        # accumulate episode return (non-terminal step)
        if hasattr(self, "episode_return"):
            self.episode_return += r

        # trackers (optional)
        self.prev_obj = new_obj
        self.prev_starts, self.prev_ends = s_delays, e_delays

        # normal step returns the new state vector
        return (self.ag_to_state(), r, False)
