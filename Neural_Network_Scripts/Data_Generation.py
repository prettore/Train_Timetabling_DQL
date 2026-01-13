import sys
import numpy as np
import pickle
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random
import time
import sys, os
print("=== MODULE LOADED ===", __name__, "->", __file__)
sys.stdout.flush()


class alterArcInfo():
    def __init__(self, node_s_j, node_i, node_s_i, node_j):
        self.node_s_j = node_s_j
        self.node_i = node_i
        self.node_s_i = node_s_i
        self.node_j = node_j
        self.isSet = False
        self.cost1 = -1
        self.cost2 = -1

    def __str__(self):
        output = "Alter arc 1: " + str(self.node_s_j) + "-->" + str(self.node_i) + " || arc 2: " + str(self.node_s_i) + "-->" + str(self.node_j)
        return output


class nodeInfo():
    def __init__(self, node_id, train_id, direction, ord, block, arr_time, dep_time):
        self.node_id = node_id
        self.train_id = train_id
        self.direction= direction
        self.ord = ord
        self.block = block
        self.arr_time = arr_time
        self.dep_time = dep_time
        self.name = train_id + "-"+ str(block)
        self.next_fixed_node = -1
        self.prev_fixed_node = -1
        self.pos_x = 0
        self.pos_y = 0
        self.typ = ""  # will be set later

    def __str__(self):
        output = str(self.name)
        return output

def add_entry_delay(nodes, max_delay, event_pro):
    entry_delay = 0
    for i in range(len(nodes)):
        node = nodes[i]
        if node.ord == 1:
            if np.random.randint(0, 100) < event_pro:
                entry_delay = np.random.randint(0, max_delay)  
                #entry_delay = 0
            #print(entry_delay)
        node.arr_time += entry_delay
        node.dep_time += entry_delay
        #print(node.arr_time)
        #print(node.dep_time)
    return nodes

def gen_nodes(train_file):
    with open(train_file, mode="r") as f:
        lines = f.readlines()

    nodes = []
    node_labels = {}
    nodeMap = {}
    train_ids = []
    node = nodeInfo(0, 'start', 0, -1, 0, 0, 0) 
    node_labels[0] = node.name
    nodes.append(node)
    nodeMap[node.name] = node.node_id
    numNode = 1

    for line_num in range(len(lines)):
        line = lines[line_num].strip()

        # skip empty lines
        if not line:
            continue

        # split on ANY whitespace (tabs or spaces)
        strs = line.split()

        # expected format:
        # train_id direction ord block arr_time dep_time
        if len(strs) < 6:
            raise ValueError(f"Malformed line {line_num}: {line}")

        train_id = strs[0].strip()
        if train_id not in train_ids:
            train_ids.append(train_id)

        direction = int(strs[1])
        ord = int(strs[2])
        block = strs[3]

        # arrival time (HH:MM:SS)
        arr = strs[4]
        h, m, s = arr.split(':')
        arr_time = int(h) * 3600 + int(m) * 60 + int(s)

        # departure time (HH:MM:SS)
        dep = strs[5]
        h, m, s = dep.split(':')
        dep_time = int(h) * 3600 + int(m) * 60 + int(s)

        node = nodeInfo(numNode, train_id, direction, ord, block, arr_time, dep_time)
        node_labels[numNode] = node.name
        nodes.append(node)
        nodeMap[node.name] = node.node_id
        numNode += 1


    print(train_ids)
    node = nodeInfo(numNode, 'end', 0, -1, 0, 0, 0) 
    nodes.append(node)
    nodeMap[node.name] = node.node_id

    return (nodes, nodeMap, train_ids, len(lines))

def gen_nodes1(train_file, train_list):
    with open(train_file, mode="r") as f:
        lines = f.readlines()

    train_list.append('start')
    train_list.append('end')

    nodes = []
    node_labels = {}
    nodeMap = {}
    node = nodeInfo(0, 'start', 0, -1, 0, 0, 0) 
    node_labels[0] = node.name
    nodes.append(node)
    nodeMap[node.name] = node.node_id
    numNode = 1

    for line_num in range(len(lines)):
        line = lines[line_num].strip()

        # skip empty lines
        if not line:
            continue

        # split on ANY whitespace (tabs or spaces)
        strs = line.split()

        if len(strs) < 6:
            raise ValueError(f"Malformed line {line_num}: {line}")

        train_id = strs[0].strip()
        direction = int(strs[1])
        ord = int(strs[2])
        block = strs[3]

        # arrival time HH:MM:SS
        arr = strs[4]
        h, m, s = arr.split(':')
        arr_time = int(h) * 3600 + int(m) * 60 + int(s)

        # departure time HH:MM:SS
        dep = strs[5]
        h, m, s = dep.split(':')
        dep_time = int(h) * 3600 + int(m) * 60 + int(s)

        # filter trains
        if train_id not in train_list:
            continue

        node = nodeInfo(numNode, train_id, direction, ord, block, arr_time, dep_time)
        node_labels[numNode] = node.name
        nodes.append(node)
        nodeMap[node.name] = node.node_id
        numNode += 1



    node = nodeInfo(numNode, 'end', 0, -1, 0, 0, 0) 
    nodes.append(node)
    nodeMap[node.name] = node.node_id

    return (nodes, nodeMap)

def gen_p_graph(network_file):
    tc_list = []
    p_graph = nx.DiGraph()
    with open(network_file, mode="r") as f:
        lines = f.readlines()

    for line_num in range(len(lines)):
        strs = lines[line_num].split('\t')
        from_block = strs[0]
        to_block = strs[1]
        direction = int(strs[2])
        distance = int(strs[3])
        p_graph.add_edge(from_block, to_block)
        p_graph[from_block][to_block]['direction'] = direction
        p_graph[from_block][to_block]['weight'] = distance

        if from_block not in tc_list:
            tc_list.append(from_block)

        if to_block not in tc_list:
            tc_list.append(to_block)

    print(tc_list)
    return [p_graph, tc_list]




def build_block_type_map(segment_pairs_file):
   
    seg = pd.read_csv(segment_pairs_file, sep="\t")
    block_type_map = {}
    for _, row in seg.iterrows():
        seg1_block = str(row['Segment1_ID'])
        seg1_type = str(row['Segment1_Type']).split('-')[0]
        block_type_map[seg1_block] = seg1_type

        seg2_block = str(row['Segment2_ID'])
        seg2_type = str(row['Segment2_Type']).split('-')[0]
        block_type_map[seg2_block] = seg2_type
    return block_type_map

def build_train_block_type_map(filepath):
   
    lookup = {}
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue
            train = parts[0].strip()
            block = parts[3].strip()
            segtype = parts[4].strip().lower()
            lookup[(train, block)] = segtype
    return lookup


def gen_train_nodes(n_trains, nodes):
    train_seq = {}
    for m in range(n_trains):
        train_seq[m] = []
    m = 0
    for i in range(len(nodes)):
        if i == len(nodes)-1 : 
            continue
        from_node = nodes[i]
        if from_node.ord == -1: continue
        to_node = nodes[i + 1]
        if (from_node.train_id == to_node.train_id):
            train_seq[m].append(from_node.node_id)
        else :
            train_seq[m].append(from_node.node_id)
            m += 1

    #print(train_seq)
    return train_seq



def gen_ag_graph2(nodes): 
    ag_graph = nx.DiGraph()
    
    pos_x = -50
    pos_y = 0
    max_ord = 0
    for i in range(len(nodes)):
        from_node = nodes[i]
        if max_ord < from_node.ord:
            max_ord = from_node.ord
        if i == len(nodes)-1 : 
            pos_x =max_ord * 50
        from_node.pos_x = pos_x
        from_node.pos_y = pos_y
        #print(from_node)
        if i == len(nodes)-1 : 
            continue
        to_node = nodes[i+1]
        if(from_node.train_id == to_node.train_id and from_node.ord+1 == to_node.ord):
            pos_x = pos_x + 50
            ag_graph.add_edge(from_node.node_id, to_node.node_id)
            from_node.next_fixed_node = to_node.node_id
            to_node.prev_fixed_node = from_node.node_id
            ag_graph[from_node.node_id][to_node.node_id]['weight'] = -(from_node.dep_time-from_node.arr_time)
            ag_graph[from_node.node_id][to_node.node_id]['type'] = 'Fixed'
        else :
            pos_x = 0
            pos_y = pos_y + 50
            if to_node.ord == 1 :
                ag_graph.add_edge(0, to_node.node_id)
                ag_graph[0][to_node.node_id]['weight'] = -(1*to_node.arr_time)
                ag_graph[0][to_node.node_id]['type'] = 'Dummy'

            if to_node.ord == 1 and from_node.ord != -1: 
                ag_graph.add_edge(from_node.node_id, len(nodes)-1)
                ag_graph[from_node.node_id][len(nodes)-1]['weight'] = -(from_node.dep_time-from_node.arr_time)
                ag_graph[from_node.node_id][len(nodes)-1]['type'] = 'Dummy'

            if to_node.ord == -1:
                ag_graph.add_edge(from_node.node_id, len(nodes)-1) 
                ag_graph[from_node.node_id][len(nodes)-1]['weight'] = 0
                ag_graph[from_node.node_id][len(nodes)-1]['type'] = 'Dummy'

    return ag_graph

def gen_ag_graph(nodes): 
    ag_graph = nx.DiGraph()
    pos_x = -50
    pos_y = 0
    max_ord = 0
    for i in range(len(nodes)):
        from_node = nodes[i]
        if max_ord < from_node.ord:
            max_ord = from_node.ord
        if i == len(nodes)-1 : 
            pos_x =max_ord * 50
        from_node.pos_x = pos_x
        from_node.pos_y = pos_y
        #print(from_node)
        if i == len(nodes)-1 : 
            continue
        to_node = nodes[i+1]
        if(from_node.train_id == to_node.train_id and from_node.ord+1 == to_node.ord):
            pos_x = pos_x + 50
            ag_graph.add_edge(from_node.node_id, to_node.node_id)
            from_node.next_fixed_node = to_node.node_id
            to_node.prev_fixed_node = from_node.node_id
            ag_graph[from_node.node_id][to_node.node_id]['weight'] = -(to_node.dep_time-to_node.arr_time)
            ag_graph[from_node.node_id][to_node.node_id]['type'] = 'Fixed'
        else :
            pos_x = 0
            pos_y = pos_y + 50
            if to_node.ord == 1: 
                ag_graph.add_edge(0, to_node.node_id)
                ag_graph[0][to_node.node_id]['weight'] = -(1*to_node.arr_time) - (to_node.dep_time - to_node.arr_time)
                ag_graph[0][to_node.node_id]['type'] = 'Dummy'

            if to_node.ord == 1 and from_node.ord != -1: 
                ag_graph.add_edge(from_node.node_id, len(nodes)-1)
                ag_graph[from_node.node_id][len(nodes)-1]['weight'] = 0
                ag_graph[from_node.node_id][len(nodes)-1]['type'] = 'Dummy'

            if to_node.ord == -1:
                ag_graph.add_edge(from_node.node_id, len(nodes)-1) 
                ag_graph[from_node.node_id][len(nodes)-1]['weight'] = 0
                ag_graph[from_node.node_id][len(nodes)-1]['type'] = 'Dummy'

    return ag_graph




def gen_alter_arcs(p_graph, nodes, train_block_type_map=None):
   
    alterArcSet = []
    for n in p_graph.nodes():
        alternative = []
        for i in range(len(nodes)):
            node = nodes[i]
            if (node.block == n):
                alternative.append(node.node_id)
        if (len(alternative) > 1):
            for i in range(len(alternative)):
                for j in alternative[i+1:]:
                    if nodes[alternative[i]].next_fixed_node == nodes[j].next_fixed_node:
                        continue
                    node_i = nodes[alternative[i]]
                    node_j = nodes[j]
                    node_s_i = nodes[node_i.next_fixed_node] if node_i.next_fixed_node != -1 else node_i
                    node_s_j = nodes[node_j.next_fixed_node] if node_j.next_fixed_node != -1 else node_j

                    if train_block_type_map is not None:
                        same_direction = (node_i.direction == node_j.direction)
                        if same_direction:
                            involved_types = [
                                train_block_type_map.get((node_i.train_id, str(node_i.block)), "").lower(),
                                train_block_type_map.get((node_j.train_id, str(node_j.block)), "").lower(),
                                train_block_type_map.get((node_s_i.train_id, str(node_s_i.block)), "").lower(),
                                train_block_type_map.get((node_s_j.train_id, str(node_s_j.block)), "").lower(),
                            ]
                            if any('weiche' in t for t in involved_types):
                                continue

                    alterArcs = alterArcInfo(node_s_j, node_i, node_s_i, node_j)
                    alterArcSet.append(alterArcs)
    return alterArcSet


def gen_alter_arcs2(p_graph, nodes, train_block_type_map=None):
   
    if train_block_type_map is None:
        train_block_type_map = build_train_block_type_map(
            "/home/s6shdeva_hpc/Python/simple_network_with_pairtype.txt"
        )

    alterArcSet = []
    for n in p_graph.nodes():
        alternative = []
        for i in range(len(nodes)):
            node = nodes[i]
            if (node.block == n):
                alternative.append(node.node_id)
        if (len(alternative) > 1):
            for i in range(len(alternative)):
                for j in alternative[i+1:]:
                    if nodes[alternative[i]].next_fixed_node == nodes[j].next_fixed_node:
                        continue
                    node_s_i = nodes[alternative[i]]
                    node_s_j = nodes[j]
                    # Use prev_fixed_node to get i, j; guard -1 by falling back
                    node_i = nodes[node_s_i.prev_fixed_node] if node_s_i.prev_fixed_node != -1 else node_s_i
                    node_j = nodes[node_s_j.prev_fixed_node] if node_s_j.prev_fixed_node != -1 else node_s_j

                    # Optional segment-type filter
                    if train_block_type_map is not None:
                        same_direction = (node_i.direction == node_j.direction)
                        if same_direction:
                            involved_types = [
                                train_block_type_map.get((node_i.train_id, str(node_i.block)), "").lower(),
                                train_block_type_map.get((node_j.train_id, str(node_j.block)), "").lower(),
                                train_block_type_map.get((node_s_i.train_id, str(node_s_i.block)), "").lower(),
                                train_block_type_map.get((node_s_j.train_id, str(node_s_j.block)), "").lower(),
                            ]
                            if any('weiche' in t for t in involved_types):
                                continue

                    alterArcs = alterArcInfo(node_s_j, node_i, node_s_i, node_j)
                    alterArcSet.append(alterArcs)
    return alterArcSet


if __name__ == '__main__':
    start_time = time.time()
    p_graph = nx.DiGraph()
    
    network_file = "Python/simple_network.txt"
    with open(network_file, mode="r") as f:
        lines = f.readlines()

    for line_num in range(len(lines)):
        #print(line_num, ",", lines[line_num])
        strs = lines[line_num].split('\t')
        from_block = strs[0]
        to_block = strs[1]
        direction = int(strs[2])
        distance = int(strs[3])
        p_graph.add_edge(from_block, to_block)
        p_graph[from_block][to_block]['direction'] =  direction
        p_graph[from_block][to_block]['weight'] = distance



    train_file = "Python/simple_train_case5.txt"
    nodes, nodeMap, train_ids, n_lines = gen_nodes(train_file)
    # Normalize train IDs so preferred ['A','C','B'] works
    for n in nodes:
        n.train_id = str(n.train_id).strip()
        n.name = f"{n.train_id}-{n.block}"  # keep label consistent

    # Build labels for drawing
    node_labels = {n.node_id: n.name for n in nodes}
    #with open(train_file, mode="r") as f:
        #lines = f.readlines()

    

   

    segment_pairs_file = "Python/segment_pairs_network.txt"
    block_type_map = build_block_type_map(segment_pairs_file)
    for node in nodes:
        node.typ = block_type_map.get(str(node.block), "")



    ag_graph = gen_ag_graph(nodes)

    pos = {n.node_id: (n.pos_x, n.pos_y) for n in nodes}

    train_block_type_map = build_train_block_type_map("Python/simple_network_with_pairtype.txt")

    alterArcSet = gen_alter_arcs(p_graph, nodes, train_block_type_map=train_block_type_map)
  
    for arc in alterArcSet:
        # forward arc (s_j -> i)
        pair1 = tuple(sorted([arc.node_s_j.train_id, arc.node_i.train_id]))
        ag_graph.add_edge(arc.node_s_j.node_id, arc.node_i.node_id,
                      weight=-120, type='Alter', train_pair=pair1)

        # symmetric arc (s_i -> j)
        pair2 = tuple(sorted([arc.node_s_i.train_id, arc.node_j.train_id]))
        ag_graph.add_edge(arc.node_s_i.node_id, arc.node_j.node_id,
                      weight=-120, type='Alter', train_pair=pair2)



    eFixed = [(u, v) for (u, v, d) in ag_graph.edges(data=True) if d["type"] == 'Fixed']
    eAlter = [(u, v) for (u, v, d) in ag_graph.edges(data=True) if d["type"] == 'Alter']
    eDummy = [(u, v) for (u, v, d) in ag_graph.edges(data=True) if d["type"] == 'Dummy']    

    if nx.negative_edge_cycle(ag_graph) == False:
        end_node = len(nodes) - 1
        longest_path = nx.bellman_ford_path(ag_graph, source=0, target = end_node, weight='weight')
        print(longest_path)
        print(nx.path_weight(ag_graph, longest_path, weight = 'weight'))


   


    nx.draw_networkx_nodes(ag_graph, pos = pos, node_size=200)
    nx.draw_networkx_edges(ag_graph, pos = pos, edgelist=eFixed, width=1)
    nx.draw_networkx_edges(ag_graph, pos = pos, edgelist=eAlter, width=1, alpha=0.5, edge_color="b", style="dashed")
    nx.draw_networkx_edges(ag_graph, pos=pos, edgelist=eDummy, width=1, alpha=0.5, edge_color="r", style="dashed")
    nx.draw_networkx_labels(ag_graph, pos, node_labels)
    plt.axis("off")
    plt.show()

    A = nx.to_numpy_array(ag_graph)
    for i in range(len(A)):
        for j in range(len(A)):
            if A[i][j] < 0:
                A[i][j] = 1

    print(A)

    print(time.time()-start_time)