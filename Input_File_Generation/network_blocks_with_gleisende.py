import pandas as pd
import re

def are_parallel(f1: str, f2: str) -> bool:
    parts1 = [p.strip() for p in str(f1).split('-')]
    parts2 = [p.strip() for p in str(f2).split('-')]
    if len(parts1) >= 3 and len(parts2) >= 3:
        return parts1[0] == parts2[0] and parts1[-1] == parts2[-1]
    return False
def get_track_from_betriebstelle(fw):
    m = re.search(r'-\s*(\d+)\s*-', str(fw))
    if m:
        return int(m.group(1))
    return None

def normalize_fw(fw):
    return str(fw).replace('_GEN', '').replace(' ', '').strip().lower()



# ==== 1. Extract elements ====
df_raw = pd.read_excel("infrastructure_element_export.xlsx")

# Switches
df_switch = df_raw[df_raw["Typ"].str.lower().str.contains("weiche", na=False)].copy()
df_switch = df_switch.rename(columns={"Kilometrierung": "km"})
df_switch["km"] = df_switch["km"].astype(float)
df_switch = df_switch.drop_duplicates(subset=["Name"], keep="first")
df_switch = df_switch.sort_values("km", ascending=True).reset_index(drop=True)
df_switch["Element_Name"] = df_switch["Name"]
df_switch["Element_km"] = df_switch["km"]
df_switch["BetriebstelleFahrwege"] = df_switch["BetriebstelleFahrwege"]
df_switch["track"] = df_switch["BetriebstelleFahrwege"].apply(get_track_from_betriebstelle)

# Hauptsignale
df_h = df_raw[df_raw["Typ"].astype(str).str.strip().str.casefold() == 'hauptsignal']
df_h = df_h.rename(columns={'Name': 'Hauptsignal_Name', 'Kilometrierung': 'km'})
df_h["km"] = df_h["km"].astype(str).str.replace(",", ".").astype(float)
df_h = df_h.drop_duplicates(subset=["Hauptsignal_Name"], keep="first").reset_index(drop=True)
df_h["Element_Name"] = df_h["Hauptsignal_Name"]
df_h["Element_km"] = df_h["km"]
df_h["BetriebstelleFahrwege"] = df_h["BetriebstelleFahrwege"]
df_h["track"] = df_h["BetriebstelleFahrwege"].apply(get_track_from_betriebstelle)
def direction(s: str) -> str:
    s_low = str(s).lower()
    return 'steigend' if s_low.find('start') < s_low.find('ziel') else 'fallend'
df_h["Richtung"] = df_h["BetriebstelleFahrwege"].astype(str).apply(direction)

# Gleisenden
df_gleisende = df_raw[df_raw["Typ"].str.lower().str.contains("gleisende", na=False)].copy()
df_gleisende = df_gleisende.rename(columns={"Kilometrierung": "km"})
df_gleisende["km"] = df_gleisende["km"].astype(float)
df_gleisende = df_gleisende.drop_duplicates(subset=["Name"], keep="first")
df_gleisende = df_gleisende.sort_values("km", ascending=True).reset_index(drop=True)
df_gleisende["Element_Name"] = df_gleisende["Name"]
df_gleisende["Element_km"] = df_gleisende["km"]
df_gleisende["track"] = df_gleisende["BetriebstelleFahrwege"].apply(get_track_from_betriebstelle)

# ==== 2. Assign IDs ====
switch_elements = (
    df_switch[["Element_Name", "Element_km", "BetriebstelleFahrwege", "track"]]
    .drop_duplicates(subset=["Element_Name"])
    .sort_values(["Element_km", "Element_Name"])
    .reset_index(drop=True)
)
switch_elements["Element_ID"] = range(1, len(switch_elements)+1)

hauptsignal_elements = (
    df_h[["Element_Name", "Element_km", "BetriebstelleFahrwege", "track"]]
    .drop_duplicates(subset=["Element_Name"])
    .sort_values(["Element_km", "Element_Name"])
    .reset_index(drop=True)
)
start_id = len(switch_elements) + 1
hauptsignal_elements["Element_ID"] = range(start_id, start_id + len(hauptsignal_elements))

gleisende_elements = (
    df_gleisende[["Element_Name", "Element_km","BetriebstelleFahrwege", "track"]]
    .drop_duplicates(subset=["Element_Name"])
    .sort_values(["Element_km", "Element_Name"])
    .reset_index(drop=True)
)
start_id = len(switch_elements) + len(hauptsignal_elements) + 1
gleisende_elements["Element_ID"] = range(start_id, start_id + len(gleisende_elements))

all_elements = pd.concat([switch_elements, hauptsignal_elements, gleisende_elements], ignore_index=True).reset_index(drop=True)
id_map = {(row["Element_Name"], row["Element_km"]): row["Element_ID"] for idx, row in all_elements.iterrows()}
id_to_name = {row["Element_ID"]: row["Element_Name"] for idx, row in all_elements.iterrows()}
id_to_km = {row["Element_ID"]: row["Element_km"] for idx, row in all_elements.iterrows()}


# For hauptsignale and gleisende: allow same Name on different Fahrwege/km
df_h_hg = df_h.drop_duplicates(subset=["Hauptsignal_Name", "BetriebstelleFahrwege", "km"]).copy()
df_gleisende_hg = df_gleisende.drop_duplicates(subset=["Name", "BetriebstelleFahrwege", "km"]).copy()

# Assign local IDs for these combinations, to avoid disturbing global IDs used elsewhere
all_hg_elements = pd.concat([
    df_h_hg[["Hauptsignal_Name", "km", "BetriebstelleFahrwege"]],
    df_gleisende_hg[["Name", "km", "BetriebstelleFahrwege"]].rename(columns={"Name": "Hauptsignal_Name"})
], ignore_index=True)

all_hg_elements = all_hg_elements.drop_duplicates(subset=["Hauptsignal_Name", "BetriebstelleFahrwege", "km"]).reset_index(drop=True)
all_hg_elements["Local_HG_ID"] = range(1, len(all_hg_elements)+1)

# Map for hauptsignal-gleisende pairing ONLY
id_map_hg = {
    (row["Hauptsignal_Name"], row["km"], row["BetriebstelleFahrwege"]): row["Local_HG_ID"]
    for idx, row in all_hg_elements.iterrows()
}



# ==== 3. Generate switch pairs (bidirectional, as before) ====
switch_pairs = []
n = len(df_switch)
i = 0
while i < n - 1:
    curr = df_switch.iloc[i]
    curr_fahrweg = curr["BetriebstelleFahrwege"]
    block = [i]
    for j in range(i + 1, n):
        next_fahrweg = df_switch.iloc[j]["BetriebstelleFahrwege"]
        if are_parallel(curr_fahrweg, next_fahrweg):
            block.append(j)
        else:
            break
    if len(block) > 1:
        for k in range(len(block) - 1):
            a, b = block[k], block[k+1]
            switch_pairs.append((
                id_map[(df_switch.iloc[a]["Name"], df_switch.iloc[a]["km"])],
                id_map[(df_switch.iloc[b]["Name"], df_switch.iloc[b]["km"])],
                1, 1
            ))
            switch_pairs.append((
                id_map[(df_switch.iloc[b]["Name"], df_switch.iloc[b]["km"])],
                id_map[(df_switch.iloc[a]["Name"], df_switch.iloc[a]["km"])],
                2, 1
            ))
        if len(block) > 2:
            first, last = block[0], block[-1]
            switch_pairs.append((
                id_map[(df_switch.iloc[first]["Name"], df_switch.iloc[first]["km"])],
                id_map[(df_switch.iloc[last]["Name"], df_switch.iloc[last]["km"])],
                1, 1
            ))
            switch_pairs.append((
                id_map[(df_switch.iloc[last]["Name"], df_switch.iloc[last]["km"])],
                id_map[(df_switch.iloc[first]["Name"], df_switch.iloc[first]["km"])],
                2, 1
            ))
        i = block[-1]
    else:
        next_ = df_switch.iloc[i+1]
        if not are_parallel(curr_fahrweg, next_["BetriebstelleFahrwege"]):
            switch_pairs.append((
                id_map[(curr["Name"], curr["km"])],
                id_map[(next_["Name"], next_["km"])],
                1, 1
            ))
            switch_pairs.append((
                id_map[(next_["Name"], next_["km"])],
                id_map[(curr["Name"], curr["km"])],
                2, 1
            ))
        i += 1

# ==== 4. Generate hauptsignal pairs (bidirectional) ====
def get_block_pairs(df: pd.DataFrame):
    pairs = []
    for dir_str in ['steigend', 'fallend']:
        df_dir = (
            df[df['Richtung'] == dir_str]
            .sort_values('km', ascending=(dir_str == 'steigend'))
            .reset_index(drop=True)
        )
        n = len(df_dir)
        for i in range(n):
            name_i = df_dir.at[i, 'Hauptsignal_Name']
            km_i   = df_dir.at[i, 'km']
            fw_i   = df_dir.at[i, 'BetriebstelleFahrwege']
            is_follower = False
            for k in range(i):
                if (
                    df_dir.at[k, 'km'] != km_i
                    and are_parallel(df_dir.at[k, 'BetriebstelleFahrwege'], fw_i)
                ):
                    is_follower = True
                    break
            if is_follower:
                for j in range(i+1, n):
                    km_j = df_dir.at[j, 'km']
                    fw_j = df_dir.at[j, 'BetriebstelleFahrwege']
                    name_j = df_dir.at[j, 'Hauptsignal_Name']
                    if km_j == km_i:
                        continue
                    if not are_parallel(fw_i, fw_j):
                        from_id = id_map.get((name_i, km_i), None)
                        to_id = id_map.get((name_j, km_j), None)
                        if from_id is not None and to_id is not None:
                            pairs.append((from_id, to_id, 1, 1))
                            pairs.append((to_id, from_id, 2, 1))
                        break
            else:
                parallels = []
                j = i + 1
                while j < n:
                    km_j = df_dir.at[j, 'km']
                    fw_j = df_dir.at[j, 'BetriebstelleFahrwege']
                    if km_j == km_i:
                        j += 1
                        continue
                    if are_parallel(fw_i, fw_j):
                        parallels.append(j)
                        j += 1
                        continue
                    break
                for p in parallels:
                    name_p = df_dir.at[p, 'Hauptsignal_Name']
                    km_p = df_dir.at[p, 'km']
                    from_id = id_map.get((name_i, km_i), None)
                    to_id = id_map.get((name_p, km_p), None)
                    if from_id is not None and to_id is not None:
                        pairs.append((from_id, to_id, 1, 1))
                        pairs.append((to_id, from_id, 2, 1))
                if not parallels:
                    while j < n and df_dir.at[j, 'km'] == km_i:
                        j += 1
                    if j < n:
                        fw_j = df_dir.at[j, 'BetriebstelleFahrwege']
                        name_j = df_dir.at[j, 'Hauptsignal_Name']
                        km_j = df_dir.at[j, 'km']
                        from_id = id_map.get((name_i, km_i), None)
                        to_id = id_map.get((name_j, km_j), None)
                        if from_id is not None and to_id is not None:
                            pairs.append((from_id, to_id, 1, 1))
                            pairs.append((to_id, from_id, 2, 1))
    return pairs

hauptsignal_pairs = get_block_pairs(df_h)

# ==== 5. Generate gleisende pairs (with switches & hauptsignal, respecting track logic) ====
pairs_gleisende = []

# --- Switch logic: pair first switch with all gleisende before it, last switch with all after ---
first_switch = switch_elements.iloc[0]
last_switch = switch_elements.iloc[-1]
first_switch_km = first_switch["Element_km"]
last_switch_km = last_switch["Element_km"]

gleisende_before_first_switch = gleisende_elements[gleisende_elements["Element_km"] < first_switch_km]
gleisende_after_last_switch = gleisende_elements[gleisende_elements["Element_km"] > last_switch_km]

for _, g in gleisende_before_first_switch.iterrows():
    pairs_gleisende.append((g["Element_ID"], first_switch["Element_ID"], 1, 1))
    pairs_gleisende.append((first_switch["Element_ID"], g["Element_ID"], 2, 1))
for _, g in gleisende_after_last_switch.iterrows():
    pairs_gleisende.append((g["Element_ID"], last_switch["Element_ID"], 2, 1))
    pairs_gleisende.append((last_switch["Element_ID"], g["Element_ID"], 1, 1))


pairs_hauptsignal_gleisende = []

# normalize once
df_h["norm_fw"]          = df_h["BetriebstelleFahrwege"].apply(normalize_fw)
df_gleisende["norm_fw"]  = df_gleisende["BetriebstelleFahrwege"].apply(normalize_fw)

# only look at FW present in both
common_fws = set(df_h["norm_fw"]) & set(df_gleisende["norm_fw"])
for fw in common_fws:
    h_fw = df_h       [df_h["norm_fw"]         == fw]
    g_fw = df_gleisende[df_gleisende["norm_fw"] == fw]
    if h_fw.empty or g_fw.empty:
        continue

    # now also group by physical track number
    for trk in set(h_fw["track"]) & set(g_fw["track"]):
        h_tr = h_fw[h_fw["track"] == trk]
        g_tr = g_fw[g_fw["track"] == trk]
        if h_tr.empty or g_tr.empty:
            continue

        # link every H ↔ every G
        for _, h_row in h_tr.iterrows():
            id_h = id_map.get((h_row["Hauptsignal_Name"], h_row["km"]))
            if not id_h:
                print(f"⚠️ missing global ID for Hauptsignal {h_row['Hauptsignal_Name']} @ km {h_row['km']}")
                continue

            for _, g_row in g_tr.iterrows():
                id_g = id_map.get((g_row["Name"], g_row["km"]))
                if not id_g:
                    print(f"⚠️ missing global ID for Gleisende {g_row['Name']} @ km {g_row['km']}")
                    continue

                # forward link
                pairs_hauptsignal_gleisende.append((id_h, id_g, 1, 1))
                # reverse link
                pairs_hauptsignal_gleisende.append((id_g, id_h, 2, 1))





# ==== 6. Combine, deduplicate, compute direction by km ====
all_pairs = switch_pairs + hauptsignal_pairs + pairs_gleisende + pairs_hauptsignal_gleisende

all_pairs_df = pd.DataFrame(all_pairs, columns=["From_ID", "To_ID", "Direction", "Weight"])
all_pairs_df = all_pairs_df.drop_duplicates(subset=["From_ID", "To_ID", "Direction"]).sort_values(["From_ID", "To_ID", "Direction"]).reset_index(drop=True)

# Add names and km columns
all_pairs_df["From_Name"] = all_pairs_df["From_ID"].map(id_to_name)
all_pairs_df["From_km"] = all_pairs_df["From_ID"].map(id_to_km)
all_pairs_df["To_Name"] = all_pairs_df["To_ID"].map(id_to_name)
all_pairs_df["To_km"] = all_pairs_df["To_ID"].map(id_to_km)

# Set direction by km as final rule
all_pairs_df["Direction"] = all_pairs_df.apply(lambda row: 1 if row["To_km"] > row["From_km"] else 2, axis=1)

cols = ["From_ID", "From_Name", "From_km", "To_ID", "To_Name", "To_km", "Direction", "Weight"]
all_pairs_df = all_pairs_df[cols]
# Add Type columns
element_type_map = df_raw.set_index("Name")["Typ"].to_dict()
all_pairs_df["From_Type"] = all_pairs_df["From_Name"].map(element_type_map)
all_pairs_df["To_Type"] = all_pairs_df["To_Name"].map(element_type_map)

# Adjust column order
cols = [
    "From_ID", "From_Name", "From_Type", "From_km",
    "To_ID", "To_Name", "To_Type", "To_km",
    "Direction", "Weight"
]
all_pairs_df = all_pairs_df[cols]


all_pairs_df.to_excel("element_network_edges_with_names.xlsx", index=False)
print("Output written to element_network_edges_with_names.xlsx")

# ==== 7. Build segment/block adjacency file ====

# Create a unique block key for each directed edge
# Use an unordered tuple for block key (no direction in Block_ID)
all_pairs_df['Block_Key'] = all_pairs_df.apply(
    lambda row: tuple(sorted([row['From_Name'], row['To_Name']])), axis=1
)
unique_blocks = all_pairs_df[['Block_Key']].drop_duplicates().reset_index(drop=True)
unique_blocks['Block_ID'] = unique_blocks.index + 1

# Attach block IDs to the edge DataFrame
all_pairs_df = all_pairs_df.merge(unique_blocks, on='Block_Key', how='left')

# Prepare mapping: (From_Name, To_Name, Direction) -> Block_ID
block_id_map = dict(zip(all_pairs_df['Block_Key'], all_pairs_df['Block_ID']))

# For each edge/block (A→B, d), find all (B→C, d)
block_adjacency = []
for idx, row in all_pairs_df.iterrows():
    from_name = row['From_Name']
    to_name = row['To_Name']
    direction = row['Direction']
    block_id_A = row['Block_ID']

    # Find all "next" edges that start from to_name, with SAME direction
    next_edges = all_pairs_df[(all_pairs_df['From_Name'] == to_name) & (all_pairs_df['Direction'] == direction)]
    for _, next_row in next_edges.iterrows():
        block_id_B = next_row['Block_ID']
        if block_id_A == block_id_B:
            continue
        block_adjacency.append({
            'from_block_id': block_id_A,
            'to_block_id': block_id_B,
            'from_pair': (from_name, to_name, direction),
            'to_pair': (next_row['From_Name'], next_row['To_Name'], direction),
            'direction': direction,
            'weight': 1
        })

block_adj_df = pd.DataFrame(block_adjacency)

# ... (previous code from block_adjacency building)

def pair_str(from_name, to_name):
    return f"{from_name}-{to_name}"

# Prepare DataFrame for output
output_df = pd.DataFrame([
    {
        "from_block_id": entry["from_block_id"],
        "to_block_id": entry["to_block_id"],
        "from_pair": pair_str(*entry["from_pair"][:2]),
        "to_pair": pair_str(*entry["to_pair"][:2]),
        "direction": entry["direction"],
        "weight": entry["weight"]
    }
    for entry in block_adjacency
])

output_df.to_excel('block_segment_adjacency_directional_full_step.xlsx', index=False)
print("Block adjacency (directional, minimal columns) written to block_segment_adjacency_directional_full_step.xlsx")
print(output_df.head(20))

