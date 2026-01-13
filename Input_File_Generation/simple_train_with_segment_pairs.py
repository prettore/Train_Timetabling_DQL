import pandas as pd
import re

# --- Step 1: Read and process infrastructure file ---
infra = pd.read_excel('infrastructure_element_export.xlsx')

# Prepare weiche (switches)
df_weiche = infra[infra["Typ"].str.strip().str.lower() == "weiche"].copy()
df_weiche["Name"] = df_weiche["Name"].astype(str).str.strip().str.upper()
df_weiche = df_weiche.drop_duplicates(subset=["Name"]).sort_values(["Name"]).reset_index(drop=True)
df_weiche["ID"] = range(1, len(df_weiche)+1)

# Prepare hauptsignal (main signals)
df_haupt = infra[infra["Typ"].str.strip().str.lower() == "hauptsignal"].copy()
df_haupt["Name"] = df_haupt["Name"].astype(str).str.strip().str.upper()
df_haupt = df_haupt.drop_duplicates(subset=["Name"]).sort_values(["Name"]).reset_index(drop=True)
df_haupt["ID"] = range(len(df_weiche)+1, len(df_weiche)+1+len(df_haupt))

infra_ids = pd.concat([df_weiche, df_haupt], ignore_index=True)[["Name", "ID"]].set_index("Name")

# --- Step 2: Parse and filter fahrzeiten file ---
fahrzeit = pd.read_excel('Fahrzeit_alle_Zuege_Toy_Network.xlsx')

def parse_infra_row(row):
    text = str(row)
    m_haupt = re.search(r'(\S+)\s+Hauptsignal', text, re.IGNORECASE)
    m_weiche = re.search(r'(\S+)\s+Weiche', text, re.IGNORECASE)
    if m_haupt:
        return pd.Series({'Name': m_haupt.group(1).strip().upper(), 'Typ': 'hauptsignal'})
    if m_weiche:
        return pd.Series({'Name': m_weiche.group(1).strip().upper(), 'Typ': 'weiche'})
    return pd.Series({'Name': None, 'Typ': None})

fahrzeit_parsed = fahrzeit["Infrastrukturelement"].apply(parse_infra_row)
fahrzeit_clean = pd.concat([fahrzeit, fahrzeit_parsed], axis=1)
fahrzeit_clean = fahrzeit_clean[fahrzeit_clean["Typ"].isin(["weiche", "hauptsignal"])]

# Remove duplicate weiche within each train
mask_weiche = fahrzeit_clean["Typ"] == "weiche"
fahrzeit_clean = pd.concat([
    fahrzeit_clean[~mask_weiche],
    fahrzeit_clean[mask_weiche].drop_duplicates(subset=["Zugnummer", "Name"])
], ignore_index=True)

# Normalize names for robust mapping
fahrzeit_clean["Name"] = fahrzeit_clean["Name"].astype(str).str.strip().str.upper()
infra_ids.index = infra_ids.index.str.strip().str.upper()
fahrzeit_clean["element_id"] = fahrzeit_clean["Name"].map(infra_ids["ID"])

# --------- Extract km from the 2nd value in Infrastrukturelement ----------
def extract_km_from_infrastruktur_element(row):
    text = str(row)
    parts = text.split()
    if len(parts) >= 2:
        try:
            return float(parts[1].replace(',', '.'))
        except Exception:
            return None
    return None

fahrzeit_clean['km'] = fahrzeit_clean['Infrastrukturelement'].apply(extract_km_from_infrastruktur_element)

# --------- Extract element_type robustly ----------
def robust_element_type(row):
    if 'Typ' in row and pd.notnull(row['Typ']):
        return str(row['Typ']).strip().lower()
    text = str(row['Infrastrukturelement'])
    if "gleisende" in text.lower():
        return "gleisende"
    parts = text.split()
    if len(parts) >= 4:
        return parts[3].strip().lower()
    return None

fahrzeit_clean['element_type'] = fahrzeit_clean.apply(robust_element_type, axis=1)


def infer_direction_for_train(kms):
    # Drop NaN
    kms = kms.dropna()
    if len(kms) < 2:
        return None
    return 1 if kms.iloc[-1] > kms.iloc[0] else 2

# Assign direction for every row based on the full train group
fahrzeit_clean['direction'] = fahrzeit_clean.groupby('Zugnummer')['km'].transform(infer_direction_for_train)


# --- Sort each train by km ---
fahrzeit_clean = fahrzeit_clean.sort_values(["Zugnummer", "km"]).reset_index(drop=True)
fahrzeit_clean["order"] = fahrzeit_clean.groupby("Zugnummer").cumcount() + 1


# --- Final output DataFrame ---
output = fahrzeit_clean[[
    "Zugnummer", "direction", "order", "element_id", "Name", "km", "element_type"
]].rename(columns={
    "Zugnummer": "trainname",
    "element_id": "ID",
    "Name": "element_name"
})

# --- Step 3: Parse all 'gleisende' elements from fahrzeit ---
def parse_gleisende(row):
    text = str(row)
    m_ende = re.search(r'(\S+)\s+Gleisende', text, re.IGNORECASE)
    if m_ende:
        return pd.Series({'Name': m_ende.group(1).strip().upper(), 'Typ': 'gleisende'})
    return pd.Series({'Name': None, 'Typ': None})

gleisende_parsed = fahrzeit["Infrastrukturelement"].apply(parse_gleisende)
fahrzeit_gleisende = pd.concat([fahrzeit, gleisende_parsed], axis=1)
fahrzeit_gleisende = fahrzeit_gleisende[fahrzeit_gleisende["Typ"] == "gleisende"]

# --- Step 4: Insert first and last gleisende per train ---
new_rows = []

for train, group in output.groupby('trainname', sort=False):
    group = group.sort_values("order").copy().reset_index(drop=True)
    gleis_rows = fahrzeit_gleisende[fahrzeit_gleisende["Zugnummer"] == train]
    if not gleis_rows.empty:
        first_gleis_row = gleis_rows.iloc[0]
        last_gleis_row = gleis_rows.iloc[-1]
        first_gleis = first_gleis_row["Name"]
        last_gleis = last_gleis_row["Name"]
        first_gleis_km = extract_km_from_infrastruktur_element(first_gleis_row["Infrastrukturelement"])
        last_gleis_km = extract_km_from_infrastruktur_element(last_gleis_row["Infrastrukturelement"])
        before = {
            "trainname": train,
            "direction": group["direction"].iloc[0],
            "order": 0,
            "ID": infra_ids["ID"].get(first_gleis) if first_gleis in infra_ids.index else None,
            "element_name": first_gleis,
            "km": first_gleis_km,
            "element_type": "gleisende",
        }
        after = {
            "trainname": train,
            "direction": group["direction"].iloc[0],
            "order": group["order"].max() + 1,
            "ID": infra_ids["ID"].get(last_gleis) if last_gleis in infra_ids.index else None,
            "element_name": last_gleis,
            "km": last_gleis_km,
            "element_type": "gleisende",
        }
        new_rows.append(before)
        new_rows.extend(group.to_dict('records'))
        new_rows.append(after)
    else:
        new_rows.extend(group.to_dict('records'))

final = pd.DataFrame(new_rows)
final = final.sort_values(['trainname', 'order']).reset_index(drop=True)
final["order"] = final.groupby('trainname').cumcount() + 1

final.to_excel('signals_switches_with_gleisende_from_fahrzeiten.xlsx', index=False)
print("Exported to signals_switches_with_gleisende_from_fahrzeiten.xlsx")

# --- New Step: Create segment pairs ---
segment_pairs = []

for train, group in final.groupby('trainname', sort=False):
    direction = group['direction'].iloc[0]
    # Always sort by km ascending for robust pairing
    group = group.sort_values('km', ascending=True).reset_index(drop=True)

    hs = group[group["element_type"] == "hauptsignal"]
    w = group[group["element_type"] == "weiche"]
    gleis = group[group["element_type"] == "gleisende"]

    # Pair hauptsignal-hauptsignal
    hs_indices = hs.index.tolist()
    for i in range(len(hs_indices)-1):
        a = group.loc[hs_indices[i]]
        b = group.loc[hs_indices[i+1]]
        if direction == 2:
            from_elem, to_elem = (b, a) if b["km"] > a["km"] else (a, b)
        else:
            from_elem, to_elem = (a, b) if a["km"] < b["km"] else (b, a)
        segment_pairs.append({
            "trainname": train,
            "direction": direction,
            "from_element_id": from_elem["ID"],
            "from_element_name": from_elem["element_name"],
            "from_element_type": from_elem["element_type"],
            "from_km": from_elem["km"],
            "to_element_id": to_elem["ID"],
            "to_element_name": to_elem["element_name"],
            "to_element_type": to_elem["element_type"],
            "to_km": to_elem["km"],
            "pair_type": "hauptsignal-hauptsignal"
        })

    # Pair weiche-weiche
    w_indices = w.index.tolist()
    for i in range(len(w_indices)-1):
        a = group.loc[w_indices[i]]
        b = group.loc[w_indices[i+1]]
        if direction == 2:
            from_elem, to_elem = (b, a) if b["km"] > a["km"] else (a, b)
        else:
            from_elem, to_elem = (a, b) if a["km"] < b["km"] else (b, a)
        segment_pairs.append({
            "trainname": train,
            "direction": direction,
            "from_element_id": from_elem["ID"],
            "from_element_name": from_elem["element_name"],
            "from_element_type": from_elem["element_type"],
            "from_km": from_elem["km"],
            "to_element_id": to_elem["ID"],
            "to_element_name": to_elem["element_name"],
            "to_element_type": to_elem["element_type"],
            "to_km": to_elem["km"],
            "pair_type": "weiche-weiche"
        })

    # Pair gleisende with extremal hauptsignal and weiche
    if not gleis.empty:
        lowest_gleis = gleis.iloc[0]
        highest_gleis = gleis.iloc[-1]
        if not hs.empty:
            lowest_hs = hs.iloc[0]
            highest_hs = hs.iloc[-1]
            # For direction 2, from must be the higher km
            if direction == 2:
                # highest_gleis <-> highest_hs
                segment_pairs.append({
                    "trainname": train,
                    "direction": direction,
                    "from_element_id": highest_gleis["ID"],
                    "from_element_name": highest_gleis["element_name"],
                    "from_element_type": highest_gleis["element_type"],
                    "from_km": highest_gleis["km"],
                    "to_element_id": highest_hs["ID"],
                    "to_element_name": highest_hs["element_name"],
                    "to_element_type": highest_hs["element_type"],
                    "to_km": highest_hs["km"],
                    "pair_type": "gleisende-hauptsignal"
                })
                # lowest_hs <-> lowest_gleis
                segment_pairs.append({
                    "trainname": train,
                    "direction": direction,
                    "from_element_id": lowest_hs["ID"],
                    "from_element_name": lowest_hs["element_name"],
                    "from_element_type": lowest_hs["element_type"],
                    "from_km": lowest_hs["km"],
                    "to_element_id": lowest_gleis["ID"],
                    "to_element_name": lowest_gleis["element_name"],
                    "to_element_type": lowest_gleis["element_type"],
                    "to_km": lowest_gleis["km"],
                    "pair_type": "hauptsignal-gleisende"
                })
            else:
                # lowest_gleis <-> lowest_hs
                segment_pairs.append({
                    "trainname": train,
                    "direction": direction,
                    "from_element_id": lowest_gleis["ID"],
                    "from_element_name": lowest_gleis["element_name"],
                    "from_element_type": lowest_gleis["element_type"],
                    "from_km": lowest_gleis["km"],
                    "to_element_id": lowest_hs["ID"],
                    "to_element_name": lowest_hs["element_name"],
                    "to_element_type": lowest_hs["element_type"],
                    "to_km": lowest_hs["km"],
                    "pair_type": "gleisende-hauptsignal"
                })
                # highest_hs <-> highest_gleis
                segment_pairs.append({
                    "trainname": train,
                    "direction": direction,
                    "from_element_id": highest_hs["ID"],
                    "from_element_name": highest_hs["element_name"],
                    "from_element_type": highest_hs["element_type"],
                    "from_km": highest_hs["km"],
                    "to_element_id": highest_gleis["ID"],
                    "to_element_name": highest_gleis["element_name"],
                    "to_element_type": highest_gleis["element_type"],
                    "to_km": highest_gleis["km"],
                    "pair_type": "hauptsignal-gleisende"
                })
        if not w.empty:
            lowest_w = w.iloc[0]
            highest_w = w.iloc[-1]
            if direction == 2:
                # highest_gleis <-> highest_w
                segment_pairs.append({
                    "trainname": train,
                    "direction": direction,
                    "from_element_id": highest_gleis["ID"],
                    "from_element_name": highest_gleis["element_name"],
                    "from_element_type": highest_gleis["element_type"],
                    "from_km": highest_gleis["km"],
                    "to_element_id": highest_w["ID"],
                    "to_element_name": highest_w["element_name"],
                    "to_element_type": highest_w["element_type"],
                    "to_km": highest_w["km"],
                    "pair_type": "gleisende-weiche"
                })
                # lowest_w <-> lowest_gleis
                segment_pairs.append({
                    "trainname": train,
                    "direction": direction,
                    "from_element_id": lowest_w["ID"],
                    "from_element_name": lowest_w["element_name"],
                    "from_element_type": lowest_w["element_type"],
                    "from_km": lowest_w["km"],
                    "to_element_id": lowest_gleis["ID"],
                    "to_element_name": lowest_gleis["element_name"],
                    "to_element_type": lowest_gleis["element_type"],
                    "to_km": lowest_gleis["km"],
                    "pair_type": "weiche-gleisende"
                })
            else:
                # lowest_gleis <-> lowest_w
                segment_pairs.append({
                    "trainname": train,
                    "direction": direction,
                    "from_element_id": lowest_gleis["ID"],
                    "from_element_name": lowest_gleis["element_name"],
                    "from_element_type": lowest_gleis["element_type"],
                    "from_km": lowest_gleis["km"],
                    "to_element_id": lowest_w["ID"],
                    "to_element_name": lowest_w["element_name"],
                    "to_element_type": lowest_w["element_type"],
                    "to_km": lowest_w["km"],
                    "pair_type": "gleisende-weiche "
                })
                # highest_w <-> highest_gleis
                segment_pairs.append({
                    "trainname": train,
                    "direction": direction,
                    "from_element_id": highest_w["ID"],
                    "from_element_name": highest_w["element_name"],
                    "from_element_type": highest_w["element_type"],
                    "from_km": highest_w["km"],
                    "to_element_id": highest_gleis["ID"],
                    "to_element_name": highest_gleis["element_name"],
                    "to_element_type": highest_gleis["element_type"],
                    "to_km": highest_gleis["km"],
                    "pair_type": "weiche-gleisende"
                })


segment_pairs_df = pd.DataFrame(segment_pairs)
segment_pairs_df.to_excel('segment_pairs.xlsx', index=False)

print("Exported segment pairs to segment_pairs.xlsx")
# Load the segment pairs from the current script's output
segment_pairs_df = pd.read_excel('segment_pairs.xlsx')

# Create the canonical segment name (upper, no spaces)
segment_pairs_df['segment_name_sorted'] = segment_pairs_df['from_element_name'].astype(str).str.strip().str.upper() + '-' + segment_pairs_df['to_element_name'].astype(str).str.strip().str.upper()


# Load the network segment pairs (with IDs)
network_segments_df = pd.read_excel('segment_pairs_network.xlsx')
network_segments_df['Segment1_Name'] = network_segments_df['Segment1_Name'].astype(str).str.strip().str.upper()
network_segments_df['Segment2_Name'] = network_segments_df['Segment2_Name'].astype(str).str.strip().str.upper()

# Create mapping for both segment1 and segment2 to their IDs
id_map = {}
for idx, row in network_segments_df.iterrows():
    id_map[row['Segment1_Name']] = row['Segment1_ID']
    id_map[row['Segment2_Name']] = row['Segment2_ID']

# Assign network_segment_id by matching segment_name_sorted
segment_pairs_df['network_segment_id'] = segment_pairs_df['segment_name_sorted'].map(id_map)

# Save to Excel
segment_pairs_df.to_excel('segment_pairs_with_network_ids.xlsx', index=False)
print("Exported segment pairs with network segment IDs to segment_pairs_with_network_ids.xlsx")


df = pd.read_excel('segment_pairs_with_network_ids.xlsx')

# 1. Add direction column
df['direction'] = df.apply(lambda row: 1 if row['to_km'] > row['from_km'] else 2, axis=1)

# 2. Remove from_element_id and to_element_id
cols_to_drop = ['from_element_id', 'to_element_id']
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

# 3 & 4. For each train, sort by from_km ascending (direction=1) or descending (direction=2)
sorted_dfs = []
for train, group in df.groupby('trainname', sort=False):
    # It is possible to have both directions in one train, so sort each direction group separately
    for dir_val, dir_group in group.groupby('direction'):
        if dir_val == 1:
            sorted_group = dir_group.sort_values('from_km', ascending=True)
        else:
            sorted_group = dir_group.sort_values('from_km', ascending=False)
        sorted_dfs.append(sorted_group)

df_sorted = pd.concat(sorted_dfs, ignore_index=True)

# Optional: Reorder columns for readability (customize as you wish)
cols_order = ['trainname', 'direction', 'from_element_name', 'to_element_name', 'from_km', 'to_km',
              'from_element_type', 'to_element_type', 'pair_type', 'network_segment_id', 'segment_name_sorted']
cols_final = [col for col in cols_order if col in df_sorted.columns] + [
    col for col in df_sorted.columns if col not in cols_order
]
df_sorted = df_sorted[cols_final]

# Export the sorted and cleaned file
df_sorted.to_excel('segment_pairs_with_network_ids_sorted.xlsx', index=False)
print("Exported sorted segment pairs with network segment IDs to segment_pairs_with_network_ids_sorted.xlsx")




# ---- Load data ----
segment_pairs = pd.read_excel('segment_pairs_with_network_ids_sorted.xlsx')

# Add a column to preserve original order
segment_pairs['original_order'] = segment_pairs.index

# Load belegung files
belegung_files = {
    'A': pd.read_excel('TrainA_belegung.xlsx'),
    'B': pd.read_excel('TrainB_belegung.xlsx'),
    'C': pd.read_excel('TrainC_belegung.xlsx'),
}

def get_train_file(trainname):
    trainname = str(trainname).strip().upper()
    if trainname.endswith('A'):
        return belegung_files['A']
    elif trainname.endswith('B'):
        return belegung_files['B']
    elif trainname.endswith('C'):
        return belegung_files['C']
    else:
        return None

def find_anf_ende_for_gleisende(df_belegung, gleisende_element):
    gleisende_element = str(gleisende_element).strip().upper()
    def extract_gleisende_id(belegung_element):
        match = re.search(r'\|GE\|([^|]+)\|', str(belegung_element).upper())
        return match.group(1) if match else None
    df_belegung = df_belegung.copy()
    df_belegung['gleisende_id'] = df_belegung['Belegungselement'].apply(extract_gleisende_id)
    matched = df_belegung[df_belegung['gleisende_id'] == gleisende_element]
    if matched.empty:
        return None, None
    row = matched.iloc[0]
    return row.get('Anf', None), row.get('Ende', None)

def find_anf_ende_for_second_element(df_belegung, element):
    element = str(element).strip().upper()
    belegung_match = df_belegung[df_belegung['Belegungselement'].str.upper().str.contains(rf'\b{re.escape(element)}\b', na=False)]
    if belegung_match.empty:
        return None, None
    row = belegung_match.iloc[0]
    return row.get('Anf', None), row.get('Ende', None)

def find_anf_ende_for_weiche_pair(df_belegung, el1, el2):
    el1 = str(el1).strip().upper()
    el2 = str(el2).strip().upper()
    g_rows = df_belegung[df_belegung['Art'].astype(str).str.upper() == 'G']
    mask = (
        g_rows['Belegungselement'].str.upper().str.contains(rf'\b{re.escape(el1)}\b', na=False) &
        g_rows['Belegungselement'].str.upper().str.contains(rf'\b{re.escape(el2)}\b', na=False)
    )
    pair_rows = g_rows[mask]
    if pair_rows.empty:
        return None, None
    row = pair_rows.iloc[0]
    return row.get('Anf', None), row.get('Ende', None)

# ---- Fill Anf/Ende for each pair ----
# ---- Fill Anf/Ende for each pair ----

result_rows = []
for idx, row in segment_pairs.iterrows():
    trainname = row['trainname']
    pair_type = str(row['pair_type'])
    seg_sorted = row['segment_name_sorted']
    elements = seg_sorted.split('-')
    belegung = get_train_file(trainname)
    ptype = pair_type.strip().lower()

    if 'hauptsignal-hauptsignal' in ptype:
        anf, ende = find_anf_ende_for_second_element(belegung, elements[1])
    elif 'hauptsignal-gleisende' in ptype:
        gleisende_name = '-'.join(elements[1:])
        anf, ende = find_anf_ende_for_gleisende(belegung, gleisende_name)
    elif 'gleisende-hauptsignal' in ptype:
        anf, ende = find_anf_ende_for_second_element(belegung, elements[1])
    elif 'weiche-gleisende' in ptype or 'gleisende-weiche' in ptype or 'weiche-weiche' in ptype:
        anf, ende = find_anf_ende_for_weiche_pair(belegung, elements[0], elements[1])
    else:
        anf, ende = None, None

    result_rows.append({
        'orig_index': idx,  # Preserve original order
        'trainname': trainname,
        'direction': row.get('direction', None),
        'network_segment_id': row.get('network_segment_id', None),
        'pair_type': row.get('pair_type', None), 
        'from_km': row.get('from_km', None),
        'Anf': anf,
        'Ende': ende
    })

# Convert to DataFrame
result_df = pd.DataFrame(result_rows)

# --- Preserve the original order from segment_pairs_with_network_ids_sorted.xlsx ---
result_df = result_df.sort_values('orig_index').reset_index(drop=True)

# Assign order based on appearance for each train and direction
result_df['order'] = result_df.groupby(['trainname', 'direction']).cumcount() + 1

# Only keep requested columns
result_df = result_df[['trainname', 'direction', 'order', 'network_segment_id', 'pair_type', 'Anf', 'Ende']]

# Output
result_df.to_excel('train_segments_with_times.xlsx', index=False)
print("Saved final output to train_segments_with_times.xlsx")




