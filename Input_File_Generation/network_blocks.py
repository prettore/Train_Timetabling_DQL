import pandas as pd

edges_df = pd.read_excel("element_network_edges_with_names.xlsx")

# Clean names
edges_df["From_Name"] = edges_df["From_Name"].astype(str).str.strip()
edges_df["To_Name"] = edges_df["To_Name"].astype(str).str.strip()

# Ensure type columns
if "From_Type" not in edges_df.columns:
    edges_df["From_Type"] = "Unknown"
if "To_Type" not in edges_df.columns:
    edges_df["To_Type"] = "Unknown"

# Create segment name and segment type string
edges_df["Segment_Name"] = edges_df["From_Name"] + "-" + edges_df["To_Name"]
edges_df["Segment_Type"] = edges_df["From_Type"].astype(str) + "-" + edges_df["To_Type"].astype(str)

# Assign unique segment IDs (A-B and B-A share ID)
segment_id_map = {}
segment_id_counter = 1
segment_ids = []
for idx, row in edges_df.iterrows():
    key = tuple(sorted([row["From_Name"], row["To_Name"]]))
    if key not in segment_id_map:
        segment_id_map[key] = segment_id_counter
        segment_id_counter += 1
    segment_ids.append(segment_id_map[key])
edges_df["Segment_ID"] = segment_ids

# Now build segment pairs including the combined type
segment_pairs = []
for idx, row in edges_df.iterrows():
    A = row["From_Name"]
    B = row["To_Name"]
    dirn = row["Direction"]
    seg1_id = row["Segment_ID"]
    seg1_name = row["Segment_Name"]
    seg1_type = row["Segment_Type"]

    # Find matching next segments
    next_rows = edges_df[
        (edges_df["From_Name"] == B) &
        (edges_df["Direction"] == dirn) &
        (edges_df["To_Name"] != A)
    ]
    for _, next_row in next_rows.iterrows():
        seg2_id = next_row["Segment_ID"]
        seg2_name = next_row["Segment_Name"]
        seg2_type = next_row["Segment_Type"]
        segment_pairs.append({
            "Segment1_ID": seg1_id,
            "Segment1_Name": seg1_name,
            "Segment1_Type": seg1_type,
            "Segment2_ID": seg2_id,
           "Segment2_Name": seg2_name,
            "Segment2_Type": seg2_type,
            "Direction": dirn,
            "Weight": 1
        })

# Output DataFrame
segment_pairs_df = pd.DataFrame(segment_pairs)
segment_pairs_df.to_excel("segment_pairs_network.xlsx", index=False)
print("Segment pairs saved to segment_pairs_network.xlsx")
print(segment_pairs_df.head())