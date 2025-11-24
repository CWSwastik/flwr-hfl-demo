import streamlit as st
import pandas as pd
import glob
import os
import altair as alt
import graphviz


st.set_page_config(page_title="HFL Dashboard", layout="wide")


st.title("📊 HFL Training Dashboard")


def list_logs_subdirs(base="./logs"):
    """Return a list of subdirectories directly under `base`.

    Always include the base folder itself as the default option (so existing
    behavior remains when no subfolders exist).
    """
    if not os.path.exists(base):
        return [base]
    entries = []
    for name in sorted(os.listdir(base)):
        path = os.path.join(base, name)
        if os.path.isdir(path):
            entries.append(path)
    # If there are no subdirectories, fall back to the base folder
    return entries or [base]


def build_logs_options(base="./logs"):
    """Return two structures for the selectbox UI:

    - names: list of short folder names to display
    - name_to_path: mapping from displayed name -> full path
    The function always returns at least one option (the base folder).
    """
    paths = list_logs_subdirs(base)
    names = []
    name_to_path = {}
    for p in paths:
        # show just the folder name (fall back to the path if basename is empty)
        name = os.path.basename(p) or p
        # ensure unique display names; if duplicate, fall back to full path
        if name in name_to_path:
            name = p
        names.append(name)
        name_to_path[name] = p
    return names, name_to_path


def make_line_chart(df, x, y, color=None, title=""):
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X(f"{x}:Q", title="Round"),
            y=alt.Y(f"{y}:Q", title=title),
            color=alt.Color(f"{color}:N") if color else alt.value("#66c2a5"),
            tooltip=[x, y] + ([color] if color else []),
        )
        .properties(height=350, width=600)
        .interactive()
    )
    return chart

def make_dist_chart(df, x_col, title, color_scheme="tableau10", sort_col=None, x_axis_label=None):
    """Creates a normalized stacked bar chart with a custom X-axis label."""
    # Melt Class columns
    class_cols = [c for c in df.columns if c.startswith("Class_")]
    
    # Determine id_vars: must include x_col, and sort_col if it exists
    id_vars = [x_col]
    if sort_col and sort_col in df.columns and sort_col != x_col:
        id_vars.append(sort_col)
    
    # Melt
    df_melt = df.melt(id_vars=id_vars, value_vars=class_cols, var_name="Class", value_name="Count")
    df_melt["ClassLabel"] = df_melt["Class"].apply(lambda x: x.replace("Class_", ""))

    # Define Sorting
    x_sort = None
    if sort_col and sort_col in df.columns:
        x_sort = alt.EncodingSortField(field=sort_col, order="ascending")

    # Use custom label if provided, else default to column name
    x_title = x_axis_label if x_axis_label else x_col

    chart = alt.Chart(df_melt).mark_bar().encode(
        x=alt.X(f"{x_col}:N", title=x_title, sort=x_sort),
        y=alt.Y("Count:Q", stack="normalize", axis=alt.Axis(format="%"), title="Label Distribution"),
        color=alt.Color("ClassLabel:N", title="Class", scale=alt.Scale(scheme=color_scheme)),
        tooltip=[x_col, "ClassLabel", "Count"]
    ).properties(title=title, height=400, width=400)
    return chart

def create_tree_graph(df, parent_col, child_col, title):
    """Generates a Graphviz Dot string for a parent-child tree."""
    dot = graphviz.Digraph()
    dot.attr(rankdir='LR', size='8,5')
    dot.attr('node', shape='box', style='filled', fillcolor='lightblue')
    
    # Title node (invisible or just a label)
    # dot.attr(label=title, labelloc='t', fontsize='20')

    # Get unique parents
    parents = sorted(df[parent_col].unique())
    
    for p in parents:
        p_str = str(p)
        # Parent Node
        dot.node(f"P_{p_str}", label=f"{title} {p_str}", fillcolor='orange', shape='folder')
        
        # Children
        children = df[df[parent_col] == p][child_col].unique()
        for c in children:
            c_str = str(c)
            dot.node(f"C_{c_str}", label=c_str, fillcolor='lightgrey', shape='note')
            dot.edge(f"P_{p_str}", f"C_{c_str}")
            
    return dot

# Logs selector
st.header("🌲 Logs Directory")

# Initialize or refresh the available names/map in session state. We avoid
# calling `st.experimental_rerun()` because some Streamlit builds may not have
# that attribute; instead we update session state and rely on Streamlit's
# normal rerun-on-interaction behavior.
initial_names, initial_map = build_logs_options("./logs")
if "logs_names" not in st.session_state or "logs_map" not in st.session_state:
    st.session_state["logs_names"] = initial_names
    st.session_state["logs_map"] = initial_map

if st.button("Refresh logs list"):
    new_names, new_map = build_logs_options("./logs")
    st.session_state["logs_names"] = new_names
    st.session_state["logs_map"] = new_map
    # reset choice to first available option to avoid stale choice
    st.session_state["logs_choice"] = new_names[0]

# Ensure there's always at least one name in the list
names = st.session_state.get("logs_names", initial_names)
name_to_path = st.session_state.get("logs_map", initial_map)

# Use a selectbox bound to session_state so changes persist and trigger reruns
if "logs_choice" not in st.session_state or st.session_state["logs_choice"] not in names:
    st.session_state["logs_choice"] = names[0]

selected_name = st.selectbox("Choose logs folder", names, index=names.index(st.session_state["logs_choice"]), key="logs_choice")
selected_logs = name_to_path.get(selected_name, "./logs")


# --- TABS ---
tab_training, tab_clustering = st.tabs(["Training Metrics", "Clustering Analysis"])

# --- TAB 1: TRAINING METRICS ---
with tab_training:
    # Central
    st.header("Central Server")
    central_log = os.path.join(selected_logs, "central", "central_server.log")
    if os.path.exists(central_log):
        df_c = pd.read_csv(central_log)
								 
        col1, col2 = st.columns(2)
        with col1:
            st.altair_chart(
                make_line_chart(df_c, "round", "loss", title="Loss"),
                use_container_width=True,
            )
        with col2:
            st.altair_chart(
                make_line_chart(df_c, "round", "accuracy", title="Accuracy"),
                use_container_width=True,
            )
    else:
        st.warning("Central server log not found.")

    # Edges
    st.header("Edge Servers")
    edge_logs = glob.glob(os.path.join(selected_logs, "edge", "*.log"))
    edge_logs = sorted(edge_logs, key=lambda p: os.path.splitext(os.path.basename(p))[0])
    if edge_logs:
        for path in edge_logs:
            name = os.path.splitext(os.path.basename(path))[0]
	
            try:
                df_e = pd.read_csv(path)
            except Exception as e:
                st.warning(f"Failed to read edge log {path}: {e}")
                continue
            st.subheader(f"Edge: {name}")
            c1, c2 = st.columns(2)
            with c1:
                st.altair_chart(
                    make_line_chart(df_e, "round", "loss", title="Loss"),
                    use_container_width=True,
                )
            with c2:
                st.altair_chart(
                    make_line_chart(df_e, "round", "accuracy", title="Accuracy"),
                    use_container_width=True,
                )
    else:
        st.warning("No edge server logs found.")

    # Clients
    st.header("Clients")
    client_logs = glob.glob(os.path.join(selected_logs, "clients", "*.log"))
    clients = {}
    for path in client_logs:
        fname = os.path.basename(path)
        cid = fname.split("_")[0]
        split = "train" if "train" in fname else "test"
        clients.setdefault(cid, {})[split] = path

    if clients:
        for cid in sorted(clients.keys(), key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else x):
            paths = clients[cid]
            st.subheader(f"Client: {cid}")
            parts = []
            for split in ("train", "test"):
                p = paths.get(split)
                if not p:
                    continue
                try:
                    df = pd.read_csv(p)
                except Exception as e:
                    st.warning(f"Failed to read client log {p}: {e}")
                    continue
                df["split"] = split.capitalize()
                parts.append(df)
            if not parts:
                st.warning(f"No readable logs for client {cid}")
                continue
            df_all = pd.concat(parts, ignore_index=True)
            c1, c2 = st.columns(2)
            with c1:
                st.altair_chart(
                    make_line_chart(df_all, "round", "loss", color="split", title="Loss"),
                    use_container_width=True,
                )
            with c2:
                st.altair_chart(
                    make_line_chart(
                        df_all, "round", "accuracy", color="split", title="Accuracy"
                    ),
                    use_container_width=True,
                )
    else:
        st.warning("No client logs found.")


# --- TAB 2: CLUSTERING ANALYSIS ---
with tab_clustering:
    cluster_strategy = selected_name.split("-cluster_")[-1].split("-")[0].upper()
    st.header(f"Clustering Analysis (None vs {cluster_strategy})")
    st.write(f"Analyzing clustering results for: **{selected_name}**")
    
    pre_csv = os.path.join(selected_logs, "distribution_pre_clustering.csv")
    post_csv = os.path.join(selected_logs, "distribution_post_clustering.csv")
    
    # Check if files exist
    files_exist = True
    if not os.path.exists(pre_csv):
        st.warning(f"Pre-clustering data not found: {pre_csv}")
        files_exist = False
    if not os.path.exists(post_csv):
        st.warning(f"Post-clustering data not found: {post_csv}")
        files_exist = False
        
    if files_exist:
        df_pre = pd.read_csv(pre_csv)
        df_post = pd.read_csv(post_csv)
        
        # --- Visualization ---
        st.subheader("1. Edge/Cluster Label Distribution (Pre vs Post)")
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("### Pre-Clustering (Original distribution at Edges)")
            # Sort by DefaultClusterID (original topology) if available, else ClientName
            sort_key_pre = "DefaultClusterID" if "DefaultClusterID" in df_pre.columns else "ClientName"
            st.altair_chart(
                make_dist_chart(df_pre, sort_key_pre, "Distribution Sorted by Initial Topology", x_axis_label="Edge Server ID"),
                use_container_width=True
            )
            
        with c2:
            st.markdown("### Post-Clustering")
            # Sort by new ClusterID
            sort_key_post = "ClusterID" if "ClusterID" in df_post.columns else "ClientName"
            st.altair_chart(
                make_dist_chart(df_post, sort_key_post, "Distribution Sorted by Assigned Cluster", x_axis_label="Edge Server ID"),
				use_container_width=True
            )
            
        
        # --- SECTION 2: TOPOLOGY TREE VIEW ---
        st.markdown("---")
        st.subheader("2. Topology Tree View")
        
        t1, t2 = st.columns(2)
        
        with t1:
            st.markdown("### Before Clustering (Static)")
            if "DefaultClusterID" in df_pre.columns:
                # Parent: DefaultClusterID, Child: ClientName
                graph_pre = create_tree_graph(df_pre, "DefaultClusterID", "ClientName", "Edge")
                st.graphviz_chart(graph_pre)
            else:
                st.info("Insufficient data for Pre-clustering tree.")

        with t2:
            st.markdown("### After Clustering (Dynamic)")
            if "ClusterID" in df_post.columns:
                # Parent: ClusterID, Child: ClientName
                graph_post = create_tree_graph(df_post, "ClusterID", "ClientName", "Edge")
                st.graphviz_chart(graph_post)
            else:
                st.info("Insufficient data for Post-clustering tree.")

    else:
        st.error("Required distribution CSV files not found in the selected logs folder. Run `simulate.py` first.")