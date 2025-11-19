import streamlit as st
import pandas as pd
import glob
import os
import altair as alt

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
        .properties(height=350, width={"step": 80})
        .interactive()
    )
    return chart


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


# Central
st.header("🌐 Central Server")
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
st.header("🏗️ Edge Servers")
edge_logs = glob.glob(os.path.join(selected_logs, "edge", "*.log"))
# sort by base filename without extension
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
st.header("👥 Clients")
client_logs = glob.glob(os.path.join(selected_logs, "clients", "*.log"))
clients = {}
for path in client_logs:
    fname = os.path.basename(path)
    cid = fname.split("_")[0]
    # classify as train/test based on filename
    split = "train" if "train" in fname else "test"
    clients.setdefault(cid, {})[split] = path

if clients:
    for cid in sorted(clients.keys()):
        paths = clients[cid]
        st.subheader(f"Client: {cid}")
        parts = []
        # prefer train then test for consistent order
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
