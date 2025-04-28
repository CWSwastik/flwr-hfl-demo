import streamlit as st
import pandas as pd
import glob
import os
import altair as alt

st.set_page_config(page_title="HFL Dashboard", layout="wide")


st.title("📊 HFL Training Dashboard")


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


# Central
st.header("🌐 Central Server")
central_log = "./logs/central/central_server.log"
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
edge_logs = glob.glob("./logs/edge/*.log")
if edge_logs:
    for path in edge_logs:
        name = os.path.basename(path).replace(".log", "")
        df_e = pd.read_csv(path)
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
client_logs = glob.glob("./logs/clients/*.log")
clients = {}
for path in client_logs:
    fname = os.path.basename(path)
    cid = fname.split("_")[0]
    clients.setdefault(cid, {})["train" if "train" in fname else "test"] = path

if clients:
    for cid, paths in clients.items():
        st.subheader(f"Client: {cid}")
        parts = []
        for split, p in paths.items():
            df = pd.read_csv(p)
            df["split"] = split.capitalize()
            parts.append(df)
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
