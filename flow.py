from graphviz import Digraph

dot = Digraph(comment='Hierarchical FL')
dot.attr(rankdir='TB', splines='ortho')

# Define styles
client_attr = {'style': 'filled', 'fillcolor': '#e1f5fe', 'color': '#01579b', 'penwidth': '2'}
edge_attr = {'style': 'filled', 'fillcolor': '#e8f5e9', 'color': '#2e7d32', 'penwidth': '2'}
cloud_attr = {'style': 'filled', 'fillcolor': '#fff3e0', 'color': '#ef6c00', 'penwidth': '2'}

# Cloud Subgraph
with dot.subgraph(name='cluster_cloud') as c:
    c.attr(label='CLOUD SERVER (Global Aggregation)', style='dashed')
    c.node('S_RECV', '9. Receive group models', **cloud_attr)
    c.node('S_AGG', '10. Compute Global Model\nx̄_(t+1)', **cloud_attr)
    c.node('S_UPD', '11. Update y_(t+1,j)', **cloud_attr)
    c.node('S_CAST', '13. Broadcast x̄_(t+1)', **cloud_attr)
    c.edge('S_RECV', 'S_AGG')
    c.edge('S_AGG', 'S_UPD')
    c.edge('S_UPD', 'S_CAST')

# Edge Subgraph
with dot.subgraph(name='cluster_edge') as e:
    e.attr(label='EDGE GROUPS (Group Aggregation)', style='dashed')
    e.node('G_RECV', '5. Receive client models\nCompute avg', **edge_attr)
    e.node('G_UPD', '6. Update z_(t,e+1,i)', **edge_attr)
    e.node('G_BUF', '7. Buffer group models', **edge_attr)
    e.node('G_SEND', '8. Send to Cloud (After E rounds)', **edge_attr)
    e.node('G_RECV_Y', '12. Receive y_(t+1,j)\nBroadcast', **edge_attr)
    e.edge('G_RECV', 'G_UPD')
    e.edge('G_UPD', 'G_BUF')
    e.edge('G_BUF', 'G_SEND')

# Client Subgraph
with dot.subgraph(name='cluster_client') as cl:
    cl.attr(label='CLIENTS (Local Training)', style='dashed')
    cl.node('C_INIT', '1. Receive Global Model', **client_attr)
    cl.node('C_ZINIT', '2. Initialize z', **client_attr)
    cl.node('C_LOOP', '3. Local SGD Loop', **client_attr)
    cl.node('C_SEND', '4. Send Local Model', **client_attr)
    cl.edge('C_INIT', 'C_ZINIT')
    cl.edge('C_ZINIT', 'C_LOOP')
    cl.edge('C_LOOP', 'C_SEND')

# Connections between layers
dot.edge('C_SEND', 'G_RECV', label=' Local Model')
dot.edge('G_SEND', 'S_RECV', label=' Group Model')
dot.edge('S_CAST', 'G_RECV_Y', label=' Global Update')
dot.edge('G_RECV_Y', 'C_INIT', label=' New Round')

# Render inside notebook
dot.render('hierarchical_fl_flowchart', view=True, format='png')