import sys
sys.path.append('')

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from fa2 import ForceAtlas2
from curved_edges import curved_edges
import pandas as pd
import cudf
import cugraph

print('\n','Data Prep...','\n')

# Number of clusters
n = 3
loop_list = list(range(0,n))

colors = ['#FF3333','#33F3FF','#FF9333','#33FF36','#33FFC7','#33A8FF','#F333FF','#FF3361']
colors = colors[:n]

# Data preprocessing to remove non-connected nodes from network & begin relabeling nodes
def pre1(x):
    data = pd.read_csv(r'/home/natalie_bucklin/network/data/net_{}.csv'.format(x),sep=',',header=None,skiprows=1,names=['from','to'])

    from_count = data['from'].value_counts().to_frame().reset_index().rename(columns={'index':'id','from':'from_count'})
    to_count = data['to'].value_counts().to_frame().reset_index().rename(columns={'index':'id','to':'to_count'})

    merged = pd.merge(data,from_count,left_on='from',right_on='id',how='left').drop(['id'],axis=1)
    merged = pd.merge(merged,to_count,left_on='to',right_on='id',how='left').drop(['id'],axis=1)
    data = merged[(merged['from_count']>1) & (merged['to_count'] > 1)].drop(['from_count','to_count'],axis=1)

    data2 = pd.concat([data['from'],data['to']]).to_frame().rename(columns={0:'id'}).drop_duplicates()
    return data, data2
net_anti, id1 = pre1('anti')
net_nat, id2 = pre1('nat')

id_same = pd.merge(id1,id2,how='inner',on='id')

def pre2(x):
    data = pd.merge(x,id_same,how='outer',indicator=True)
    data2 = data[data['_merge']=='left_only'].drop(['_merge'],axis=1).drop_duplicates()
    return data2
id1_only, id2_only = pre2(id1), pre2(id2)

id_diff = pd.concat([id1_only,id2_only])

id_master = pd.concat([id_same,id_diff])
id_master = id_master.reset_index(drop=True).reset_index().rename(columns={'index':'new_id'})
id_master

# Mapping up new ids to networks
def pre3(x):
    data = pd.merge(x,id_master,left_on='from',right_on='id',how='left').drop(['id'],axis=1).rename(columns={'new_id':'new_from'})
    data = pd.merge(data,id_master,left_on='to',right_on='id',how='left').drop(['id'],axis=1).rename(columns={'new_id':'new_to'})

    data2 = data.drop(['from','to'],axis=1).rename(columns={'new_from':'from','new_to':'to'})

    return data, data2
net_nat_map, net_nat = pre3(net_nat)
net_anti_map, net_anti = pre3(net_anti)

print('\n','BFS...','\n')
def data(df):
    net = cudf.from_pandas(df)

    net['to'] = net['to'].astype('int32')
    net['from'] = net['from'].astype('int32')

    n = net.iloc[0,0]

    G = cugraph.Graph()
    G.add_edge_list(net['from'],net['to'],None)
    out_bfs = cugraph.bfs(G,n,directed=True)
    out_page = cugraph.pagerank(G)
    out_bfs = out_bfs.to_pandas()
    out_page = out_page.to_pandas()

    out_bfs.loc[out_bfs['distance'] < 3,'group'] = 2
    out_bfs.loc[out_bfs['distance'] == 3,'group'] = 0
    out_bfs.loc[out_bfs['distance'] > 3,'group'] = 1
    out_bfs = out_bfs[['vertex','group']]
    return out_bfs, out_page

bfs_nat, page_nat = data(net_nat)
bfs_anti, page_anti = data(net_anti)

net = pd.concat([net_nat,net_anti])
biggest_to = net['to'].value_counts().to_frame().reset_index().rename(columns={'index':'id','to':'count'})

bfs = pd.concat([bfs_nat,bfs_anti])
page = pd.concat([page_nat,page_anti])

map = pd.concat([net_nat_map,net_anti_map])
map_from = map[['from','new_from']].rename(columns={'from':'original_id','new_from':'new_id'})
map_to = map[['to','new_to']].rename(columns={'to':'original_id','new_to':'new_id'})
map = pd.concat([map_from,map_to]).drop_duplicates()


print('\n','Exporting Data...','\n')
bfs.to_csv('',sep=',',header=True)
page.to_csv('',sep=',',header=True)
map.to_csv('',sep=',',header=True)
biggest_to.to_csv('',sep=',',header=True)

print('\n','Force Atlas...','\n')
# Running Force Atlas
forceatlas2 = ForceAtlas2()
G = nx.from_pandas_edgelist(net,'from','to',create_using=nx.Graph())
positions = forceatlas2.forceatlas2_networkx_layout(G,pos=None,iterations=500)

positions_df = pd.DataFrame.from_dict(positions,orient='index')
positions_df.to_csv('',sep=',',header=False)

#positions_df = pd.read_csv('',sep=',',header=None)
#positions_df = positions_df.reset_index()
#positions = positions_df.set_index('index')[[0,1]].T.apply(tuple).to_dict()

print('\n','Curves...','\n')
lc_dict = {}
for n, c in zip(loop_list,colors):
# Making a new network based on selected cluster
    network_sub = pd.merge(net,bfs[bfs['group'] == n],left_on='to',right_index=True,how='inner').drop(['group'],axis=1).reset_index(drop=True)

    # Making a list that has the nodes we want to keep
    node_list = list(dict.fromkeys(network_sub['to'].tolist() + network_sub['from'].tolist()))
    node_list = set(node_list)

    # Getting the subset of the atlast positions dict
    positions_sub = {k:v for k, v in positions.items() if k in node_list}

    # Loading a new network based on the subset
    G_sub = nx.from_pandas_edgelist(network_sub,'from','to',create_using=nx.Graph())
    print(G_sub.number_of_nodes())

    curves = curved_edges(G_sub,positions_sub)
    lc = LineCollection(curves,color=c,alpha=.01)

    lc_dict[n] = lc

print('\n','Plotting...','\n')
# Plot
def plot():
    plt.figure(figsize=(19.20,10.80))
    plt.gca().set_facecolor('k')
    for key,value in lc_dict.items():
        plt.gca().add_collection(lc_dict[key])
    plt.tick_params(axis='both',which='both',bottom=False,left=False,labelbottom=False,labelleft=False)
    plt.axis('tight')
    plt.savefig(r'',format='png',dpi=1000)
plot()
