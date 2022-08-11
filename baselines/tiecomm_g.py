import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv
import networkx as nx
import argparse
from modules.graph import measure_strength
from torch_geometric.data import Data
from .tiecomm import TieCommAgent, AgentAC, GodAC


class TieCommAgentG(TieCommAgent):

    def __init__(self, agent_config):
        super(TieCommAgentG, self).__init__(agent_config)
        self.god = GodACG(self.args)







    def communicate(self, local_obs, graph, set):

        local_obs = self.agent.local_emb(local_obs)
        intra_obs = self.agent.intra_GNN(local_obs, graph)

        adj_matrix = torch.tensor(nx.to_numpy_array(graph), dtype=torch.float).view(1,-1).repeat(self.n_agents,1)


        num_coms = len(set)
        inter_obs = torch.zeros_like(local_obs)
        if num_coms != 1:
            group_emd_list = []
            for i in range (num_coms):
                group_emd_list = []
                for i in range(num_coms):
                    group_id_list = set[i]
                    group_obs = local_obs[group_id_list, :]
                    group_emd = self.group_pooling(group_obs, mode='sum')
                    group_emd_list.append(group_emd)
            group_emd_list = self.agent.inter_com(torch.cat(group_emd_list,dim=0))

            for index, group_ids in enumerate (set):
                inter_obs[group_ids, :] = group_emd_list[index,:].repeat(len(group_ids), 1)

        if self.block == 'no':
            after_comm = torch.concat((local_obs, inter_obs,  intra_obs), dim=1)
        elif self.block == 'inter':
            after_comm = torch.concat((local_obs, intra_obs), dim=1)
        elif self.block == 'intra':
            after_comm = torch.concat((local_obs, inter_obs), dim=1)
        else:
            raise ValueError('block must be one of no, inter, intra')


        # if self.block == 'no':
        #     after_comm = torch.concat((local_obs, inter_obs, intra_obs, adj_matrix), dim=1)
        # elif self.block == 'inter':
        #     after_comm = torch.concat((local_obs, intra_obs,adj_matrix), dim=1)
        # elif self.block == 'intra':
        #     after_comm = torch.concat((local_obs, inter_obs, adj_matrix), dim=1)
        # else:
        #     raise ValueError('block must be one of no, inter, intra')

        return after_comm






class GodACG(GodAC):
    def __init__(self, args):
        super(GodAC, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.hid_size = args.hid_size
        self.threshold = self.args.threshold
        self.tanh = nn.Tanh()


        self.fc1 = nn.Linear(args.obs_shape, self.hid_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hid_size + self.n_agents**2, nhead=4, dim_feedforward=self.hid_size,
                                                         batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)



        self.fc2 = nn.Linear((self.hid_size + self.n_agents**2) * self.n_agents, self.hid_size *2)

        self.head = nn.Linear(self.hid_size * 2, 10)
        self.value = nn.Linear(self.hid_size * 2, 1)


    def forward(self, input, graph):


        adj_matrix = torch.tensor(nx.to_numpy_array(graph), dtype=torch.float).view(1,-1).repeat(self.n_agents,1)
        hid = self.tanh(self.fc1(input))
        hid = torch.cat((hid, adj_matrix), dim=1)
        hid = self.transformer(hid.unsqueeze(0)).squeeze(0)
        hid = hid.reshape(1, -1)
        hid = self.tanh(self.fc2(hid))
        a = F.log_softmax(self.head(hid), dim=-1)
        v = self.value(hid)
        return a, v



    def graph_partition(self, G, thershold):
        g = nx.Graph()
        g.add_nodes_from(G.nodes(data=False))
        for e in G.edges():
            strength = measure_strength(G, e[0], e[1])
            if strength > thershold:
                g.add_edge(e[0], e[1])
        set = [list(c) for c in nx.connected_components(g)]
        return G, set

