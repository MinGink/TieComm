import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import  GATConv, GATv2Conv
import networkx as nx
import argparse
from modules.graph import measure_strength
from torch_geometric.data import Data



class TieCommAgent(nn.Module):

    def __init__(self, agent_config):
        super(TieCommAgent, self).__init__()

        self.args = argparse.Namespace(**agent_config)
        self.seed = self.args.seed

        self.n_agents = self.args.n_agents
        self.hid_size = self.args.hid_size

        self.agent = AgentAC(self.args)
        self.god = GodAC(self.args)

        if hasattr(self.args, 'random_prob'):
            self.random_prob = self.args.random_prob

        self.block = self.args.block
        self.threshold = self.args.threshold




    def communicate(self, local_obs, graph, node_set):

        core_node, set = node_set

        local_obs = self.agent.local_emb(local_obs)
        intra_obs = self.agent.intra_com(local_obs, graph)

        inter_obs = torch.zeros_like(local_obs)
        if len(set) != 1:
            core_obs = intra_obs[core_node, :]
            group_obs = self.agent.inter_com(core_obs)

            for index, group_ids in enumerate (set):
                inter_obs[group_ids, :] = group_obs[index,:].repeat(len(group_ids), 1)


        if self.block == 'no':
            after_comm = torch.cat((inter_obs,  intra_obs), dim=1)
        elif self.block == 'inter':
            after_comm = torch.cat((intra_obs), dim=1)
        elif self.block == 'intra':
            after_comm = torch.cat((inter_obs), dim=1)
        else:
            raise ValueError('block must be one of no, inter, intra')


        return after_comm






class GodAC(nn.Module):
    def __init__(self, args):
        super(GodAC, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.hid_size = args.hid_size
        self.threshold = self.args.threshold
        self.tanh = nn.Tanh()


        self.fc1 = nn.Linear(args.obs_shape, self.hid_size)
        self.fc2 = nn.Linear(self.hid_size * self.n_agents + self.n_agents**2, self.hid_size *2)
        self.head = nn.Linear(self.hid_size * 2, 10)
        self.value = nn.Linear(self.hid_size * 2, 1)


    def forward(self, input, graph):

        adj_matrix = torch.tensor(nx.to_numpy_array(graph), dtype=torch.float).view(1,-1)
        hid = self.tanh(self.fc1(input)).view(1,-1)
        hid = torch.cat((hid, adj_matrix), dim=1)
        hid = self.tanh(self.fc2(hid))

        a = F.log_softmax(self.head(hid), dim=-1)
        v = self.value(hid)
        return a, v






    def graph_partition(self, G, thershold):
        g = nx.Graph()
        g.add_nodes_from(G.nodes(data=False), node_strength = 0.0)
        for e in G.edges():
            strength = measure_strength(G, e[0], e[1])
            if strength > thershold:
                g.add_edge(e[0], e[1], strength = strength)
                g.nodes[e[0]]['node_strength'] += strength
                g.nodes[e[1]]['node_strength'] += strength

        attr_dict = nx.get_node_attributes(g, 'node_strength')
        set = []
        core_node = []
        for c in nx.connected_components(g):
            list_c = list(c)
            set.append(list_c)
            list_c_attr = [attr_dict[i] for i in list_c]
            core_node.append(list_c[list_c_attr.index(max(list_c_attr))])
        return g, (core_node, set)







class AgentAC(nn.Module):
    def __init__(self, args):
        super(AgentAC, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.hid_size = args.hid_size

        self.tanh = nn.Tanh()

        self.emb_fc = nn.Linear(args.obs_shape, self.hid_size)

        self.intra = GATv2Conv(self.hid_size, self.hid_size, heads=1)

        inter_layer = nn.TransformerEncoderLayer(d_model=self.hid_size, nhead=4, dim_feedforward=self.hid_size,
                                                 batch_first=True)
        self.inter = nn.TransformerEncoder(inter_layer, num_layers=1)


        if self.args.block == 'no':
            self.actor_fc1 = nn.Linear(self.hid_size * 2, self.hid_size)
            self.value_fc1 = nn.Linear(self.hid_size * 2, self.hid_size)
        else:
            self.actor_fc1 = nn.Linear(self.hid_size * 1, self.hid_size)
            self.value_fc1 = nn.Linear(self.hid_size * 1, self.hid_size)

        self.actor_head = nn.Linear(self.hid_size, args.n_actions)
        self.value_head = nn.Linear(self.hid_size, 1)

    def local_emb(self, input):
        local_obs = self.tanh(self.emb_fc(input))
        return local_obs

    def intra_com(self, x, graph):

        if list(graph.edges()) == []:
            edge_index = torch.zeros((1, 2), dtype=torch.long)
        else:
            edge_index = torch.tensor(list(graph.edges()), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index.t().contiguous())
        h = self.tanh(self.intra(data.x, data.edge_index))

        return h

    def inter_com(self, input):
        h = self.tanh(self.inter(input.unsqueeze(0)))
        return h.squeeze(0)



    def forward(self, after_comm):

        final_obs = after_comm

        a = self.tanh(self.actor_fc1(final_obs))
        a = F.log_softmax(self.actor_head(a), dim=-1)

        v = self.tanh(self.value_fc1(final_obs))
        v = self.value_head(v)

        return a, v