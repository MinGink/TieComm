import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv,GCNConv
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



    def random_set(self):
        G = nx.binomial_graph(self.n_agents, self.random_prob, seed=self.seed , directed=False)
        set = self.graph_partition(G, 0.5)
        return G, set


    def graph_partition(self, G, god_action):

        #min_value = 0.5
        min_max_list = []
        for e in G.edges():
            strength = measure_strength(G, e[0], e[1])
            G.add_edge(e[0], e[1], weight = strength)
            min_max_list.append(strength)

        if min_max_list:
            min_max_list.sort()
            min_value = min_max_list[0]
            max_value = min_max_list[-1]
            thershold = ((max_value - min_value) / 10) * int(god_action[0]) + min_value
        else:
            thershold = 0.0

        # thershold = 2

        g = nx.Graph()
        g.add_nodes_from(G.nodes(data=False), node_strength =0.0)
        for e in G.edges():
            strength = G.get_edge_data(e[0], e[1])['weight']
            if strength >= thershold:
                g.nodes[e[0]]['node_strength'] += strength
                g.nodes[e[1]]['node_strength'] += strength
                g.add_edge(e[0], e[1])
            # print(strength)
            # raise ValueError('strength > thershold')

        attr_dict = nx.get_node_attributes(g, 'node_strength')
        set = []
        core_node = []
        for c in nx.connected_components(g):
            list_c = list(c)
            set.append(list_c)
            list_c_attr = [attr_dict[i] for i in list_c]
            core_node.append(list_c[list_c_attr.index(max(list_c_attr))])

        return g, (core_node, set)








    def communicate(self, local_obs, graph=None, node_set =None):

        core_node, set = node_set

        local_obs = self.agent.local_emb(local_obs)
        intra_obs = self.agent.intra_com(local_obs, graph)


        adj_matrix = torch.tensor(nx.to_numpy_array(graph), dtype=torch.float).view(1, -1).repeat(self.n_agents, 1)

        inter_obs = torch.zeros_like(intra_obs)
        if len(set) != 1:
            core_obs = intra_obs[core_node, :] #.clone().detach()
            group_obs = self.agent.inter_com(core_obs)
            for index, group_members in enumerate (set):
                inter_obs[group_members, :] = group_obs[index,:].repeat(len(group_members), 1)


        if self.block == 'no':
            #after_comm = (local_obs, inter_obs, intra_obs)
            after_comm = torch.cat((local_obs, intra_obs, inter_obs), dim=-1)
            #after_comm = torch.cat((local_obs,  inter_obs,  intra_obs, adj_matrix), dim=-1)
        elif self.block == 'inter':
            after_comm = torch.cat((local_obs,  intra_obs), dim=-1)
        elif self.block == 'intra':
            after_comm = torch.cat((local_obs, inter_obs), dim=-1)
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

        self.fc1_1 = nn.Linear(args.obs_shape * self.n_agents , self.hid_size * 1)
        self.fc1_2 = nn.Linear(self.n_agents**2 , self.hid_size)
        self.fc2 = nn.Linear(self.hid_size *2, self.hid_size)
        self.head = nn.Linear(self.hid_size, 10)
        self.value = nn.Linear(self.hid_size, 1)


    def forward(self, input, graph):

        adj_matrix = torch.tensor(nx.to_numpy_array(graph), dtype=torch.float).view(1, -1)
        h1 = self.tanh(self.fc1_1(input.view(1, -1)))
        h2 = self.tanh(self.fc1_2(adj_matrix))
        hid = torch.cat([h1,h2], dim=1)
        hid = self.tanh(self.fc2(hid))

        a = F.log_softmax(self.head(hid), dim=-1)
        v = self.value(hid)

        return a, v



class GodActor(nn.Module):
    def __init__(self, args):
        super(GodActor, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.hid_size = args.hid_size
        self.threshold = self.args.threshold
        self.tanh = nn.Tanh()

        self.fc1 = nn.Linear(args.obs_shape * self.n_agents + self.n_agents**2 , self.hid_size * 4)
        self.fc2 = nn.Linear(self.hid_size * 4 , self.hid_size)
        self.head = nn.Linear(self.hid_size, 10)

    def forward(self, input, graph):

        adj_matrix = torch.tensor(nx.to_numpy_array(graph), dtype=torch.float).view(1, -1)
        hid = torch.cat([input.view(1,-1), adj_matrix], dim=1)
        hid = self.tanh(self.fc1(hid))
        hid = self.tanh(self.fc2(hid))
        a = F.log_softmax(self.head(hid), dim=-1)

        return a



class GodCritic(nn.Module):
    def __init__(self, args):
        super(GodCritic, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.hid_size = args.hid_size
        self.threshold = self.args.threshold
        self.tanh = nn.ReLU()

        self.fc1 = nn.Linear(args.obs_shape * self.n_agents + self.n_agents ** 2, self.hid_size * 4)
        self.fc2 = nn.Linear(self.hid_size * 4 , self.hid_size)
        self.value = nn.Linear(self.hid_size, 1)

    def forward(self, input, graph):

        adj_matrix = torch.tensor(nx.to_numpy_array(graph), dtype=torch.float).view(1, -1)
        hid = torch.cat([input.view(1,-1), adj_matrix], dim=1)
        hid = self.tanh(self.fc1(hid))
        hid = self.tanh(self.fc2(hid))
        v = self.value(hid)

        return v




class AgentAC(nn.Module):
    def __init__(self, args):
        super(AgentAC, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.hid_size = args.hid_size
        self.n_actions = self.args.n_actions
        self.tanh = nn.Tanh()

        self.emb_fc = nn.Linear(args.obs_shape, self.hid_size)

        self.intra = GATConv(self.hid_size, self.hid_size, heads=1, add_self_loops =False, concat=False)
        #self.intra = GCNConv(self.hid_size, self.hid_size, add_self_loops= False)

        #encoder_layer = nn.TransformerEncoderLayer(d_model=self.hid_size, nhead=1, dim_feedforward=self.hid_size,
        #                                                batch_first=True)
        #self.inter = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.inter = nn.MultiheadAttention(self.hid_size, num_heads=1, batch_first=True)


        if self.args.block == 'no':
            self.affine2 = nn.Linear(self.hid_size *3 , self.hid_size)
        else:
            self.affine2 = nn.Linear(self.hid_size * 2, self.hid_size)

        self.actor_head = nn.Linear(self.hid_size, self.n_actions)
        self.value_head = nn.Linear(self.hid_size, 1)

    def local_emb(self, input):
        local_obs = self.tanh(self.emb_fc(input))
        return local_obs

    def intra_com(self, x, graph):

        if list(graph.edges()) == []:
            h = torch.zeros(x.shape)
        else:
            edge_index = torch.tensor(list(graph.edges()), dtype=torch.long)
            data = Data(x=x, edge_index=edge_index.t().contiguous())
            h = self.tanh(self.intra(data.x, data.edge_index))

        return h

    def inter_com(self, input):
        x = input.unsqueeze(0)
        h, weights = self.inter(x, x, x)
        #h = self.inter(x)

        return h.squeeze(0)




    def forward(self, final_obs):
        h = self.tanh(self.affine2(final_obs))
        a = F.log_softmax(self.actor_head(h), dim=-1)
        v = self.value_head(h)

        return a, v