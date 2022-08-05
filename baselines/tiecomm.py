import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import networkx as nx
import argparse
from modules.graph import measure_strength




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

        self.pooling_avg = nn.AdaptiveAvgPool1d(1)
        self.pooling_max = nn.AdaptiveMaxPool1d(1)


    def random_set(self):
        G = nx.binomial_graph(self.n_agents, self.random_prob, seed=self.seed , directed=False)
        set = self.god.graph_partition(G)
        return set



    def communicate(self, local_obs, set):

        local_obs = self.agent.local_emb(local_obs)
        #
        num_coms = len(set)
        #
        intra_obs = torch.zeros_like(local_obs)
        inter_obs = torch.zeros_like(local_obs)
        #
        # # inter_obs
        if num_coms == 1:
            intra_obs = self.intra_com(local_obs)
        else:
              group_emd_list = []
              for i in range (num_coms):
                  group_emd_list = []
                  for i in range(num_coms):
                      group_id_list = set[i]
                      group_obs = local_obs[group_id_list, :]
                      group_att = self.intra_com(group_obs)
                      group_emd = self.group_pooling(group_obs, mode='max')
                      group_emd_list.append(group_emd)
                      intra_obs[group_id_list, :] = group_att
              group_emd_list = self.inter_com(torch.cat(group_emd_list,dim=0))
              for index, group_ids in enumerate (set):
                  inter_obs[group_ids, :] = group_emd_list[index,:].repeat(len(group_ids), 1)

        if self.block == 'no':
            after_comm = torch.stack((local_obs, inter_obs, intra_obs), dim=1)
        elif self.block == 'inter':
            after_comm = torch.stack((local_obs, intra_obs), dim=1)
        elif self.block == 'intra':
            after_comm = torch.stack((local_obs, inter_obs), dim=1)
        else:
            raise ValueError('block must be one of no, inter, intra')
        return after_comm


    def group_pooling(self, input, mode):
        if mode == 'mean':
            group_emb = self.pooling_avg(input)
            #group_emb = torch.mean(input, dim=0).unsqueeze(0)
        elif mode == 'max':
            #group_emb = self.pooling_max(input)
            group_emb = torch.max(input, dim=0).values.unsqueeze(0)
        elif mode == 'sum':
            group_emb = torch.sum(input, dim=0).unsqueeze(0)
        else:
            raise ValueError('mode must be one of mean, max, sum')
        return group_emb



    def intra_com(self, input):
        weighted_emb = self.agent.intra_tf(input.unsqueeze(0))


        # hidden = self.agent.intra_fc(input)
        # score = torch.softmax(hidden, dim=0)
        # weighted_emb = score * input
        #weighted_emb, _  = self.agent.intra_attn(input.unsqueeze(0), input.unsqueeze(0), input.unsqueeze(0))
        return weighted_emb.squeeze(0)
        #return weighted_emb



    def inter_com(self, input):
        weighted_emb = self.agent.intra_tf(input.unsqueeze(0))
        # hidden = self.agent.inter_fc(input)
        # score = torch.softmax(hidden, dim=0)
        # weighted_emb = score * input
        #weighted_emb,_ = self.agent.inter_attn(input.unsqueeze(0), input.unsqueeze(0), input.unsqueeze(0))
        return weighted_emb.squeeze(0)
        #return weighted_emb



    # def group_embedding_sum(self, local_obs):
    #     score = torch.sum(local_obs,dim=0).unsqueeze(0)
    #     x = score.unsqueeze(0)
    #     embedding = self.agent.group_embedding_layer(x)
    #     return embedding





class AgentAC(nn.Module):
    def __init__(self, args):
        super(AgentAC, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.hid_size = args.hid_size

        self.tanh = nn.Tanh()


        self.emb_fc = nn.Linear(args.obs_shape, self.hid_size, bias=True)



        self.intra_fc = nn.Linear(self.hid_size, self.hid_size, bias=False)
        self.intra_attn = nn.MultiheadAttention(self.hid_size, 1, batch_first=True)

        intra_tf_layer = nn.TransformerEncoderLayer(d_model=self.hid_size, nhead=4, dim_feedforward=self.hid_size,
                                                         batch_first=True)
        self.intra_tf = nn.TransformerEncoder(intra_tf_layer, num_layers=1)


        self.inter_fc = nn.Linear(self.hid_size, self.hid_size, bias=False)
        self.inter_attn = nn.MultiheadAttention(self.hid_size, 1, batch_first=True)

        inter_tf_layer = nn.TransformerEncoderLayer(d_model=self.hid_size, nhead=4, dim_feedforward=self.hid_size,
                                                         batch_first=True)
        self.inter_tf = nn.TransformerEncoder(inter_tf_layer, num_layers=1)


        self.final_attn = nn.MultiheadAttention(self.hid_size, 1, batch_first=True)

        final_tf_layer = nn.TransformerEncoderLayer(d_model=self.hid_size, nhead=4, dim_feedforward=self.hid_size,
                                                         batch_first=True)
        self.final_tf = nn.TransformerEncoder(final_tf_layer, num_layers=1)




        if self.args.block == 'no':
            self.actor_fc1 = nn.Linear(self.hid_size * 3, self.hid_size)
            self.value_fc1 = nn.Linear(self.hid_size * 3, self.hid_size)
        else:
            self.actor_fc1 = nn.Linear(self.hid_size * 2, self.hid_size)
            self.value_fc1 = nn.Linear(self.hid_size * 2, self.hid_size)


        self.actor_head = nn.Linear(self.hid_size, args.n_actions)
        self.value_head = nn.Linear(self.hid_size, 1)


    def local_emb(self, input):
        local_obs = self.tanh(self.emb_fc(input))
        return local_obs



    def final_attn(self, after_comm):
        attn_output, attn_output_weights = self.final_attn(after_comm, after_comm, after_comm)
        #attn_output = sum([attn_output,after_comm])
        final_obs = torch.flatten(attn_output, start_dim=1, end_dim=-1)
        return final_obs



    def forward(self, after_comm):
        # h = self.final_tf(after_comm)

        final_obs = after_comm.flatten(start_dim=1, end_dim=-1)
        # final_obs = self.final_attn(after_comm)
        # final_obs =h.flatten(start_dim=1, end_dim=-1)

        a = self.tanh(self.actor_fc1(final_obs))
        a = F.log_softmax(self.actor_head(a), dim=-1)

        v = self.tanh(self.value_fc1(final_obs))
        v = self.value_head(v)

        return a, v



class GodAC(nn.Module):
    def __init__(self, args):
        super(GodAC, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.hid_size = args.hid_size
        self.threshold = self.args.threshold

        self.fc1 = nn.Linear(args.obs_shape, self.hid_size)
        self.multihead_attn = nn.MultiheadAttention(self.hid_size, 8, batch_first=True)
        self.fc2 = nn.Linear(self.hid_size * 4, self.hid_size)
        self.fc3 = nn.Linear(self.hid_size, 2)

        self.value_fc1 = nn.Linear(self.hid_size * self.n_agents, self.hid_size * 2)
        self.value_fc2 = nn.Linear(self.hid_size * 2, self.hid_size)
        self.value_fc3 = nn.Linear(self.hid_size, 1)


        index = list(range(self.n_agents * self.n_agents))
        index = np.array(index).reshape((self.n_agents, self.n_agents))
        index = np.triu(index, 1)
        index = index.flatten().tolist()
        self.index = [value for value in index if value != 0]

        self.i_lower = np.tril_indices(self.n_agents, -1)

        self.tanh = nn.Tanh()

        # self.batch_size = args.batch_size


    def forward(self, inputs):

        hid = self.tanh(self.fc1(inputs))
        attn_output, attn_weights = self.multihead_attn(hid.unsqueeze(0), hid.unsqueeze(0), hid.unsqueeze(0))

        h = torch.cat([attn_output.squeeze(0), hid], dim=1)

        matrixs = torch.cat([h.repeat(1, self.n_agents).view(self.n_agents * self.n_agents, -1),
                                     h.repeat(self.n_agents, 1)], dim=1)
        matrixs = matrixs[self.index, :]

        x = self.tanh(self.fc2(matrixs))



        log_action_out = F.log_softmax(self.fc3(x), dim=-1)
        relation = torch.multinomial(log_action_out .exp(), 1).squeeze(-1).detach()
        adj_matrix = self._generate_adj(relation)
        G = nx.from_numpy_matrix(adj_matrix)
        set = self.graph_partition(G)

        value = self.tanh(self.value_fc1(hid.unsqueeze(0).flatten(start_dim=1, end_dim=-1)))
        value = self.tanh(self.value_fc2(value))
        value = self.value_fc3(value)


        return set, log_action_out, value, relation


    def graph_partition(self, G):
        g = nx.Graph()
        g.add_nodes_from(G.nodes(data=False))

        for e in G.edges():
            strength = measure_strength(G, e[0], e[1])
            if strength > self.threshold:
                g.add_edge(e[0], e[1])

        set = [list(c) for c in nx.connected_components(g)]
        return set



    def _generate_adj(self, relation):
        adj_matrixs = torch.zeros(self.n_agents * self.n_agents,1, dtype=torch.long)
        adj_matrixs[self.index,:] = relation.unsqueeze(-1)
        adj_matrixs = adj_matrixs.view(self.n_agents, self.n_agents).detach().numpy()
        adj_matrixs[self.i_lower] = adj_matrixs.T[self.i_lower]
        return adj_matrixs







