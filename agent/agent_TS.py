import copy

from params import configs
import pickle, torch, os, time

from torch.distributions import Categorical
from utils import get_x_dim
from network import GNN, Hetero_GNN_type_aware_all_pred, Hetero_GNN_type_aware


class AgentTS():
    def __init__(self, model_i=0):
        self.model_i = model_i
        self.agent_type = configs.agent_type
        self.update_seed()

        # input dim, output dim ########################
        configs.in_dim_op, configs.in_dim_rsc = get_x_dim()

        # model generation ########################
        if 'multi_policy' in self.agent_type:
            self.model = self.get_model('multi_policy')
        elif 'policy' in self.agent_type:
            self.model = self.get_model('policy')
        elif 'actor_critic' in self.agent_type:
            self.model = self.get_model('actor_critic')
        elif 'value' in self.agent_type:
            self.model = self.get_model('value')
        else:
            print("Unknown agent_type.")

        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()

    # model value ####################################################################################
    def get_(self, obs, model=None):
        """
        get greedy action with the highest value
        """
        if not model:
            model = self.model
        probs = model(obs)

        actions = list()
        graph_index = obs['op'].batch
        for batch_i, prob in enumerate(probs):
            if not prob.size()[0]:
                actions.append(torch.tensor([0], dtype=torch.int64).to(configs.device))
                continue

            a_tensor = torch.argmax(prob).view(-1)
            indices = torch.where(graph_index == batch_i)
            actions.append(obs['op_remain'][indices][a_tensor])

        return actions



if __name__ == '__main__':
    from utils import all_benchmarks, HUN_100

    configs.agent_type = 'GNN_BC_policy'
    configs.env_type = 'dyn'  # 'dyn', ''
    configs.state_type = 'mc_gap_mc_load_prt_norm'  # 'basic_norm', 'simple_norm', 'norm', 'mc_load_norm'
    configs.model_type = 'type_all_pred_GAT2'  # 'type_all_pred_GAT2'

    configs.training_len = len(HUN_100[0][3])

    for configs.action_type in ['conflict']:
        for i in range(5):
            agent = AgentGNN(model_i=i)

            save_path = f'./../result/bench_model_{i}.csv'
            agent.perform_model_benchmarks(all_benchmarks, save_path=save_path)
