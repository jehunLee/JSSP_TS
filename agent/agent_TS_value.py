from agent_TS import *
import os
from network import Hetero_GNN_type_aware_all_pred_value
from agent_BC3 import AgentBC


class AgentTS_v(AgentBC):
    def __init__(self, model_i=0):
        super().__init__(model_i)
        self.model_value_load()

    # rollout ###################################################################################
    def rollout_rules(self, curr_envs, rules=[]):
        """
        perform for an env
        return: cum_r, avg_run_t, avg_decision_t, decision_n, trajectory
        """
        agent_type = configs.agent_type
        configs.agent_type = 'rule'

        env = JobShopEnv(pomo_n=len(rules), load_envs=curr_envs)
        obs, reward, done = env.get_obs(), None, env.done()
        actions = list()

        depth = 0
        while not done:  # and depth < configs.rollout_depth:
            a = env.get_action_rule(obs, rules)
            obs, reward, done = env.step(a)

            actions.append(a)
            depth += 1

        configs.agent_type = agent_type

        # if not done:
        #     reward = -env.get_LB()

        return reward, actions

    def rollout_model_once(self, curr_envs, model):
        """
        perform for an env
        return: cum_r, avg_run_t, avg_decision_t, decision_n, trajectory
        """
        env = JobShopEnv(pomo_n=1, load_envs=curr_envs)
        obs, reward, done = env.get_obs(), None, env.done()
        actions = list()

        model.eval()
        with torch.no_grad():
            depth = 0
            while not done and depth < configs.rollout_depth:
                a = self.get_action_model(obs, model=model)
                obs, reward, done = env.step(a)

                actions.append(torch.concat(a))
                depth += 1

        if not done:
            reward = -self.get_opt_value(env)

        return reward, actions

    def rollout_models(self, curr_envs, models):
        """
        perform for an env
        return: cum_r, avg_run_t, avg_decision_t, decision_n, trajectory
        """
        env = JobShopEnv(pomo_n=len(models), load_envs=curr_envs)
        obs_list, reward, done = env.get_obs(), None, env.done()
        actions = list()

        for model in models:
            model.eval()
        with torch.no_grad():
            depth = 0
            action_t = 0
            transit_t = 0
            while not done and depth < configs.rollout_depth:
                s_t = time.time()
                a_list = self.get_action_models(obs_list, models=models, env_n=len(curr_envs))
                action_t += time.time()-s_t
                s_t = time.time()
                obs_list, reward, done = env.step(a_list)
                transit_t += time.time()-s_t

                actions.append(torch.concat(a_list))
                depth += 1

        if not done:
            reward = -env.get_LB()

        return reward, actions

    def get_action_beam(self, env, existing_tree=None, problem=None,
                        rules=None, models=None) -> (float, list, list, float, list):
        """
        perform for an env
        return: cum_r, avg_run_t, avg_decision_t, decision_n, trajectory
        """
        remain_done_nodes = list()
        remain_non_done_nodes = list()

        obs = env.get_obs()
        done = env.done()

        # initial #########################################################################
        s_t = time.time()
        if existing_tree:
            non_done_nodes, done_nodes, _, _ = existing_tree
            depth = configs.depth - 1
        else:
            node = Node(env, obs=obs, done=done)  # initial expansion
            # if node.done:
            #     return 0, [], [], node

            non_done_nodes = self.initial_rollout([node], rules, models)  # simulation
            # UB = non_done_nodes[0].UB
            done_nodes = list()
            depth = 0

        # loop ########################################################################################
        while non_done_nodes and depth < configs.depth:
            depth += 1

            # bounded expansion ###########################################
            non_done_nodes, new_done_nodes, _ = self.expansion_leaf(non_done_nodes, 0)

            if new_done_nodes:
                done_nodes += new_done_nodes
                done_nodes.sort(key=lambda x: x.UB)
                done_nodes = done_nodes[:configs.beam_n]  # too many -> cut

            if not non_done_nodes:  # no more expansion -> break
                remain_nodes = new_done_nodes
                break

            # simulation ##################################################
            non_done_nodes = self.initial_rollout(non_done_nodes, rules, models)
            non_done_nodes.sort(key=lambda x: x.UB)

            # separation  #################################################
            remain_done_nodes.clear()
            remain_non_done_nodes.clear()

            if 'terminal' in configs.pruning_type:
                remain_done_nodes = done_nodes
            remain_non_done_nodes = non_done_nodes

            # pruning  ####################################################
            remain_nodes = remain_done_nodes + remain_non_done_nodes
            for node in remain_nodes:
                node.update_value()
            remain_nodes.sort(key=lambda x: (x.value, x.done))  # priorities non_done nodes

            if configs.beam_n >= len(remain_done_nodes) + len(remain_non_done_nodes):
                non_done_nodes = remain_non_done_nodes

            else:
                non_done_nodes = []
                for node in remain_nodes[:configs.beam_n]:
                    if not node.done:
                        non_done_nodes.append(node)

        run_t = round(time.time() - s_t, 4)

        return run_t, non_done_nodes, done_nodes, remain_nodes[0]

    def run_episode_with_beam2(self, problem=None, rules=None, models=None) -> (float, float, float, float, list):
        """
        perform for an env
        return: cum_r, avg_run_t, avg_decision_t, decision_n, trajectory
        """
        # initial environment #########################################################################
        env = JobShopEnv([problem], pomo_n=1)
        obs, reward, done = env.reset()

        with torch.no_grad():
            configs.total_extend_n = 0
            configs.real_extend_n = 0
            configs.bound_rates = list()

            run_ts = list()

            while not done:
                run_t, non_done_nodes, done_nodes, best_node = self.get_action_beam(env, rules=rules, models=models)

                if best_node.route:
                    a = best_node.route.pop(0).view(1, -1)
                else:
                    a = best_node.best_follow[0].view(1, -1)
                    best_node.best_follow = best_node.best_follow[1:]

                run_ts.append(run_t)

                obs, reward, done = env.step([a])

        return -reward.item(), sum(run_ts), len(run_ts), \
               configs.total_extend_n, configs.real_extend_n, configs.bound_rates


if __name__ == '__main__':
    from utils import TA_small

    save_path = f'./../result/model_beam_test.csv'
    # configs.value_learn_type = 'LB_norm'
    configs.value_learn_type = ''
    configs.model_value_type = 'mm_global'

    configs.pruning_type = 'terminal'
    configs.expansion_type = 'all'

    for configs.depth in [1]:  # 3, 5
        for configs.rollout_depth in [1]:  # 3, 5,
            if configs.rollout_depth < 1e6 and 'bound' in configs.expansion_type:
                continue

            for configs.rollout_n in [1]:
                for configs.value_type in ['best']:  # quantile_LB, best, mean, norm, beta, t, chi2
                    if configs.rollout_n == 1 and configs.value_type != 'best':
                        continue

                    for configs.beam_n in [5]:  # 5
                        if 'with_best' in configs.pruning_type and configs.beam_n == 1 and configs.value_type != 'best':
                            continue

                        for configs.value_learn_type in ['LB_norm', '']:  #, 'prt_norm']:

                            for configs.rollout_type in ['multi_model']:  # model_rule, rule, multi_model
                                if 'multi_model' in configs.rollout_type and configs.rollout_n == 1:
                                    configs.rollout_type = 'model'

                                print(configs.depth, configs.rollout_depth, configs.rollout_n, configs.value_type,
                                      configs.beam_n, configs.rollout_type)
                                agent = AgentTS_v()
                                # agent.perform_beam_benchmarks([['TA', 15, 15, list(range(10))]], save_path=save_path)
                                agent.perform_beam_benchmarks([['HUN', 6, 4, list(range(3, 10))]], save_path=save_path)