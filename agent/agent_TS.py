from params import configs
import torch, time
from agent_GNN import AgentGNN
from environment.env import JobShopEnv
from utils import get_quantile_beta, get_quantile_norm, get_quantile_weighted


class Node():
    def __init__(self, curr_env, obs=None, done=False, route=[], best_follow=[]):
        self.curr_env = curr_env

        if obs:
            self.obs = obs
        else:
            obs = curr_env.get_obs()
            # if 'multi_model' in configs.rollout_type and obs != 0:
            #     obs = obs[0]
            self.obs = obs

        if done:
            self.done = done
        else:
            self.done = curr_env.done()

        self.LB = curr_env.get_LB()[0, 0].item()
        self.UBs = list()
        self.value = -1
        self.route = route
        self.best_follow = best_follow

    @property
    def UB(self):
        if self.UBs:
            return min(self.UBs)
        elif self.done:
            return self.LB
        else:
            return float('inf')

    def update_value(self):
        if self.done or configs.value_type == 'best' or not self.UBs:
            self.value = self.UB
        else:
            if 'best' in configs.pruning_type and configs.beam_n == 1:
                self.value = 0
            else:
                if 'best' in configs.value_type:
                    self.value = self.UB
                elif 'mean' in configs.value_type:
                    self.value = sum(self.UBs) / len(self.UBs)
                elif 'beta' in configs.value_type:
                    self.value = get_quantile_beta(self.UBs)
                elif 'norm' in configs.value_type:
                    self.value = get_quantile_norm(self.UBs)
                elif 'quantile_w' in configs.value_type:
                    self.value = get_quantile_weighted(self.UBs)
                else:
                    self.value = self.UB


class AgentTS(AgentGNN):
    def __init__(self, model_i=0):
        super().__init__(model_i)
        self.model_load()

    # rollout ###################################################################################
    def rollout_rules(self, curr_envs, rules=[]):
        """
        perform for an env
        return: cum_r, avg_run_t, avg_decision_t, decision_n, trajectory
        """
        agent_type = configs.agent_type
        # action_type = configs.action_type
        configs.agent_type = 'rule'
        # configs.action_type = 'buffer'

        env = JobShopEnv(pomo_n=len(rules), load_envs=curr_envs)
        obs, reward, done = env.get_obs(), None, env.done()

        while not done:
            a = env.get_action_rule(obs, rules)
            obs, reward, done = env.step(a)

        configs.agent_type = agent_type
        # configs.action_type = action_type

        return reward

    def rollout_model_once(self, curr_envs, model=None):
        """
        perform for an env
        return: cum_r, avg_run_t, avg_decision_t, decision_n, trajectory
        """
        if not model:
            model = self.model
        env = JobShopEnv(pomo_n=1, load_envs=curr_envs)

        model.eval()
        with torch.no_grad():
            obs, reward, done = env.get_obs(), None, env.done()
            while not done:
                a = self.get_action_model(obs, model=model)
                obs, reward, done = env.step(a)

        return reward

    def rollout_models(self, curr_envs, models):
        """
        perform for an env
        return: cum_r, avg_run_t, avg_decision_t, decision_n, trajectory
        """
        if not models:
            return self.rollout_model_once(curr_envs)
        env = JobShopEnv(pomo_n=len(models), load_envs=curr_envs)

        for model in models:
            model.eval()

        with torch.no_grad():
            obs_list, reward, done = env.get_obs(), None, env.done()
            while not done:
                a_list = self.get_action_models(obs_list, models=models, env_n=len(curr_envs))
                obs_list, reward, done = env.step(a_list)

        return reward

    def initial_rollout(self, nodes, rules=None, models=None):
        # current model ######
        if 'multi_model' in configs.rollout_type:
            cum_r = self.rollout_models([node.curr_env for node in nodes], models)
            for i, node in enumerate(nodes):
                node.UBs += [-r.item() for r in cum_r[i, :]]

        elif 'model' in configs.rollout_type:
            cum_r = self.rollout_model_once([node.curr_env for node in nodes])
            for i, node in enumerate(nodes):
                node.UBs.append(-cum_r[i, 0].item())

        # rules ######
        if 'rule' in configs.rollout_type and rules:
            cum_r = self.rollout_rules([node.curr_env for node in nodes], rules=rules)
            for i, node in enumerate(nodes):
                node.UBs += [-r.item() for r in cum_r[i, :]]

        # update_value ######
        for node in nodes:
            node.update_value()
        return nodes

    # run episode ###################################################################################
    def expansion_leaf(self, leaf_nodes, UB):
        new_done_nodes = list()
        new_non_done_nodes = list()
        total_n = 0
        UB_update = False

        for node in leaf_nodes:
            if node.done:
                new_done_nodes.append(node)
                continue

            candidate_a = node.obs['op_mask'].x.nonzero(as_tuple=True)[0]  # all actions
            # expansion with beam_n
            if 'all' not in configs.expansion_type:
                candidate_a = configs.beam_n  # nodes with highest prob.

            # expansion #################################################

            for a in candidate_a:
                a = node.obs['op_remain'][a]
                env = JobShopEnv(pomo_n=1, load_envs=[node.curr_env])
                _, reward, done = env.step([a.view(1).to('cpu')])

                node_ = Node(env, obs=None, done=done, route=node.route + [a.item()],
                             best_follow=node.best_follow)

                if done:
                    node_.update_value()
                    new_done_nodes.append(node_)
                    if node_.LB < UB:
                        UB = node_.LB
                        UB_update = True
                else:
                    total_n += 1
                    if 'bound' in configs.expansion_type and node_.LB > UB:  # bound
                        continue
                    new_non_done_nodes.append(node_)

        if UB_update and 'bound' in configs.expansion_type:
            new_non_done_nodes = [node for node in new_non_done_nodes if node.LB <= UB]

        if total_n:
            real_n = len(new_non_done_nodes)
            configs.total_extend_n += total_n
            configs.real_extend_n += real_n
            configs.bound_rates.append(1 - round(real_n / total_n, 3))

        return new_non_done_nodes, new_done_nodes, UB

    def run_episode_beam(self, problem, rules=None, models=None) -> (float, float, float, float, list):
        """
        perform for an env
        return: cum_r, avg_run_t, avg_decision_t, decision_n, trajectory
        """
        s_t = time.time()

        configs.total_extend_n = 0
        configs.real_extend_n = 0
        configs.bound_rates = list()

        # initial environment #########################################################################
        env = JobShopEnv([problem], pomo_n=1)
        obs, reward, done = env.reset()
        # if 'multi_model' in configs.rollout_type:
        #     obs = obs[0]

        node = Node(env, obs=obs, done=done)  # initial expansion
        non_done_nodes = self.initial_rollout([node], rules, models)  # simulation
        UB = non_done_nodes[0].UB

        depth = 1
        remain_done_nodes = list()
        new_non_done_nodes = list()
        if non_done_nodes[0].done:
            best_non_done_nodes = list()
            best_done_nodes = [non_done_nodes[0]]
            done_nodes = [non_done_nodes[0]]
        else:
            best_non_done_nodes = [non_done_nodes[0]]
            best_done_nodes = list()
            done_nodes = list()

        # loop ########################################################################################
        while non_done_nodes:
            depth += 1
            if depth > configs.depth:
                break
            # print(depth, '------------------------------')

            # bounded expansion #########################
            new_non_done_nodes, new_done_nodes, UB = self.expansion_leaf(non_done_nodes, UB)

            if new_done_nodes:
                done_nodes += new_done_nodes
                done_nodes.sort(key=lambda x: x.UB)
                done_nodes = done_nodes[:configs.beam_n]

            if not new_non_done_nodes:  # no more expansion -> break
                break

            # simulation ################################
            new_non_done_nodes = self.initial_rollout(new_non_done_nodes, rules, models)

            new_non_done_nodes.sort(key=lambda x: x.UB)
            if new_non_done_nodes[0].UB < UB:
                UB = new_non_done_nodes[0].UB

            # separation  ##################################
            best_done_nodes.clear()
            best_non_done_nodes.clear()
            remain_done_nodes.clear()

            if 'with_best' in configs.pruning_type:
                if 'terminal' in configs.pruning_type:
                    pos = len(done_nodes)
                    for i, node in enumerate(done_nodes):
                        if node.UB > UB:
                            pos = i
                            break
                    best_done_nodes = done_nodes[:pos]
                    remain_done_nodes = done_nodes[pos:]

                pos = len(new_non_done_nodes)
                for i, node in enumerate(new_non_done_nodes):
                    if node.UB > UB:
                        pos = i
                        break
                best_non_done_nodes = new_non_done_nodes[:pos]
                remain_non_done_nodes = new_non_done_nodes[pos:]

            else:
                if 'terminal' in configs.pruning_type:
                    remain_done_nodes = done_nodes
                remain_non_done_nodes = new_non_done_nodes

            # pruning  ##################################
            if configs.beam_n <= len(best_done_nodes) + len(best_non_done_nodes):  # 'with_best' in configs.pruning_type
                if not best_non_done_nodes:  # configs.beam_n <= len(best_done_nodes)
                    break
                non_done_nodes = best_non_done_nodes[:configs.beam_n]
            else:
                non_done_nodes = best_non_done_nodes

                remain_nodes = remain_done_nodes + remain_non_done_nodes
                remain_nodes.sort(key=lambda x: (x.value, x.done))  # priorities non_done nodes
                for node in remain_nodes[:configs.beam_n - len(best_non_done_nodes)]:
                    if not node.done:
                        non_done_nodes.append(node)

        run_t = round(time.time() - s_t, 4)
        # env.show_gantt_plotly(0, 0)

        all_nodes = done_nodes + new_non_done_nodes
        if not all_nodes:
            all_nodes = best_non_done_nodes
        all_nodes.sort(key=lambda x: (x.value, -x.done))  # priorities done nodes

        best_node = all_nodes[0]

        return best_node.UB, run_t, configs.total_extend_n, configs.real_extend_n, configs.bound_rates, \
            best_node.curr_env.decision_n[0, 0].item()

    def perform_beam_benchmarks(self, benchmarks, save_path='') -> None:
        """
        perform for envs
        """
        import csv
        from tqdm import tqdm
        from utils import get_model_i_list, get_rules

        rules = get_rules(configs.rollout_n)
        models = None
        if 'multi_model' in configs.rollout_type:
            model_i_list = get_model_i_list(configs.rollout_n)
            models = self.models_load(model_i_list)
        elif 'model' in configs.rollout_type:
            self.model_load()
            self.model.to(configs.device)

        for (benchmark, job_n, mc_n, instances) in benchmarks:
            print(benchmark, job_n, mc_n, len(instances))

            for i in tqdm(instances):
                reward, run_t, total_n, real_n, bound_rates, decision_n = self.run_episode_beam((
                    benchmark, job_n, mc_n, i), rules=rules, models=models)
                with open(save_path, 'a', newline='') as f:
                    wr = csv.writer(f)
                    rate, mean_bound_rate = 0, 0
                    if total_n:
                        rate = 1 - round(real_n / total_n, 3)
                    if bound_rates:
                        mean_bound_rate = round(sum(bound_rates) / len(bound_rates), 3)
                    wr.writerow([benchmark, job_n, mc_n, i, 'beam', configs.action_type,
                                 configs.beam_n, configs.expansion_type, configs.value_type, configs.pruning_type,
                                 configs.q_level, configs.rollout_n, configs.rollout_type, configs.depth,
                                 reward, run_t, total_n, real_n, rate, mean_bound_rate,
                                 decision_n, round(run_t / decision_n, 4)])


if __name__ == '__main__':
    from utils import TA_small

    save_path = f'./../result/model_beam0.csv'
    configs.expansion_type = 'bound_all'

    for configs.rollout_n in [3, 5]:
        for configs.value_type in ['best']:  # quantile_w, best, mean, norm, beta, t, chi2
            if configs.rollout_n == 1 and configs.value_type != 'best':
                continue

            for configs.beam_n in [1, 3, 5]:  # 5
                for configs.rollout_type in ['multi_model']:  # model_rule, rule, multi_model
                    if 'multi_model' in configs.rollout_type and configs.rollout_n == 1:
                        continue

                    agent = AgentTS()
                    agent.perform_beam_benchmarks(TA_small, save_path=save_path)

