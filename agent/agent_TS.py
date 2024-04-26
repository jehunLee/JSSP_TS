from params import configs
import torch, time
from agent_GNN import AgentGNN
from environment.env import JobShopEnv
from utils import get_quantile_beta, get_quantile_norm, get_quantile_weighted, get_quantile_LB, \
    get_quantile_t, get_quantile_chi2, get_quantile_LB2


class Node():
    def __init__(self, curr_env, obs=None, done=False, route=[], best_follow=[]):
        self.curr_env = curr_env

        if obs:
            self.obs = obs
        else:
            self.obs = curr_env.get_obs()

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
        if self.done:
            self.value = self.LB
        elif 'best' in configs.pruning_type and configs.beam_n == 1:
            self.value = self.UB
        elif 'best' in configs.value_type:
            self.value = self.UB
        elif 'mean' in configs.value_type:
            self.value = sum(self.UBs) / len(self.UBs)
        elif 'beta' in configs.value_type:
            self.value = get_quantile_beta(self.UBs)
        elif 'norm' in configs.value_type:
            self.value = get_quantile_norm(self.UBs)
        elif 'quantile_w' in configs.value_type:
            self.value = get_quantile_weighted(self.UBs)
        elif 'quantile_LB2' in configs.value_type:
            self.value = get_quantile_LB2(self.LB, self.UBs)
        elif 'quantile_LB' in configs.value_type:
            self.value = get_quantile_LB(self.LB, self.UBs)
        elif 'chi2' in configs.value_type:
            self.value = get_quantile_chi2(self.UBs)
        elif 't' in configs.value_type:
            self.value = get_quantile_t(self.UBs)
        else:
            raise NotImplementedError


class AgentTS(AgentGNN):
    def __init__(self, model_i=0):
        super().__init__(model_i)
        if 'model' in configs.rollout_type:
            self.model_load()

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

                # print(depth)
                if depth == 83:
                    print()
                if 80 < depth and depth < 85:
                    print(depth, a)

            ###################################################
            if not done:
                for i, a in enumerate(actions):
                    actions[i] = a.to('cpu').view(-1, 1)

                agent_type = configs.agent_type
                configs.agent_type = 'rule'
                obs, reward, done = env.get_obs(), None, env.done()

                while not done:  # and depth < configs.rollout_depth:
                    a = env.get_action_rule(obs, ['LTT'])
                    obs, reward, done = env.step(a)

                    actions.append(a)
                    depth += 1

                configs.agent_type = agent_type
            ###################################################

        if not done:
            reward = -env.get_LB()

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

    def initial_rollout(self, nodes, rules=None, models=None):
        # model ######
        if 'multi_model' in configs.rollout_type:
            cum_r, actions = self.rollout_models([node.curr_env for node in nodes], models)
            a_ = torch.concat(actions).reshape(-1, len(models), len(nodes)).transpose(0, 2)
            for i, node in enumerate(nodes):
                j = cum_r[i, :].argmax(dim=0).item()
                node.best_follow = a_[i, j]
                node.UBs += [-r.item() for r in cum_r[i, :]]

        elif 'model' in configs.rollout_type:
            cum_r, actions = self.rollout_model_once([node.curr_env for node in nodes], self.model)
            a_ = torch.concat(actions).reshape(-1, len(nodes))
            for i, node in enumerate(nodes):
                node.UBs.append(-cum_r[i, 0].item())
                node.best_follow = a_[:, i]

        # rules ######
        if 'rule' in configs.rollout_type and rules:
            cum_r, actions = self.rollout_rules([node.curr_env for node in nodes], rules=rules)
            a_ = torch.concat(actions).reshape(-1, len(nodes), len(rules)).transpose(0, 2)
            for i, node in enumerate(nodes):
                j = cum_r[i, :].argmax(dim=0).item()
                if node.UBs:
                    if node.UB > -cum_r[i, j]:
                        node.best_follow = a_[j, i]
                else:
                    node.best_follow = a_[j, i]
                node.UBs += [-r.item() for r in cum_r[i, :]]

        return nodes

    # tree ###################################################################################
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
            if 'all' not in configs.expansion_type:
                # candidate_a = configs.beam_n  # nodes with highest prob. # child_n
                raise NotImplementedError

            # expansion #################################################
            for a in candidate_a:
                a = node.obs['op_remain'][a].view(1).to('cpu')
                env = JobShopEnv(pomo_n=1, load_envs=[node.curr_env])
                _, reward, done = env.step([a])

                node_ = Node(env, obs=None, done=done, route=node.route + [a])

                if done:
                    new_done_nodes.append(node_)
                    if node_.LB < UB:
                        UB = node_.LB
                        UB_update = True
                else:
                    total_n += 1
                    if 'bound' in configs.expansion_type and node_.LB >= UB:  # bound
                        continue
                    new_non_done_nodes.append(node_)

        if UB_update and 'bound' in configs.expansion_type:  # bound
            new_non_done_nodes = [node for node in new_non_done_nodes if node.LB < UB]

        if total_n:
            real_n = len(new_non_done_nodes)
            configs.total_extend_n += total_n
            configs.real_extend_n += real_n
            configs.bound_rates.append(1 - round(real_n / total_n, 3))

        return new_non_done_nodes, new_done_nodes, UB

    def get_best_nodes(self, nodes, UB):
        pos = len(nodes)
        for i, node in enumerate(nodes):
            if node.UB > UB:
                pos = i
                break

        return nodes[:pos], nodes[pos:]

    def get_action_beam(self, env, existing_tree=None, problem=None,
                        rules=None, models=None) -> (float, list, list, float, list):
        """
        perform for an env
        return: cum_r, avg_run_t, avg_decision_t, decision_n, trajectory
        """
        best_non_done_nodes = list()
        best_done_nodes = list()
        remain_done_nodes = list()
        remain_non_done_nodes = list()

        obs = env.get_obs()
        done = env.done()

        # initial #########################################################################
        s_t = time.time()
        if existing_tree:
            non_done_nodes, done_nodes, best_node, UB = existing_tree
            depth = configs.depth - 1
        else:
            node = Node(env, obs=obs, done=done)  # initial expansion
            if node.done:
                return 0, [], [], node

            non_done_nodes = self.initial_rollout([node], rules, models)  # simulation
            UB = non_done_nodes[0].UB
            best_node = non_done_nodes[0]
            done_nodes = list()
            depth = 0

        # loop ########################################################################################
        while non_done_nodes and depth < configs.depth:
            depth += 1

            # bounded expansion ###########################################
            non_done_nodes, new_done_nodes, UB_ = self.expansion_leaf(non_done_nodes, UB)

            if new_done_nodes:
                done_nodes += new_done_nodes
                done_nodes.sort(key=lambda x: x.UB)
                done_nodes = done_nodes[:configs.beam_n]  # too many -> cut
                if UB_ < UB:
                    UB = UB_
                    best_node = done_nodes[0]

            if not non_done_nodes:  # no more expansion -> break
                break

            # simulation ##################################################
            non_done_nodes = self.initial_rollout(non_done_nodes, rules, models)

            non_done_nodes.sort(key=lambda x: x.UB)
            UB_ = non_done_nodes[0].UB
            if UB_ < UB:
                UB = UB_
                best_node = non_done_nodes[0]

            # separation  #################################################
            remain_done_nodes.clear()
            remain_non_done_nodes.clear()

            if 'with_best' in configs.pruning_type:
                best_non_done_nodes.clear()
                best_done_nodes.clear()

                if 'terminal' in configs.pruning_type:
                    best_done_nodes, remain_done_nodes = self.get_best_nodes(done_nodes, UB)
                best_non_done_nodes, remain_non_done_nodes = self.get_best_nodes(non_done_nodes, UB)

            else:
                if 'terminal' in configs.pruning_type:
                    remain_done_nodes = done_nodes
                remain_non_done_nodes = non_done_nodes

            # pruning  ####################################################
            if configs.beam_n <= len(best_done_nodes) + len(best_non_done_nodes):  # 'with_best'
                if not best_non_done_nodes:  # configs.beam_n <= len(best_done_nodes)
                    break
                non_done_nodes = best_non_done_nodes[:configs.beam_n]

            elif configs.beam_n >= len(best_done_nodes) + len(best_non_done_nodes) + \
                    len(remain_done_nodes) + len(remain_non_done_nodes):
                non_done_nodes = best_non_done_nodes + remain_non_done_nodes

            else:
                non_done_nodes = best_non_done_nodes

                remain_nodes = remain_done_nodes + remain_non_done_nodes
                for node in remain_nodes:
                    node.update_value()
                remain_nodes.sort(key=lambda x: (x.value, x.done))  # priorities non_done nodes

                for node in remain_nodes[:configs.beam_n - len(best_non_done_nodes)]:
                    if not node.done:
                        non_done_nodes.append(node)

        run_t = round(time.time() - s_t, 4)

        return run_t, non_done_nodes, done_nodes, best_node

    def run_episode_with_beam(self, problem=None, rules=None, models=None) -> (float, float, float, float, list):
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

            start = True
            while not done:
                if start:
                    run_t, non_done_nodes, done_nodes, best_node = self.get_action_beam(env, rules=rules, models=models)
                    start = False
                else:
                    run_t, non_done_nodes, done_nodes, best_node = self.get_action_beam(
                        env, existing_tree=(non_done_nodes, done_nodes, best_node, UB), rules=rules, models=models)
                if best_node.route:
                    a = best_node.route.pop(0).view(1, -1)
                else:
                    a = best_node.best_follow[0].view(1, -1)
                    best_node.best_follow = best_node.best_follow[1:]
                UB = best_node.UB
                run_ts.append(run_t)

                # route, best_follow update ############################
                del_list = list()
                for i, node in enumerate(done_nodes):
                    if node == best_node:
                        continue
                    if node.route:
                        if a.item() != node.route.pop(0).item():
                            del_list.append(i)
                    else:
                        if a.item() != node.best_follow[0].item():
                            del_list.append(i)
                        node.best_follow = node.best_follow[1:]
                for i in reversed(del_list):
                    del done_nodes[i]

                del_list = list()
                for i, node in enumerate(non_done_nodes):
                    if node == best_node:
                        continue
                    if node.route:
                        if a.item() != node.route.pop(0).item():
                            del_list.append(i)
                    else:
                        if a.item() != node.best_follow[0].item():
                            del_list.append(i)
                        node.best_follow = node.best_follow[1:]
                for i in reversed(del_list):
                    del non_done_nodes[i]

                obs, reward, done = env.step([a])

        return -reward.item(), sum(run_ts), len(run_ts), \
               configs.total_extend_n, configs.real_extend_n, configs.bound_rates

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
            for model in models:
                model.eval()
        elif 'model' in configs.rollout_type:
            self.model_load()
            self.model.to(configs.device)
            self.model.eval()

        for (benchmark, job_n, mc_n, instances) in benchmarks:
            print(benchmark, job_n, mc_n, len(instances))

            for i in tqdm(instances):
                if configs.rollout_depth < 1e6:
                    UB, run_t, decision_n, total_n, real_n, bound_rates = self.run_episode_with_beam2((
                        benchmark, job_n, mc_n, i), rules=rules, models=models)
                else:
                    UB, run_t, decision_n, total_n, real_n, bound_rates = self.run_episode_with_beam((
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
                                 configs.q_level, configs.rollout_n, configs.rollout_type,
                                 configs.depth, configs.rollout_depth,
                                 UB, run_t, total_n, real_n, rate, mean_bound_rate,
                                 decision_n, round(run_t / decision_n, 4)])


if __name__ == '__main__':
    from utils import TA_small

    save_path = f'./../result/model_beam_test.csv'
    configs.expansion_type = 'bound_all'

    for configs.depth in [3]:

        for configs.rollout_n in [3]:
            for configs.value_type in ['best']:  # quantile_LB, best, mean, norm, beta, t, chi2
                if configs.rollout_n == 1 and configs.value_type != 'best':
                    continue

                for configs.beam_n in [3]:  # 5
                    if 'with_best' in configs.pruning_type and configs.beam_n == 1 and configs.value_type != 'best':
                        continue

                    for configs.rollout_type in ['model_rule']:  # model_rule, rule, multi_model
                        if 'multi_model' in configs.rollout_type and configs.rollout_n == 1:
                            continue

                        agent = AgentTS()
                        # agent.perform_TS_benchmarks([['HUN', 6, 4, list(range(40, 50))]], save_path=save_path)
                        agent.perform_beam_benchmarks([['HUN', 6, 6, list(range(40, 50))]], save_path=save_path)
                        # agent.perform_beam_benchmarks(TA_small, save_path=save_path)
