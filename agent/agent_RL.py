import torch

from agent_TS import AgentTS, Node
from environment.env import JobShopEnv
from params import configs
import time, os, pickle, random
from tqdm import tqdm
from torch.nn.functional import softmax
from layer import get_mlp
from utils import get_RL_x_dim, get_envs, get_new_envs, get_rules, get_model_i_list

from torch.nn.utils import clip_grad_norm_
from torch.distributions import Categorical


class NN(torch.nn.Module):
    def __init__(self, in_dim, out_dim=1, agent_type='policy'):
        super().__init__()
        self.agent_type = agent_type
        self.layer = get_mlp(in_dim, out_dim).to(configs.device)

    def forward(self, obs):
        z = self.layer(obs)

        return self.final_policy(z)

    def final_policy(self, z):
        """
        return a policy or a value
        """
        if 'policy' in self.agent_type:
            if configs.softmax_tau != 1:
                z /= configs.softmax_tau
            return softmax(z, dim=0)
        else:
            return z


class AgentRL(AgentTS):  # for pruning
    def __init__(self, model_i=0):
        super().__init__(model_i)

        x_dim = get_RL_x_dim()
        self.model_pruning = NN(x_dim, 1, 'policy')

        self.optimizer_pruning = self.get_optimizer_pruning()
        self.scheduler_pruning = self.get_scheduler_pruning()

    # model ###########################################################################################
    def get_optimizer_pruning(self, model=None, lr=None):
        if not model:
            model = self.model_pruning
        if not lr:
            lr = configs.lr

        if configs.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=configs.L2_norm_w)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=configs.L2_norm_w)
        return optimizer

    def get_scheduler_pruning(self, optimizer=None, lr=None, scheduler_type=None):
        if not optimizer:
            optimizer = self.optimizer_pruning
        if not lr:
            lr = configs.lr
        if scheduler_type is None:
            scheduler_type = configs.scheduler_type

        if 'cyclic' in scheduler_type:
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                          base_lr=configs.min_lr, max_lr=lr,
                                                          step_size_up=1, step_size_down=49,
                                                          mode='triangular2', cycle_momentum=False)
        elif 'step' in scheduler_type:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=[250, 500, 750], gamma=0.5)
        else:
            scheduler = None

        return scheduler

    # torch model save, load ##########################################################################
    def model_load_pruning(self) -> None:
        """ torch model parameter load
        """
        self.model_pruning.load_state_dict(torch.load(f'{self.model_path_pruning()}.pt'))

    def model_save_pruning(self) -> None:
        """
        save torch model parameter
        """
        torch.save(self.model_pruning.state_dict(), f'{self.model_path_pruning()}.pt')

    def model_path_pruning(self) -> str:
        save_path = f'./../model_selector/{configs.action_type}'

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        model_path = f'{save_path}/{self.get_model_name_pruning()}'

        return model_path

    def get_model_name_pruning(self) -> str:
        model_name = f'model_{configs.rollout_type}_{configs.rollout_n}_{configs.RL_state}_{self.model_i}'
        return model_name

    def get_current_lr_pruning(self) -> float:
        return round(self.optimizer_pruning.param_groups[0]['lr'], 7)

    # etc function for learning #######################################################################
    def save_training_process_pruning(self, valids: list, losses: list) -> None:
        """
        save loss and training validation performance
        """
        result_path = self.model_path_pruning() + '_result.pkl'
        result = [valids, losses]
        with open(result_path, 'wb') as f:
            pickle.dump(result, f)

    def get_training_process_pruning(self) -> (list, list):
        """
        load loss and training validation performance
        """
        result_path = self.model_path_pruning() + '_result.pkl'
        with open(result_path, 'rb') as f:
            valids, losses = pickle.load(f)

        return valids, losses

    def training_process_fig_pruning(self, valids: list, losses: list, save_TF: bool=False) -> None:
        from matplotlib import pyplot as plt

        plt.clf()

        fig, ax1 = plt.subplots()
        ax1.set_xlabel('repeat_n')
        ax1.set_ylabel('loss')
        line1 = ax1.plot(losses, color='green', label='train loss')
        # ax1.set_ylim([200, 350])

        ax2 = ax1.twinx()
        ax2.set_ylabel('makespan')
        line2 = ax2.plot(valids, color='deeppink', label='valid makespan')
        # ax2.set_ylim([-1450, -1320])

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')
        plt.title(self.get_model_name())

        if save_TF:
            result_path = self.model_path_pruning() + '_result.png'
            plt.savefig(result_path)
        else:
            plt.show()

    # state, value #######################################################################
    def get_pruning_obs(self, candidate_nodes) -> torch.tensor:
        state = list()

        if configs.RL_state == 'pos':
            for node in candidate_nodes:
                if node.done:
                    state.append([0 for _ in range(configs.rollout_n)])
                else:
                    state.append([(UB - node.LB)/node.LB for UB in node.UBs])

        elif configs.RL_state == 'sort':
            for node in candidate_nodes:
                if node.done:
                    state.append([0 for _ in range(2 * configs.rollout_n - 1)])
                else:
                    node.UBs.sort()
                    state.append([(UB - node.LB)/node.LB for UB in node.UBs] +
                                 [(UB - node.UBs[0])/node.LB for UB in node.UBs[1:]])

        elif configs.RL_state == 'pos_sort':
            for node in candidate_nodes:
                if node.done:
                    state.append([0 for _ in range(3 * configs.rollout_n - 1)])
                else:
                    sort_UB = sorted(node.UBs)
                    state.append([(UB - node.LB)/node.LB for UB in node.UBs] +
                                 [(UB - node.LB)/node.LB for UB in sort_UB] +
                                 [(UB - sort_UB[0])/node.LB for UB in sort_UB[1:]])

        else:
            raise NotImplementedError

        return torch.tensor(state).to(configs.device, dtype=torch.float32)

    def model_value(self, pruning_obs):
        return self.model_pruning(pruning_obs)

    # run #######################################################################
    def get_action_beam(self, env, existing_tree=None, rules=None, models=None):
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

                # selection model ######################################
                if len(remain_nodes) > 2:
                    pruning_obs = self.get_pruning_obs(remain_nodes)
                    prob = self.model_value(pruning_obs)

                    for i, node in enumerate(remain_nodes):
                        node.value = round(prob[i].item(), 5)
                    remain_nodes.sort(key=lambda x: (-x.value, x.done))  # priorities non_done nodes
                    for node in remain_nodes[:configs.beam_n - len(best_non_done_nodes)]:
                        if not node.done:
                            non_done_nodes.append(node)
                else:
                    non_done_nodes.append(remain_nodes[0])

                ########################################################

        run_t = round(time.time() - s_t, 4)

        return run_t, non_done_nodes, done_nodes, best_node

    def get_action_beam_train(self, env, existing_tree=None, rules=None, models=None, episodic=True):
        """
        perform for an env
        return: cum_r, avg_run_t, avg_decision_t, decision_n, trajectory
        """
        best_non_done_nodes = list()
        best_done_nodes = list()
        remain_done_nodes = list()
        remain_non_done_nodes = list()

        log_prob = list()
        rewards = list()

        obs = env.get_obs()
        done = env.done()

        # initial #########################################################################
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
        action_TF = False
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
                if not episodic and action_TF:
                    if new_done_nodes:
                        new_done_nodes.sort(key=lambda x: x.UB)
                        rewards.append((UB - new_done_nodes[0].UB) / new_done_nodes[0].LB)
                    else:
                        rewards.append(0)
                break

            # simulation ##################################################
            non_done_nodes = self.initial_rollout(non_done_nodes, rules, models)

            non_done_nodes.sort(key=lambda x: x.UB)
            UB_ = non_done_nodes[0].UB
            if not episodic and action_TF:
                rewards.append((UB - UB_)/non_done_nodes[0].LB)
                action_TF = False

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

                # selection model ######################################
                if len(remain_nodes) > 2:
                    pruning_obs = self.get_pruning_obs(remain_nodes)
                    prob = self.model_value(pruning_obs)

                    action_TF = True
                    policy = Categorical(prob.view(1, -1))
                    a_tensor = policy.sample()
                    log_prob += torch.log(prob[a_tensor])

                    non_done_nodes.append(remain_nodes[a_tensor.item()])
                else:
                    non_done_nodes.append(remain_nodes[0])

                ########################################################

        return best_node, log_prob, rewards

    def perform_envs_RL(self, envs, rules=None, models=None) -> (float, float):
        """
        perform for an env
        return: cum_r, avg_run_t, avg_decision_t, decision_n, trajectory
        """
        rewards = list()
        run_ts = list()
        self.model_pruning.eval()
        for env in envs:
            _, _, _ = env.reset()

            UB, run_t, _, _, _, _ = self.run_episode_with_beam(env, rules=rules, models=models)
            rewards.append(-UB)
            run_ts.append(run_t)

        return round(sum(rewards) / len(rewards), 3), round(sum(run_ts) / len(run_ts), 4)

    def perform_beam_model_benchmarks(self, benchmarks, save_path='') -> None:
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

        self.model_load_pruning()
        self.model_pruning.to(configs.device)

        for (benchmark, job_n, mc_n, instances) in benchmarks:
            print(benchmark, job_n, mc_n, len(instances))

            for i in tqdm(instances):
                env = JobShopEnv([(benchmark, job_n, mc_n, i)], pomo_n=1)
                if configs.rollout_depth < 1e6:
                    raise NotImplementedError

                self.model_pruning.eval()
                UB, run_t, decision_n, total_n, real_n, bound_rates = self.run_episode_with_beam(
                    env, rules=rules, models=models)

                with open(save_path, 'a', newline='') as f:
                    wr = csv.writer(f)
                    rate, mean_bound_rate, mean_run_t = 0, 0, 0
                    if total_n:
                        rate = 1 - round(real_n / total_n, 3)
                    if bound_rates:
                        mean_bound_rate = round(sum(bound_rates) / len(bound_rates), 3)
                    if decision_n > 0:
                        mean_run_t = round(run_t / decision_n, 4)
                    wr.writerow([benchmark, job_n, mc_n, i, 'beam', configs.action_type,
                                 configs.beam_n, configs.expansion_type, configs.value_type, configs.pruning_type,
                                 configs.q_level, configs.rollout_n, configs.rollout_type,
                                 configs.depth, configs.rollout_depth,
                                 UB, run_t, total_n, real_n, rate, mean_bound_rate,
                                 decision_n, mean_run_t])

    # learning #####################################################################################
    def update_RL(self, reward, log_prob):
        """
        update weight of model
        """
        self.model_pruning.train()  # (dropout=True)
        self.optimizer_pruning.zero_grad()

        loss = -reward.mul(log_prob).mean()  # minus sign: to increase reward
        loss.backward()

        clip_grad_norm_(self.model_pruning.parameters(), configs.max_grad_norm)
        self.optimizer_pruning.step()

        return loss

    def run_episode_once_RL_train(self, env, rules=None, models=None, episodic=True):
        """
        perform for an env
        return: cum_r, avg_run_t, avg_decision_t, decision_n, trajectory
        """
        # initial environment #########################################################################
        self.model_pruning.train()
        best_node, log_prob, rewards = self.get_action_beam_train(env, rules=rules, models=models, episodic=episodic)
        if episodic:
            rewards = [-best_node.UB]

        return log_prob, rewards

    def run_episodes_RL_train(self, envs, rules=None, models=None, episodic=True):
        """
        perform for an env
        return: cum_r, avg_run_t, avg_decision_t, decision_n, trajectory
        """

        rewards = list()
        log_probs = list()

        for env in envs:
            _, _, _ = env.reset()
            rs = list()
            cum_log_ps = list()

            if episodic:
                for _ in range(configs.pomo_n):
                    log_prob, r = self.run_episode_once_RL_train(env=env, rules=rules, models=models)
                    cum_log_ps.append(torch.concat(log_prob).sum().view(1))  # episodic
                    rs.append(r[0])
                mean_r = sum(rs) / configs.pomo_n

                for i in range(configs.pomo_n):
                    rs[i] = round((rs[i] - mean_r) / mean_r, 5)

                log_probs += cum_log_ps
                rewards += rs

            # else:
                #
                #
                # reward_ = list()
                #     cum_r = 0
                #     for r in reversed(reward):
                #         cum_r += r
                #         reward_.insert(0, cum_r)
                #         cum_r *= configs.gamma

                # log_probs += log_prob
                # rewards += reward_

        return rewards, log_probs

    def learning_opt_RL(self, train_info, valid_problem_set, train_problem_info) -> None:
        """
        learn optimal policies
        """
        envs_train = get_envs(train_info, pomo_n=1)
        envs_valid = get_envs(valid_problem_set, pomo_n=1)
        losses = list()
        valids = list()

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

        # init valid ###############################################################
        reward, mean_run_t = self.perform_envs_RL(envs_valid, rules=rules, models=models)
        valids.append(reward)
        print(f'initial valid mean_cum_r: {reward} \tmean_run_t: {mean_run_t} sec')

        # repeat ##################################################################
        self.model_save_pruning()
        for epoch in tqdm(range(configs.RL_max_ep_n)):
            # new envs ##########################################
            # job_n, mc_n = random.choice(train_problem_info)
            # envs_train = get_new_envs([(job_n, mc_n, configs.train_env_n)])

            # trajectories #####################################
            rewards, log_probs = self.run_episodes_RL_train(envs_train, rules=rules, models=models, episodic=True)

            rewards = torch.tensor(rewards).to(configs.device)
            log_probs = torch.concat(log_probs).to(configs.device)

            _ = self.update_RL(rewards, log_probs)

            self.scheduler_pruning.step()
            if configs.device != 'cpu':
                torch.cuda.empty_cache()  # clear memory

            # valid ######################
            reward, mean_run_t = self.perform_envs_RL(envs_valid, rules=rules, models=models)
            valids.append(reward)

            if min(valids) >= valids[-1]:
                self.model_save_pruning()
                print(f'-- epoch: {epoch} '
                      f'\tvalid mean_cum_r: {reward} \tmean_run_t: {mean_run_t} sec '
                      f'\tbest valid mean_cum_r: {round(min(valids), 4)} \tlearning rate: {self.get_current_lr()}')

            if (epoch + 1) % 100 == 0:
                print(f'-- epoch: {epoch} '
                      f'\tvalid mean_cum_r: {reward} \tmean_run_t: {mean_run_t} sec '
                      f'\tbest valid mean_cum_r: {round(min(valids), 4)} \tlearning rate: {self.get_current_lr()}')

                self.save_training_process(valids, losses)
                # self.training_process_fig(valids, losses, save_TF=False)

        # save result #############################################################
        self.save_training_process(valids, losses)
        self.training_process_fig(valids, losses, save_TF=True)
        # from environment import visualize
        # visualize.get_gantt_plotly(envs_valid[0])

    # def learning_opt_RL(self, train_info, valid_problem_set, train_problem_info) -> None:
    #     """
    #     learn optimal policies
    #     """
    #     envs_test
    #
    #     envs_valid = get_envs(valid_problem_set, pomo_n=1)
    #     losses = list()
    #     valids = list()
    #
    #     rules = get_rules(configs.rollout_n)
    #     models = None
    #     if 'multi_model' in configs.rollout_type:
    #         model_i_list = get_model_i_list(configs.rollout_n)
    #         models = self.models_load(model_i_list)
    #         for model in models:
    #             model.eval()
    #     elif 'model' in configs.rollout_type:
    #         self.model_load()
    #         self.model.to(configs.device)
    #         self.model.eval()
    #
    #     # init valid ###############################################################
    #     reward, mean_run_t = self.perform_beam_model_envs(envs=envs_valid, rules=rules, models=models)
    #     valids.append(reward)
    #     print(f'initial valid mean_cum_r: {reward} \tmean_run_t: {mean_run_t} sec')
    #
    #     # repeat ##################################################################
    #     self.model_save_pruning()
    #     for epoch in tqdm(range(configs.RL_max_ep_n)):
    #         # new envs ##########################################
    #         job_n, mc_n = random.choice(train_problem_info)
    #         train_envs = get_new_envs([(job_n, mc_n, configs.train_env_n)])
    #
    #         # trajectories #####################################
    #         # rewards, log_probs = self.run_episodes_RL_train(train_envs, rules=rules, models=models)
    #         rewards, log_probs = self.run_episodes_RL_train2(train_envs, rules=rules, models=models)
    #
    #         rewards = torch.tensor(rewards).to(configs.device)
    #         log_probs = torch.concat(log_probs).to(configs.device)
    #
    #         # policy_loss = log_probs * rewards
    #         # policy_loss = - policy_loss.sum(axis=0)
    #
    #         # mean_r = rewards.to(torch.float32).mean(dim=1)[:, None].expand(-1, configs.policy_n)
    #         # # reward_ = -(reward - mean_r) / mean_r
    #         # # reward_ = reward - mean_r
    #
    #         _ = self.update_RL(rewards, log_probs)
    #
    #         # self.update_RL(rewards[13:14], log_probs[13:14])
    #         # configs.probs[0], configs.obss[0], configs.actions[0], self.model_value(configs.obss[0])
    #
    #         if self.scheduler_pruning:
    #             self.scheduler_pruning.step()
    #         if configs.device != 'cpu':
    #             torch.cuda.empty_cache()  # clear memory
    #
    #         # valid ######################
    #         reward, mean_run_t = self.perform_beam_model_envs(envs=envs_valid, rules=rules, models=models)
    #         valids.append(reward)
    #
    #         if min(valids) >= valids[-1]:
    #             self.model_save_pruning()
    #             print(f'-- epoch: {epoch} '
    #                   f'\tvalid mean_cum_r: {reward} \tmean_run_t: {mean_run_t} sec '
    #                   f'\tbest valid mean_cum_r: {round(min(valids), 4)} \tlearning rate: {self.get_current_lr()}')
    #
    #         if (epoch + 1) % 100 == 0:
    #             print(f'-- epoch: {epoch} '
    #                   f'\tvalid mean_cum_r: {reward} \tmean_run_t: {mean_run_t} sec '
    #                   f'\tbest valid mean_cum_r: {round(min(valids), 4)} \tlearning rate: {self.get_current_lr()}')
    #
    #             self.save_training_process(valids, losses)
    #             # self.training_process_fig(valids, losses, save_TF=False)
    #
    #     # save result #############################################################
    #     self.save_training_process(valids, losses)
    #     self.training_process_fig(valids, losses, save_TF=True)
    #     # from environment import visualize
    #     # visualize.get_gantt_plotly(envs_valid[0])


if __name__ == '__main__':
    # from utils import TA_small

    save_path = f'./../result/model_beam_RL_0.csv'

    configs.beam_n = 3
    configs.rollout_n = 3
    configs.expansion_type = 'bound_all'
    configs.pruning_type = 'terminal_with_best'  # with_best

    configs.depth = 1e6
    configs.rollout_depth = 1e6

    configs.pomo_n = 5
    problem_info = [(10, 6), (10, 8), (8, 6), (8, 8)]  # , (8, 6), (8, 8)
    train_info = [['HUN', 8, 6, list(range(46, 47))]]  # 46

    for configs.RL_state in ['pos_sort', ]:  # 'pos', 'sort',
        if configs.rollout_n == 1 and configs.value_type != 'best':
            continue

        for configs.rollout_type in ['model_rule']:  # model_rule, rule, multi_model
            if 'multi_model' in configs.rollout_type and configs.rollout_n == 1:
                continue

            for i in [0]:  # , 1, 2, 3, 4

                agent = AgentRL(model_i=i)
                # valid_problem_set = [('HUN', 6, 4, list(range(40, 50)))]
                valid_problem_set = [('HUN', 8, 6, list(range(46, 47)))]

                agent.learning_opt_RL(train_info, valid_problem_set, problem_info)
                # agent.perform_beam_model_benchmarks(valid_problem_set, save_path=save_path)

# 6x4 40-49
# 365, 335, 404, 318, 345, 414, 376, 382, 433, 440 -> 381.2

# 8x6 40-49
# 573, 575, 532, 555, 643, 502, 706, 518, 649, 536 -> 578.9