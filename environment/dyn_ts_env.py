import copy, torch

from utils import load_data
from collections import defaultdict
from params import configs
import torch.nn.functional as F
from environment.dyn_env import JobShopDynEnv


class JobShopTSDynEnv(JobShopDynEnv):
    def next_state(self):
        next_state = False
        configs.event = False
        while not next_state:
            job_mask = torch.where(self.job_last_step < self.job_step_n, 0, self.M)
            start_t = self.job_ready_t[self.ENV_IDX_J, self.POMO_IDX_J, self.JOB_IDX, self.job_last_step] + job_mask
            self.curr_t = torch.min(start_t, dim=2)[0]  # (env_n, pomo_n)

            if 'job_arrival' in configs.dyn_type:
                curr_t_ = self.curr_t.unsqueeze(dim=2).expand(-1, -1, self.max_job_n+1)
                job_mask = torch.where(curr_t_ >= self.job_arrival_t_, job_mask, self.M)  # reflect non-arrival
                start_t += job_mask

            # update started jobs #################################################################
            if 'conflict' in configs.action_type and configs.dyn_reserve_reset:
                for i in range(self.env_n):
                    for j in range(self.pomo_n):
                        if self.reserved[i, j]:
                            curr_t_ = self.curr_t[i, j]
                            del_list = list()
                            for pos, info in enumerate(self.reserved[i, j]):
                                if curr_t_ > info[0]:
                                    del_list.append(i)

                            for pos in reversed(del_list):
                                del self.reserved[i, j][pos]

            # dynamic event #########################################################################
            event = False
            for i in range(self.env_n):
                for j in range(self.pomo_n):
                    while True:
                        if not self.events[i, j]:
                            break
                        if self.events[i, j][0][0] > self.curr_t[i, j]:
                            break

                        event = True
                        s_t, event_type, target, etc = self.events[i, j].pop(0)  # (t, 'job_arrival', j, None)

                        # dynamic reservation reset ###############################
                        if 'conflict' in configs.action_type and configs.dyn_reserve_reset:
                            # cancel reservations ##########################
                            # mc_update_t = defaultdict(lambda: float('inf'))
                            if self.reserved[i, j]:
                                for _, (job, _, mc), (mc_last_job, mc_last_step), ready_t_pred, e_info \
                                        in reversed(self.reserved[i, j]):
                                    self.job_last_step[i, j, job] -= 1
                                    self.mc_last_job[i, j, mc] = mc_last_job
                                    self.mc_last_job_step[i, j, mc] = mc_last_step

                                    # update precedence_t ###################
                                    self.job_ready_t_precedence[i, j, job] = ready_t_pred

                                    # update t_mc_gap #######################
                                    mc_t = self.job_done_t[i, j, self.mc_last_job[i, j], self.mc_last_job_step[i, j]]
                                    job_mc_t = mc_t[self.job_mcs[i, j]]

                                    last_step = self.job_last_step[i, j].view(self.max_job_n + 1, -1).expand(
                                        -1, self.max_mc_n + 1)
                                    job_remain = torch.where(last_step <= torch.arange(self.max_mc_n + 1), 1, 0)
                                    job_remain[-1, :] = 0

                                    gap = (job_mc_t - self.job_ready_t_precedence[i, j]).mul(job_remain)
                                    gap = torch.where(gap > 0, gap, 0)
                                    gap_max = gap.max(dim=1)[0].view(-1, 1).expand(-1, self.max_mc_n+1)
                                    gap_arg_max = gap.argmax(dim=1).view(-1, 1).expand(-1, self.max_mc_n+1)

                                    tails = torch.where(self.JOB_STEP_IDX[i, j] > gap_arg_max, 1, 0).mul(job_remain)
                                    gap_ = torch.where(self.JOB_STEP_IDX[i, j] <= gap_arg_max, gap, 0)
                                    self.job_ready_t_mc_gap[i, j] = gap_ + tails * gap_max

                                    # cancel e_disj #########################
                                    if 'rule' not in configs.agent_type:
                                        prev_op, e_disj_list = e_info
                                        self.e_disj[i, j] += e_disj_list
                                        self.mc_ops[i, j][mc].append(self.mc_prev_op[i, j][mc])
                                        if prev_op is None:
                                            del self.mc_prev_op[i, j][mc]
                                        else:
                                            self.mc_prev_op[i, j][mc] = prev_op

                                # last update #######################################
                                self.job_ready_t = self.job_ready_t_precedence + self.job_ready_t_mc_gap
                                self.job_done_t = self.job_ready_t + self.job_durations
                                del self.reserved[i, j]

                        # new job arrival -> dynamic add e_disj ###################
                        if 'job_arrival' in event_type:
                            # target: arrival job idx
                            if 'rule' not in configs.agent_type:
                                for k in range(self.job_step_n[i, j, target]):
                                    op = self.op_map[i, target, k].item()
                                    m = self.job_mcs[i, j][target][k].item()

                                    for op2 in self.mc_ops[i, j][m]:
                                        self.e_disj[i, j].append((op, op2))
                                        self.e_disj[i, j].append((op2, op))
                                        if m in self.mc_prev_op[i, j].keys():
                                            self.e_disj[i, j].append((self.mc_prev_op[i, j][m], op))
                                    self.mc_ops[i, j][m].append(op)

                        # machine breakdown #######################################
                        if 'mc_breakdown' in event_type:
                            # target: target mc idx, etc: repairing timing
                            if 'known' in configs.dyn_type:  # add only repair time
                                target_mc = F.one_hot(torch.tensor(target), num_classes=self.max_mc_n)
                                # self.mc_break[i, j, target] = s_t
                                self.mc_repair[i, j, target] = etc

                                # being process job update ##########################
                                mc_last_job = self.mc_last_job[i, j, target]
                                if mc_last_job < self.max_job_n:
                                    mc_last_step = self.mc_last_job_step[i, j, target]
                                    if self.job_done_t[i, j, mc_last_job, mc_last_step] > s_t:
                                        self.job_durations[i, j, mc_last_job, mc_last_step] += (etc - s_t)
                                        self.job_ready_t_precedence[i, j, mc_last_job, mc_last_step+1:] += (etc - s_t)
                                self.job_done_t[i, j] = self.job_ready_t[i, j] + self.job_durations[i, j]

                                # mc t #################################
                                mc_t = self.job_done_t[i, j][self.mc_last_job[i, j], self.mc_last_job_step[i, j]]
                                mc_repair = torch.where(self.mc_repair[i, j] <= self.curr_t[i, j], self.mc_repair[i, j], 0)
                                mc_t = torch.where(mc_repair > mc_t, mc_repair, mc_t)
                                # if mc_t[target] > s_t:
                                #     mc_t[target] += (etc - s_t)
                                # else:
                                #     mc_t[target] = etc
                                # mc_t = torch.where(self.mc_repair[i, j] > mc_t, self.mc_repair[i, j], mc_t)
                                job_mc_t = mc_t[self.job_mcs[i, j]]

                                # remain operations #################################
                                last_step = self.job_last_step[i, j].view(self.max_job_n + 1,
                                                                          -1).expand(-1, self.max_mc_n + 1)
                                job_remain = torch.where(last_step <= self.JOB_STEP_IDX[i, j], 1, 0)
                                job_remain[-1, :] = 0
                                job_remain[:, -1] = 0

                                # same step #########################################
                                same_mc_step = target_mc[self.job_mcs[i, j]].mul(job_remain).mul(self.JOB_OP[i, j])

                                step_pos = same_mc_step.argmax(dim=1).view(self.max_job_n + 1,
                                                                           -1).expand(-1, self.max_mc_n + 1)
                                step_sum = same_mc_step.sum(dim=1)
                                step_sum = torch.where(step_sum > 0, 1, 0).view(self.max_job_n + 1,
                                                                                -1).expand(-1, self.max_mc_n + 1)
                                same_mc_step_follow = torch.where(step_pos <= self.JOB_STEP_IDX[i, j],
                                                                  1, 0).mul(step_sum)

                                mc_update = (job_mc_t - self.job_ready_t_precedence[i, j]).mul(
                                    same_mc_step).sum(dim=1).view(self.max_job_n + 1, -1).expand(
                                    -1, self.max_mc_n + 1).mul(same_mc_step_follow)

                                self.job_ready_t_mc_gap[i, j] = torch.where(self.job_ready_t_mc_gap[i, j] > mc_update,
                                                                            self.job_ready_t_mc_gap[i, j], mc_update)

                                # ready_t update ####################################
                                self.job_ready_t[i, j] = self.job_ready_t_precedence[i, j] + self.job_ready_t_mc_gap[i, j]
                                self.job_done_t[i, j] = self.job_ready_t[i, j] + self.job_durations[i, j]

            if event:
                configs.event = True
                continue

            # candidate job ########################################################################
            job_mcs = self.job_mcs[self.ENV_IDX_J, self.POMO_IDX_J, self.JOB_IDX, self.job_last_step][:, :, :-1]

            if 'conflict' in configs.action_type:
                done_t = self.job_done_t[self.ENV_IDX_J, self.POMO_IDX_J, self.JOB_IDX, self.job_last_step] + job_mask
                c_min = torch.min(done_t, dim=2)[0].unsqueeze(dim=2).expand(-1, -1, self.max_job_n+1)  # (env_n, pomo_n)
                min_jobs = torch.where(done_t == c_min, 1, 0)

                min_mcs = torch.zeros(self.env_n, self.pomo_n, self.max_mc_n, dtype=torch.long)
                min_mcs.scatter_add_(2, index=job_mcs, src=min_jobs)
                min_mcs = torch.where(min_mcs > 0, 1, 0)
                cand_jobs = torch.where(start_t < c_min, 1, 0)  # (env_n, pomo_n, max_job_n+1)
            elif 'buffer' in configs.action_type:
                curr_t = self.curr_t.unsqueeze(dim=2).expand(-1, -1, self.max_job_n+1)
                cand_jobs = torch.where(start_t == curr_t, 1, 0)  # (env_n, pomo_n, max_job_n+1)
            else:
                cand_jobs = 0

            # count mc actions #####################################################################
            mc_count = torch.zeros(self.env_n, self.pomo_n, self.max_mc_n, dtype=torch.long)
            mc_count.scatter_add_(2, index=job_mcs, src=cand_jobs)  # (env_n, pomo_n, max_mc_n)
            if 'conflict' in configs.action_type:
                mc_count = mc_count.mul(min_mcs)

            # automatic assign job ################################################################
            if len(sum(torch.where(mc_count == 1))) > 0:
                if 'conflict' in configs.action_type:
                    cand_jobs_ = torch.where(start_t < c_min, self.JOB_IDX, 0)  # (env_n, pomo_n, max_job_n+1)
                elif 'buffer' in configs.action_type:
                    cand_jobs_ = torch.where(start_t == curr_t, self.JOB_IDX, 0)  # (env_n, pomo_n, max_job_n+1)
                else:
                    cand_jobs_ = 0

                mc_job_sum = torch.zeros(self.env_n, self.pomo_n, self.max_mc_n, dtype=torch.long)
                mc_job_sum.scatter_add_(2, index=job_mcs, src=cand_jobs_)  # (env_n, pomo_n, max_mc_n)

                # auto assign ###############################
                self.target_mc = torch.where(mc_count == 1, 1, 0)  # (env_n, pomo_n, max_mc_n)

                while torch.any(self.target_mc.sum(dim=2) > 1):
                    index = self.MC_PRIOR.mul(self.target_mc)
                    index_max = index.max(dim=2)[0].view(self.env_n, self.pomo_n, 1).expand(-1, -1, self.max_mc_n)
                    target_mc = copy.deepcopy(self.target_mc)
                    self.target_mc = torch.where((index == index_max) & (index > 0), 1, 0)  # (env_n, pomo_n, max_mc_n)

                    mc_auto_job = torch.where(self.target_mc == 1, mc_job_sum, self.max_job_n)
                    self.assign(mc_auto_job)

                    self.target_mc = target_mc - self.target_mc

                mc_auto_job = torch.where(self.target_mc == 1, mc_job_sum, self.max_job_n)  # null job: dummy job
                self.assign(mc_auto_job)

            # next state ##########################################################################
            else:
                self.target_mc = torch.where(mc_count > 0, 1, 0)  # (env_n, pomo_n, max_mc_n)
                if torch.any(self.target_mc.sum(dim=2) > 1):
                    index = self.MC_PRIOR.mul(self.target_mc)
                    index_max = index.max(dim=2)[0].view(self.env_n, self.pomo_n, 1).expand(-1, -1, self.max_mc_n)
                    self.target_mc = torch.where((index == index_max) & (index > 0), 1, 0)  # (env_n, pomo_n, max_mc_n)

                job_mc = F.one_hot(self.job_last_step, num_classes=self.max_mc_n+1).mul(self.job_mcs).sum(dim=3)
                cand_jobs = self.target_mc[self.ENV_IDX_J, self.POMO_IDX_J, job_mc].mul(cand_jobs)
                cand_jobs_ = cand_jobs[:, :, :-1].unsqueeze(dim=3).expand(-1, -1, -1, self.max_mc_n)

                job_last_step_one_hot = F.one_hot(self.job_last_step, num_classes=self.max_mc_n+1)[:, :, :-1, :-1]
                self.op_mask = job_last_step_one_hot.mul(cand_jobs_).view(self.env_n, self.pomo_n, -1)

                next_state = True

    def done(self):
        finished = (self.job_last_step == self.job_step_n).all(dim=2)  # (env_n, pomo_n)
        return finished.all().item()


if __name__ == "__main__":
    from utils import rules_5, all_benchmarks, REAL_D, FLOW
    import csv, time

    ###########################################################################################################
    from utils import all_dyn_benchmarks, all_benchmarks
    import os

    configs.action_type = 'buffer'
    configs.agent_type = 'rule'
    rules = rules_5
    # rules = ['LTT']

    configs.dyn_type = 'job_arrival'
    parameters = [100, 200, 300, 400]
    configs.init_batch = 2

    # configs.dyn_type = 'mc_breakdown_known'
    # parameters = [100, 200, 300, 400]

    # configs.dyn_type = 'prt_stochastic_known'
    # parameters = [0.1, 0.2, 0.3, 0.4]

    #########################################################
    save_folder = f'./../result/{configs.dyn_type}/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = save_folder + f'result_rule.csv'

    for configs.dyn_reserve_reset in [True]:
        for configs.parameter in parameters:
            for (benchmark, job_n, mc_n, instances) in all_dyn_benchmarks:  # [['HUN', 6, 4, [0]]]  all_benchmarks
                print(benchmark, job_n, mc_n, instances)
                for i in instances:
                    envs_info = [(benchmark, job_n, mc_n, i)]
                    env = JobShopTSDynEnv(envs_info, pomo_n=len(rules))

                    ##############################################
                    s_t = time.time()
                    obs, reward, done = env.reset()

                    while not done:
                        a = env.get_action_rule(obs, rules)
                        obs, reward, done = env.step(a)
                    # env.show_gantt_plotly(0, 0)
                    run_t = round(time.time() - s_t, 4)
                    # print(i, run_t)

                    ############################################
                    with open(save_path, 'a', newline='') as f:
                        wr = csv.writer(f)
                        for j in range(len(rules)):
                            wr.writerow([benchmark, job_n, mc_n, i,
                                         configs.agent_type, configs.action_type, rules[j],
                                         -reward[0, j].item(), run_t,
                                         configs.dyn_type, configs.parameter, configs.dyn_reserve_reset])