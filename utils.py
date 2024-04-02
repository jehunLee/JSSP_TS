import os, csv
from params import configs
import pickle


# load data ##############################################################################################
def load_data(benchmark: str, job_n: int, mc_n: int, instance_i: int) -> (list, list):
    problem = f'{benchmark}{job_n}x{mc_n}'
    folder_path = f'./../benchmark/{benchmark}/{problem}'
    if not os.path.isdir(folder_path):
        folder_path = f'./benchmark/{benchmark}/{problem}'

    file_path = f'{folder_path}/instance_{instance_i}'

    job_prts = list()
    job_mcs = list()

    with open(file_path, 'r') as f:
        rdr = csv.reader(f)
        for i, line in enumerate(rdr):
            if i == 0:
                continue
            line = [int(i) for i in line]
            job_mcs.append(line[0::2])
            job_prts.append(line[1::2])

    return job_mcs, job_prts


# load result data #########################################################################################
import pandas as pd


def get_opt_makespan_once(benchmark: str, job_n: int, mc_n: int, instance_i: int) -> int:
    file_path = "../result/basic/bench_opt.csv"
    if not os.path.exists(file_path):
        file_path = "./result/basic/bench_opt.csv"

    opt_data = pd.read_csv(file_path)
    opt_data.columns = ['benchmark', 'job_n', 'mc_n', 'instance_i', 'makespan', 'convergence', 'UB', 'name', 'set']
    opt_data['UB'] = opt_data['UB'].astype('int')

    return opt_data.loc[(opt_data['benchmark'] == benchmark) & (opt_data['job_n'] == job_n)
                        & (opt_data['mc_n'] == mc_n) & (opt_data['instance_i'] == instance_i)].iloc[0]['UB']


def get_load_result_data(test_csv_file_name='./basic/bench_ours.csv') -> pd.DataFrame:
    # opt_module result data #########################################################
    result_opt = pd.read_csv("./basic/bench_opt.csv")
    result_opt.columns = ['benchmark', 'job_n', 'mc_n', 'instance_i',
                          'opt_makespan', 'convergence', 'UB', 'name', 'set']
    result_opt['problem'] = result_opt['benchmark'] + ' ' + result_opt['job_n'].astype('string') + 'x' \
                            + result_opt['mc_n'].astype('string')
    result_opt = result_opt.drop(['opt_makespan', 'convergence', 'set'], axis=1)

    # baseline data ####################################################################
    result_ref = pd.read_csv("./basic/bench_ref.csv")
    result_ref.columns = ['benchmark', 'job_n', 'mc_n', 'instance_i', 'name',
                          'makespan', 'method']
    result_ref['problem'] = result_ref['benchmark'] + ' ' + result_ref['job_n'].astype('string') + 'x' \
                            + result_ref['mc_n'].astype('string')
    result_ref = result_ref.drop(['benchmark', 'job_n', 'mc_n', 'name'], axis=1)

    # cp data #######################################################################
    result_cp = pd.read_csv("./basic/bench_cp.csv")
    result_cp.columns = ['time_limit', 'benchmark', 'job_n', 'mc_n', 'instance_i',
                         'makespan', 'time', 'convergence']
    result_cp['problem'] = result_cp['benchmark'] + ' ' + result_cp['job_n'].astype('string') + 'x' \
                           + result_cp['mc_n'].astype('string')
    result_cp['method'] = 'CP (' + result_cp['time_limit'].astype('string') + ' sec)'
    result_cp = result_cp.drop(['time_limit', 'benchmark', 'job_n', 'mc_n', 'time', 'convergence'], axis=1)

    # our data #####################################################################
    result_ours = pd.read_csv("./basic/bench_ours.csv")
    result_ours.columns = ['benchmark', 'job_n', 'mc_n', 'instance_i', 'agent_type', 'action_type', 'state_type',
                           'method', 'makespan', 'time', 'mean_decision_t', 'decision_n']
    result_ours['problem'] = result_ours['benchmark'] + ' ' + result_ours['job_n'].astype('string') + 'x' \
                             + result_ours['mc_n'].astype('string')
    result_ours = result_ours.drop(['benchmark', 'job_n', 'mc_n', 'agent_type', 'action_type', 'state_type',
                                    'time', 'mean_decision_t', 'decision_n'], axis=1)

    # rule data #####################################################################
    result_rule = pd.read_csv("./basic/bench_rule.csv")
    result_rule.columns = ['benchmark', 'job_n', 'mc_n', 'instance_i', 'agent_type', 'action_type', 'state_type',
                           'method', 'makespan', 'time']
    result_rule['problem'] = result_rule['benchmark'] + ' ' + result_rule['job_n'].astype('string') + 'x' \
                             + result_rule['mc_n'].astype('string')
    result_rule = result_rule[result_rule['action_type'] == 'single_mc_buffer']
    result_rule = result_rule.drop(['benchmark', 'job_n', 'mc_n', 'agent_type', 'action_type', 'state_type',
                                    'time'], axis=1)

    # result + opt ##################################################################
    result_opt2 = result_opt.drop(['benchmark', 'job_n', 'mc_n', 'UB', 'name'], axis=1)
    result_opt2['makespan'] = result_opt['UB']
    result_opt2['method'] = 'OPT'

    result_data = pd.concat([result_ours, result_ref, result_rule, result_cp, result_opt2])
    # result_data = pd.concat([result_ref, result_rule,  result_opt2, result_ours])
    merge_data = pd.merge(result_data, result_opt, on=['problem', 'instance_i'], how="left")

    merge_data = merge_data.dropna(subset=['UB'])
    merge_data['opt_gap'] = (merge_data['makespan'] - merge_data['UB']) / merge_data['UB']
    if min(merge_data['opt_gap']) < 0:
        print("error: opt_module gap error ----------------------------------------------")

    merge_data = merge_data.drop(['makespan', 'UB'], axis=1)
    merge_data['job_n'] = merge_data['job_n'].astype('int')
    merge_data['mc_n'] = merge_data['mc_n'].astype('int')

    return merge_data


def get_load_result_data_ablation(test_csv_files=[]) -> pd.DataFrame:
    # opt_module result data #########################################################
    result_opt = pd.read_csv("./basic/bench_opt.csv")
    result_opt.columns = ['benchmark', 'job_n', 'mc_n', 'instance_i',
                          'opt_makespan', 'convergence', 'UB', 'name', 'set']
    result_opt['problem'] = result_opt['benchmark'] + ' ' + result_opt['job_n'].astype('string') + 'x' \
                            + result_opt['mc_n'].astype('string')
    result_opt = result_opt.drop(['opt_makespan', 'convergence', 'set'], axis=1)

    # ours result data #########################################################
    result_ours = pd.read_csv("./basic/bench_ours.csv")
    result_ours.columns = ['benchmark', 'job_n', 'mc_n', 'instance_i', 'agent_type', 'action_type', 'state_type',
                           'method', 'makespan', 'time', 'unit_t', 'decision_n']
    result_ours['problem'] = result_ours['benchmark'] + ' ' + result_ours['job_n'].astype('string') + 'x' \
                             + result_ours['mc_n'].astype('string')
    result_ours = result_ours.drop(['benchmark', 'job_n', 'mc_n', 'agent_type',
                                    'time', 'unit_t', 'decision_n', 'method'], axis=1)
    result_ours['env_type'] = 'dyn'
    result_ours['model_type'] = 'type_all_pred_GAT2_mean_global'
    result_ours['model_i'] = -1

    # rule data #####################################################################
    def get_data(test_csv_file_name):
        result_test = pd.read_csv(test_csv_file_name, header=None)
        try:
            result_test.columns = ['benchmark', 'job_n', 'mc_n', 'instance_i', 'agent_type', 'action_type',
                                   'state_type',
                                   'method', 'rl', 'decay', 'loss',
                                   'makespan', 'time', 'decision_n', 'model_i',
                                   'env_type', 'model_type', 'dyn_type', 'parameter']
        except:
            try:
                result_test.columns = ['benchmark', 'job_n', 'mc_n', 'instance_i', 'agent_type', 'action_type',
                                       'state_type',
                                       'method', 'rl', 'decay', 'loss',
                                       'makespan', 'time', 'decision_n', 'model_i', 'dyn_type', 'parameter']
                result_test['env_type'] = ''
                result_test['model_type'] = ''
            except:
                result_test.columns = ['benchmark', 'job_n', 'mc_n', 'instance_i', 'agent_type', 'action_type',
                                       'state_type',
                                       'method', 'rl', 'decay', 'loss',
                                       'makespan', 'time', 'decision_n', 'model_i']
                result_test['dyn_type'] = ''
                result_test['parameter'] = ''
                result_test['env_type'] = ''
                result_test['model_type'] = ''

        result_test['problem'] = result_test['benchmark'] + ' ' + result_test['job_n'].astype('string') + 'x' \
                                 + result_test['mc_n'].astype('string')
        result_test = result_test.drop(['benchmark', 'job_n', 'mc_n', 'agent_type',
                                        'rl', 'decay', 'loss', 'dyn_type', 'parameter',
                                        'time', 'decision_n', 'method'], axis=1)  # , 'dyn_type', 'parameter'
        return result_test

    data = list()
    for test_csv_file_name in test_csv_files:
        data.append(get_data(test_csv_file_name))
    result_data = pd.concat(data + [result_ours])

    # result_data['method'] = result_data['method'] + result_data['model_i'].astype('string')
    # result_data['method'] = ''

    # result + opt ##################################################################
    merge_data = pd.merge(result_data, result_opt, on=['problem', 'instance_i'], how="left")

    merge_data = merge_data.dropna(subset=['UB'])
    merge_data['opt_gap'] = (merge_data['makespan'] - merge_data['UB']) / merge_data['UB']
    if min(merge_data['opt_gap']) < 0:
        print("error: opt_module gap error ----------------------------------------------")

    merge_data = merge_data.drop(['makespan', 'UB'], axis=1)
    merge_data['job_n'] = merge_data['job_n'].astype('int')
    merge_data['mc_n'] = merge_data['mc_n'].astype('int')

    return merge_data


# load data ##############################################################################################
def get_x_dim():
    """
    get feature dimension
    """
    from environment.env import JobShopEnv

    rollout_type = configs.rollout_type
    configs.rollout_type = 'model'

    sample_env = JobShopEnv([('HUN', 4, 3, 0)], pomo_n=1)
    obs, _, _ = sample_env.reset()

    in_dim_op = obs['op'].x.shape[1]
    if 'mc_node' in configs.state_type:
        in_dim_rsc = obs['rsc'].x.shape[1]
    else:
        in_dim_rsc = 0

    configs.rollout_type = rollout_type
    return in_dim_op, in_dim_rsc


# env ##############################################################################################
def get_env(problem_set: list, pomo_n: int=1):
    from environment.env import JobShopEnv

    env_list = list()
    for benchmark, job_n, mc_n, instances in problem_set:
        env_list += [(benchmark, job_n, mc_n, instance_i) for instance_i in instances]

    return JobShopEnv(env_list, pomo_n=pomo_n)




# total 242 instances ####################################################################################
TA = [['TA', 15, 15, list(range(10))],
      ['TA', 20, 15, list(range(10))], ['TA', 20, 20, list(range(10))],
      ['TA', 30, 15, list(range(10))], ['TA', 30, 20, list(range(10))],
      ['TA', 50, 15, list(range(10))], ['TA', 50, 20, list(range(10))],
      ['TA', 100, 20, list(range(10))], ]  # 80
LA = [['LA', 10, 5, list(range(5))], ['LA', 15, 5, list(range(5))], ['LA', 20, 5, list(range(5))],
      ['LA', 10, 10, list(range(5))], ['LA', 15, 10, list(range(5))], ['LA', 20, 10, list(range(5))],
      ['LA', 30, 10, list(range(5))], ['LA', 15, 15, list(range(5))], ]  # 40
DMU = [['DMU', 20, 15, list(range(10))], ['DMU', 20, 20, list(range(10))],
       ['DMU', 30, 15, list(range(10))], ['DMU', 30, 20, list(range(10))],
       ['DMU', 40, 15, list(range(10))], ['DMU', 40, 20, list(range(10))],
       ['DMU', 50, 15, list(range(10))], ['DMU', 50, 20, list(range(10))], ]  # 80
SWV = [['SWV', 20, 10, list(range(5))], ['SWV', 20, 15, list(range(5))], ['SWV', 50, 10, list(range(10))], ]  # 20
ABZ = [['ABZ', 10, 10, list(range(2))], ['ABZ', 20, 15, list(range(3))], ]  # 5
ORB = [['ORB', 10, 10, list(range(10))], ]  # 10
YN = [['YN', 20, 20, list(range(4))]]  # 4
FT = [['FT', 6, 6, list(range(1))], ['FT', 10, 10, list(range(1))], ['FT', 20, 5, list(range(1))], ]  # 3

all_benchmarks = TA + LA + ABZ + FT + ORB + SWV + YN + DMU

TA_small = [['TA', 15, 15, list(range(10))],
      ['TA', 20, 15, list(range(10))], ['TA', 20, 20, list(range(10))],
      ['TA', 30, 15, list(range(10))], ]  # < 500 operations

# total 86 instances ####################################################################################
TA_dyn = [['TA', 50, 15, list(range(10))], ['TA', 50, 20, list(range(10))], ['TA', 100, 20, list(range(10))]]  # 30
LA_dyn = [['LA', 15, 5, list(range(5))], ['LA', 20, 5, list(range(5))], ['LA', 30, 10, list(range(5))]]  # 15
DMU_dyn = [['DMU', 40, 15, list(range(5))], ['DMU', 50, 15, list(range(5))], ['DMU', 50, 20, list(range(5))],
           ['DMU', 40, 15, list(range(5, 10))], ['DMU', 50, 15, list(range(5, 10))], ['DMU', 50, 20, list(range(5, 10))],
           ]  # 30
SWV_dyn = [['SWV', 50, 10, list(range(10))]]  # 10
FT_dyn = [['FT', 20, 5, list(range(1))]]  # 1

all_dyn_benchmarks = DMU_dyn + TA_dyn + LA_dyn + SWV_dyn + FT_dyn

# real distribution ####################################################################################
REAL = [['REAL', 100, 20, list(range(40))], ['REAL', 100, 50, list(range(40))],
        ['REAL', 300, 20, list(range(40))], ['REAL', 300, 50, list(range(40))]]
# REAL_D = [['REAL_D', 200, 20, list(range(10))], ['REAL_D', 300, 20, list(range(10))]]
REAL_D = [['REAL_D', 200, 20, list(range(40))], ['REAL_D', 300, 20, list(range(40))]]

FLOW = [['FLOW', 10, 5, list(range(10))], ['FLOW', 10, 10, list(range(10))],
        ['FLOW', 20, 5, list(range(10))], ['FLOW', 20, 10, list(range(10))],
        ['FLOW', 50, 5, list(range(10))], ['FLOW', 50, 10, list(range(10))]]


# dispatching rules ######################################################################################
action_types = ['conflict', 'buffer']

all_rules = ['LTT', 'MOR', 'FDD/MWKR', 'LRPT', 'SPT', 'STT', 'SRPT', 'LOR', 'LPT', 'FIFO']
rules_5 = ['LTT', 'MOR', 'FDD/MWKR', 'LRPT', 'SPT']
rules_3 = ['LTT', 'MOR', 'FDD/MWKR']
rules_1 = ['LTT']

rules_4 = ['LTT', 'MOR', 'FDD/MWKR', 'SPT']
rules_2 = ['LTT', 'MOR']

def get_rules(rollout_n):
    if configs.rollout_type == 'model_rule':
        if rollout_n == 3:
            rules = rules_2
        elif rollout_n == 5:
            rules = rules_4
        elif rollout_n == 1:
            rules = []
        else:
            raise NotImplementedError

    elif configs.rollout_type == 'rule':
        if rollout_n == 3:
            rules = rules_3
        elif rollout_n == 5:
            rules = rules_5
        elif rollout_n == 1:
            rules = rules_1
        else:
            raise NotImplementedError
    else:
        rules = []

    return rules


def get_model_i_list(rollout_n):
    # -DMU
    # 0: 0.08714
    # 1: 0.08768
    # 2: 0.08213
    # 3: 0.08033
    # 4: 0.08659

    if rollout_n == 3:
        i_list = [3, 2, 4]
    elif rollout_n == 5:
        i_list = [0, 1, 2, 3, 4]
    # elif rollout_n == 1:
    #     i_list = [3]
    else:
        raise NotImplementedError

    return i_list


# value estimator ####################################################################################
def get_quantile_inter(v_list: list):  # interpolation
    N = len(v_list)
    v_list.sort()
    tau = [(2*i+1)/2/N for i in range(N)]
    v = [(v_list[i+1]-v_list[0])*(tau[0]-configs.q_level)/(tau[i+1]-tau[0]) for i in range(N-1)]
    return sum(v) / N


def get_quantile_weighted(v_list: list):
    N = len(v_list)
    v_list.sort(reverse=True)
    tau = [(2*i+1)/2/N for i in range(N)]
    tau = [round(tau_/sum(tau), 3) for tau_ in tau]
    v = 0
    for i, w in enumerate(tau):
        v += (w * v_list[i])

    return v


from scipy.stats import beta, norm, gamma, t, chi2


def get_quantile_beta(v_list: list):
    try:
        param = beta.fit(v_list)
    except:
        try:
            param = beta.fit(v_list[:-1])
        except:
            try:
                param = beta.fit(v_list[:-2])
            except:
                return min(v_list)
    X = beta(param[0], param[1], param[2], param[3])
    return X.ppf(configs.q_level)


def get_quantile_norm(v_list: list):
    param = norm.fit(v_list)
    X = norm(param[0], param[1])
    return X.ppf(configs.q_level)


def get_quantile_gamma(v_list: list):
    param = gamma.fit(v_list)
    X = gamma(param[0], param[1], param[2])
    return X.ppf(configs.q_level)


def get_quantile_t(v_list: list):
    param = t.fit(v_list)
    X = t(param[0], param[1], param[2])
    return X.ppf(configs.q_level)


def get_quantile_chi2(v_list: list):
    param = chi2.fit(v_list)
    X = chi2(param[0], param[1], param[2])
    return X.ppf(configs.q_level)


def quantile_test(v_list):
    print('beta', get_quantile_beta(v_list))
    print('norm', get_quantile_norm(v_list))
    print('gamma', get_quantile_gamma(v_list))
    print('t', get_quantile_t(v_list))
    print('chi2', get_quantile_chi2(v_list))

    print('weighted', get_quantile_weighted(v_list))


if __name__ == "__main__":
    # data = load_data('FT', 6, 6, 0)
    # opt_mc_seq = load_opt_sol('FT', 6, 6, 0)
    configs.q_level = 0.01

    v_list = [110, 260, 270, 280, 290]
    quantile_test(v_list)

    v_list = [170, 180, 190, 200, 210]
    quantile_test(v_list)

    v_list = [120, 190, 200, 210, 280]
    quantile_test(v_list)

    v_list = [120, 160, 200, 240, 280]
    quantile_test(v_list)

    v_list = [120, 150, 175, 275, 280]
    quantile_test(v_list)

    v_list = [120, 140, 185, 275, 280]
    quantile_test(v_list)

    v_list = [120, 125, 215, 260, 280]
    quantile_test(v_list)

    v_list = [110, 125, 260, 280, 290]
    quantile_test(v_list)

    v_list = [110, 200, 210, 220, 280]
    quantile_test(v_list)

    v_list = [119, 125, 260, 280, 290]
    quantile_test(v_list)

    v_list = [119, 180, 210, 220, 270]
    quantile_test(v_list)

    print()