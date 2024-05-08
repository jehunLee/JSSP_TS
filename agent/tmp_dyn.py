from utils import TA_small, small_benchmarks
from params import configs
from agent_TS import AgentTS


# configs.dyn_type = 'mc_breakdown_known'
# parameters = [100, 200, 300, 400]
# test_set = small_benchmarks


# configs.dyn_type = 'job_arrival'
# parameters = [100, 200, 300, 400]
# configs.init_batch = 2
# # test_set = all_dyn_benchmarks
# test_set = [['TA', 100, 20, list(range(10))], ['TA', 50, 20, list(range(10))]]


configs.dyn_type = 'prt_stochastic_known'
parameters = [0.1, 0.2, 0.3, 0.4]
test_set = small_benchmarks


save_path = f'./../result/model_beam_{configs.dyn_type}_1.csv'
configs.expansion_type = 'bound_all'


for configs.depth in [1]:  # 3, 5
    for configs.rollout_depth in [1e6]:  # 3, 5,
        if configs.rollout_depth < 1e6 and 'bound' in configs.expansion_type:
            continue

        for configs.rollout_n in [1, 3]:
            for configs.value_type in ['best']:  # quantile_LB, best, mean, norm, beta, t, chi2
                if configs.rollout_n == 1 and configs.value_type != 'best':
                    continue

                for configs.beam_n in [3]:  # 5
                    if 'with_best' in configs.pruning_type and configs.beam_n == 1 and configs.value_type != 'best':
                        continue

                    for configs.rollout_type in ['multi_model']:  # model_rule, rule, multi_model
                        if 'multi_model' in configs.rollout_type and configs.rollout_n == 1:
                            configs.rollout_type = 'model'

                        for configs.parameter in parameters:

                            print(configs.depth, configs.rollout_depth, configs.rollout_n, configs.value_type,
                                  configs.beam_n, configs.rollout_type)
                            agent = AgentTS()
                            agent.perform_beam_benchmarks(test_set, save_path=save_path)





configs.dyn_type = 'mc_breakdown_known'
parameters = [100, 200, 300, 400]
test_set = small_benchmarks


# configs.dyn_type = 'job_arrival'
# parameters = [100, 200, 300, 400]
# configs.init_batch = 2
# # test_set = all_dyn_benchmarks
# test_set = [['TA', 100, 20, list(range(10))], ['TA', 50, 20, list(range(10))]]


# configs.dyn_type = 'prt_stochastic_known'
# parameters = [0.1, 0.2, 0.3, 0.4]
# test_set = small_benchmarks


save_path = f'./../result/model_beam_{configs.dyn_type}_1.csv'
configs.expansion_type = 'bound_all'
test_set = [['LA', 10, 10, [1]]]


for configs.depth in [1]:  # 3, 5
    for configs.rollout_depth in [1e6]:  # 3, 5,
        if configs.rollout_depth < 1e6 and 'bound' in configs.expansion_type:
            continue

        for configs.rollout_n in [1, 3]:
            for configs.value_type in ['best']:  # quantile_LB, best, mean, norm, beta, t, chi2
                if configs.rollout_n == 1 and configs.value_type != 'best':
                    continue

                for configs.beam_n in [3]:  # 5
                    if 'with_best' in configs.pruning_type and configs.beam_n == 1 and configs.value_type != 'best':
                        continue

                    for configs.rollout_type in ['multi_model']:  # model_rule, rule, multi_model
                        if 'multi_model' in configs.rollout_type and configs.rollout_n == 1:
                            configs.rollout_type = 'model'

                        for configs.parameter in parameters:

                            print(configs.depth, configs.rollout_depth, configs.rollout_n, configs.value_type,
                                  configs.beam_n, configs.rollout_type)
                            agent = AgentTS()
                            agent.perform_beam_benchmarks(test_set, save_path=save_path)
