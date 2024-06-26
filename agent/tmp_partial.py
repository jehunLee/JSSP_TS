from utils import TA_small, small_benchmarks
from params import configs
from agent_TS import AgentTS


save_path = f'./../result/model_beam_1.csv'
configs.expansion_type = 'bound_all'
configs.unit_t = 1

from utils import LA, ABZ, FT, ORB
# small_benchmarks = [['TA', 20, 15, list(range(1, 10))]] + ABZ + FT + ORB + \
#                    [['SWV', 20, 10, list(range(5))], ['SWV', 20, 15, list(range(5))]] + \
#                    [['DMU', 20, 15, list(range(10))]]


for configs.depth in [1]:  # 3, 5
    for configs.rollout_depth in [1e6]:  # 3, 5,
        if configs.rollout_depth < 1e6 and 'bound' in configs.expansion_type:
            continue

        for configs.rollout_n in [3]:
            for configs.value_type in ['best']:  # quantile_LB, best, mean, norm, beta, t, chi2
                if configs.rollout_n == 1 and configs.value_type != 'best':
                    continue

                for configs.beam_n in [3]:  # 5
                    if 'with_best' in configs.pruning_type and configs.beam_n == 1 and configs.value_type != 'best':
                        continue

                    for configs.rollout_type in ['rule']:  # model_rule, rule, multi_model

                        print(configs.depth, configs.rollout_depth, configs.rollout_n, configs.value_type,
                              configs.beam_n, configs.rollout_type)
                        agent = AgentTS()
                        agent.perform_beam_benchmarks(small_benchmarks, save_path=save_path)
                        # agent.perform_beam_benchmarks([['LA', 20, 5, list(range(0, 5))]], save_path=save_path)

