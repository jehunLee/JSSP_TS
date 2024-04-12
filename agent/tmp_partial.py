from utils import TA_small
from params import configs
from agent_TS import AgentTS


save_path = f'./../result/model_beam_1.csv'
configs.expansion_type = 'bound_all'


for configs.depth in [1, 2]:
    for configs.rollout_depth in [1e6]:  # 3, 5,
        if configs.rollout_depth < 1e6 and 'bound' in configs.expansion_type:
            continue

        for configs.rollout_n in [1, 3, 5]:
            for configs.value_type in ['best']:  # quantile_LB, best, mean, norm, beta, t, chi2
                if configs.rollout_n == 1 and configs.value_type != 'best':
                    continue

                for configs.beam_n in [3, 5]:  # 5
                    if 'with_best' in configs.pruning_type and configs.beam_n == 1 and configs.value_type != 'best':
                        continue

                    for configs.rollout_type in ['model_rule']:  # model_rule, rule, multi_model
                        if 'multi_model' in configs.rollout_type and configs.rollout_n == 1:
                            continue

                        agent = AgentTS()
                        agent.perform_beam_benchmarks(TA_small, save_path=save_path)
                        # agent.perform_beam_benchmarks([['HUN', 6, 4, list(range(3, 10))]], save_path=save_path)