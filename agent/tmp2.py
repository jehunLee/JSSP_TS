from utils import get_rules, TA_small
from params import configs
from agent_TS import AgentTS


configs.expansion_type = 'bound_all'
# configs.pruning_type = 'with_best'
# configs.action_type = 'conflict'

for configs.rollout_n in [1]:
    for configs.beam_n in [1]:  # 5
        for configs.value_type in ['best']:
            if configs.rollout_n == 1 and configs.value_type != 'best':
                continue

            for configs.rollout_type in ['model']:  # model_rule
                rules = get_rules(configs.rollout_n)

                agent = AgentTS()

                save_path = f'./../result/model_beam6.csv'
                agent.perform_beam_benchmarks([['TA', 15, 15, [6]]], save_path=save_path, rules=rules)
