from params import configs
from agent_BC3 import AgentBC

from utils import HUN_100

configs.agent_type = 'GNN_TS_value'
configs.env_type = 'dyn'  # 'dyn', ''
configs.state_type = 'mc_gap_mc_load_prt_norm'  # 'basic_norm', 'simple_norm', 'norm', 'mc_load_norm'
configs.model_type = 'type_all_pred_GAT2'  # 'type_all_pred_GAT2'
# configs.model_type = 'type_all_pred_GAT2'  # 'type_all_pred_GAT2'

configs.value_learn_type = ''
configs.model_global_type = 'mm_global'

valid_problem_set = [('TA', 15, 15, list(range(10)))]

for data_set in [HUN_100]:  # HUN_2, HUN_5, HUN_10, HUN_20, HUN_30, HUN_40,
    configs.training_len = len(data_set[0][3])

    for configs.action_type in ['conflict']:  # action_types ['single_mc_conflict', 'conflict', 'buffer_being', 'single_mc_buffer', 'single_mc_buffer_being']
        for i in [0, 1, 2, 3, 4]:
            agent = AgentBC(model_i=i)
            agent.learning_opt(valid_problem_set, data_set)

# TA15x15 - OPT: 1228.9 / schedule_Net: 1417.2 / ICML: 1397.5 / ours: 1350.7

# LA10x10 - OPT: 864.2 / schedule_Net: 966.8 / ours: 953.6 / ICML: 932.4
# LA15x10 - OPT: 983.4 / schedule_Net: 1127.6 / ours: 1066.8 / ICML: 1123
# LA15x15 - OPT: 1263.2 / schedule_Net: 1466.6 / ours: 1411.2 / ICML: 1400