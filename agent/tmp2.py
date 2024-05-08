from utils import HUN_100, HUN_200
from params import configs
from agent_BC3 import AgentBC


# configs.agent_type = 'GNN_TS_value'
configs.env_type = 'dyn'  # 'dyn', ''
configs.state_type = 'mc_gap_mc_load_prt_norm'  # 'basic_norm', 'simple_norm', 'norm', 'mc_load_norm'
configs.model_type = 'type_all_pred_GAT2'  # 'type_all_pred_GAT2'
# configs.model_type = 'type_all_pred_GAT2'  # 'type_all_pred_GAT2'

configs.value_learn_type = 'LB_norm'
configs.model_value_type = 'mm_global'

valid_problem_set = [('TA', 15, 15, list(range(10)))]

for data_set in [HUN_200]:  # HUN_2, HUN_5, HUN_10, HUN_20, HUN_30, HUN_40,
    configs.training_len = len(data_set[0][3])

    for configs.action_type in [
        'conflict']:  # action_types ['single_mc_conflict', 'conflict', 'buffer_being', 'single_mc_buffer', 'single_mc_buffer_being']
        for i in [0, 1, 2, 3, 4]:
            if i > 0:
                configs.value_learn_type = 'LB_norm'
                configs.model_value_type = 'mm_global'
                agent = AgentBC(model_i=i)
                agent.learning_opt(valid_problem_set, data_set)

            configs.value_learn_type = ''
            configs.model_value_type = 'mm_global'
            agent = AgentBC(model_i=i)
            agent.learning_opt(valid_problem_set, data_set)