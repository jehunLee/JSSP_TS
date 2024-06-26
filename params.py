import argparse
import torch

parser = argparse.ArgumentParser(description='GNN_JSSP')

# device
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                    help='computing device type: cpu or gpu')
# parser.add_argument('--device', type=str, default='cpu')

# environment
parser.add_argument('--state_type', type=str, default='mc_gap_mc_load_prt_norm', help='state type: mc_gap_mc_load_prt_norm, mc_gap_mc_load, prt_norm')
parser.add_argument('--env_type', type=str, default='dyn', help='dynamics of scheduling environment: dyn, ')
parser.add_argument('--model_type', type=str, default='type_all_pred_GAT2', help='GNN model type : type_GAT2, type_all_pred_GAT2, type_all_pred_GAT2')
parser.add_argument('--model_global_type', type=str, default='mean_global', help='global state information of GNN model: mean_global, mmm_global, ')
parser.add_argument('--action_type', type=str, default='conflict', help='action type of MDP: buffer, conflict')
parser.add_argument('--agent_type', type=str, default='GNN_BC_policy', help='agent type: GNN_BC_policy, rule')

# dynamic environment
parser.add_argument('--dyn_type', type=str, default='', help='dynamic environment type: job_arrival, mc_breakdown_known, prt_stochastic_known, ')
parser.add_argument('--parameter', type=float, default=0, help='')
parser.add_argument('--dyn_reserve_reset', type=bool, default=True, help='')

# model update
parser.add_argument('--loss_type', type=str, default='MSE', help='loss function of imitation learning')
parser.add_argument('--max_ep_n', type=int, default=500, help='number of learning epochs')
parser.add_argument('--batch_size', type=int, default=1024, help='learning batch size')
parser.add_argument('--training_len', type=int, default=500, help='number of instances of learning target with a problem size')

parser.add_argument('--L2_norm_w', type=float, default=1e-5, help='regularization factor, L2 norm')
# parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--scheduler_type', type=str, default='step', help='adaptive function of learning rate')
parser.add_argument('--optimizer_type', type=str, default='adam', help='update function of gradient descent')
parser.add_argument('--max_grad_norm', type=float, default=1.0, help='maximum normalization value of gradient')

# model
parser.add_argument('--hi_dim', type=int, default=8, help='hidden dimension of NN models')
parser.add_argument('--em_dim', type=int, default=8, help='output dimension of NN models')
parser.add_argument('--aggr_n', type=int, default=2, help='number of aggregations of GNN')
parser.add_argument('--layer_n', type=int, default=2, help='number of layers in a NN model')
parser.add_argument('--dropout_p', type=float, default=0.2, help='probability of dropout in a NN model')
parser.add_argument('--attn_head_n', type=int, default=2, help='number of head of attention mechanism')

parser.add_argument('--self_loop', type=bool, default=True, help='self loop of NN model, True: residual NN')
# parser.add_argument('--softmax_tau', type=float, default=1.0, help='Boltzmann temperature of softmax')
parser.add_argument('--softmax_tau', type=int, default=1, help='Boltzmann temperature of softmax')

parser.add_argument('--act_type', type=str, default='leaky_relu', help='activation function type')
parser.add_argument('--batch_norm_TF', type=bool, default=False, help='batch normalization')
parser.add_argument('--duel_TF', type=bool, default=False, help='duel function of actor')

# learning target of imitation learning
parser.add_argument('--sol_type', type=str, default='full_active', help='learning solution type')
parser.add_argument('--rollout_type', type=str, default='model', help='rollout type: model, rule')
parser.add_argument('--policy_symmetric_TF', type=bool, default=False, help='learning target ')

#####################################################
parser.add_argument('--pruning_type', type=str, default='terminal_with_best', help='terminal_with_best, with_best')
parser.add_argument('--expansion_type', type=str, default='bound_all', help='bound_all, all, bound')

parser.add_argument('--q_level', type=float, default=0.01, help='')
parser.add_argument('--value_type', type=str, default='quantile_weighted', help='best, mean, quantile_beta, quantile_norm, quantile_weighted')

parser.add_argument('--beam_n', type=int, default=3, help='beam width')
parser.add_argument('--depth', type=int, default=1e6, help='maximum depth')
parser.add_argument('--rollout_depth', type=int, default=1e6, help='maximum rollout depth')

parser.add_argument('--value_learn_type', type=str, default='', help='maximum depth')
parser.add_argument('--model_value_type', type=str, default='mm_global', help='maximum rollout depth')

parser.add_argument('--unit_t', type=float, default=1e6, help='maximum rollout depth')

#####################################################
parser.add_argument('--RL_max_ep_n', type=int, default=500, help='number of learning epochs')
parser.add_argument('--train_env_n', type=int, default=20, help='number of learning epochs')
parser.add_argument('--gamma', type=float, default=0.9, help='number of learning epochs')

parser.add_argument('--lr', type=float, default=1e-6, help='learning rate')


configs = parser.parse_args()

