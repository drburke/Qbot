from subprocess import call
import pickle

# call("python LunarLander.py", shell=True)

LOG_BASE = './logs/5/'

learning_rate_options = [4e-3,1e-3,4e-4]

num_worker_options = [8]

num_optimizer_options = [2]

n_step_return_options = [8]

min_batch_options = [32]

loss_v_options = [0.5]

loss_entropy_options = [0.01,0.04,0.1]

alpha_options = [0.05]

hidden_size_options = [16,32,64]

gpu = "1"

for learning_rate_index in range(len(learning_rate_options)):
	learning_rate = learning_rate_options[learning_rate_index]

	for num_workers in num_worker_options:

		for num_optimizer in num_optimizer_options:

			for n_step_return in n_step_return_options:

				for min_batch in min_batch_options:

					for loss_v_index in range(len(loss_v_options)):
						loss_v = loss_v_options[loss_v_index]

						for loss_entropy_index in range(len(loss_entropy_options)):
							loss_entropy = loss_entropy_options[loss_entropy_index]

							for alpha_index in range(len(alpha_options)):
								alpha = alpha_options[alpha_index]

								for hidden_size in hidden_size_options:


									log_msg = 'lr='+str(learning_rate)+'_nw='+str(num_workers)+'_no='+str(num_optimizer)+\
										'_nsr='+str(n_step_return)+'_bs'+str(min_batch)+'_lv='+str(loss_v)+'_le='+str(loss_entropy)+\
										'_ai='+str(alpha)+'_hs='+str(hidden_size)

									hyper_params = {}
									hyper_params['learning_rate'] = learning_rate
									hyper_params['num_workers'] = num_workers
									hyper_params['num_optimizer'] = num_optimizer
									hyper_params['n_step_return'] = n_step_return
									hyper_params['min_batch'] = min_batch
									hyper_params['loss_v'] = loss_v
									hyper_params['loss_entropy'] = loss_entropy
									hyper_params['alpha'] = alpha
									hyper_params['hidden_size'] = hidden_size
									hyper_params['log_path'] = LOG_BASE + log_msg
									hyper_params['gpu'] = gpu
				
									with open('hyper_params.pickle','wb') as f:
										pickle.dump(hyper_params,f)

									call("python LunarLander.py", shell=True)

									

									print(LOG_BASE+log_msg)
									print('done')
	# 								break
	# 							break
	# 						break
	# 					break
	# 				break
	# 			break
	# 		break
	# 	break
	# break

# LOG_PATH = './logs/2/logs_run'
# #ENV = 'CartPole-v1'
# RUN_TIME = 600
# NUM_THREADS = 8
# NUM_OPTIMIZERS = 2
# THREAD_DELAY = 0.001


# GAMMA = 0.99
# N_STEP_RETURN = 8
# GAMMA_N = GAMMA ** N_STEP_RETURN

# MIN_BATCH = 128
# LEARNING_RATE = 5e-4

# LOSS_V = 0.5
# LOSS_ENTROPY = 0.01

# ALPHA = 0.05
# HIDDEN_SIZE = 64