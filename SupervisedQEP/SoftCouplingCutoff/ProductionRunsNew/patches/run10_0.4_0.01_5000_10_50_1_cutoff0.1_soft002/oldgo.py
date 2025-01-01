M=10
finite_nudging=0.4
learning_rate=0.01
num_batches=5000
batch_size=10
test_step=50
seed=1
coupling_bound=0.1

# M=...
# finite_nudging=...
# learning_rate=...
# num_batches=...
# batch_size=...
# test_step=...
# seed=...

import qep
import numpy as np
import time

symm_nudging=True

n_test_samples=200

# set up phase sensor (Hamiltonian terms etc)
N_sys=8
N_sens=2

x_test_samples,y_test_samples=qep.sample_batch_cluster_Ising_phase_sensor(n_test_samples)

x_grid_samples=qep.get_cluster_Ising_grid_samples(20)


def shifted_corners_sampler_large(batch_size):
  return qep.sample_batch_cluster_Ising_phase_sensor_around_corners(batch_size,
                      sampling_locations=np.array([[1.0,.5,0.4],
                        [2.5,.5,0.4], [.5,3.0,0.4]]))


start=time.time()
for training_seed in np.arange(10)+1000*seed:
    print(training_seed)
    np.random.seed(training_seed)
    
    phase_sensor = qep.prepare_cluster_Ising_phase_sensor(N_sys,N_sens)
    phase_sensor = qep.initialize_phase_sensor_parameters(phase_sensor,qep.optax.adam(learning_rate))

    phase_sensor,training_history=qep.train_phase_sensor(phase_sensor,
                            shifted_corners_sampler_large,
                            qep.cost_function_mse,qep.grad_cost_function_mse,
                                learning_rate,batch_size,num_batches,
                            test_step=test_step,test_samples=(x_test_samples,y_test_samples),
                                                nonsparse=False,
                                                         finite_nudging=finite_nudging,
                                                         symmetric_nudging=symm_nudging,
                                                         shot_noise=M,
                                                         alternative_accuracies=True,
                                                         coupling_bound=coupling_bound)

    y_pred_grid=qep.evaluate_phase_sensor_on_batch(phase_sensor,x_grid_samples)

    np.savez(f"results{training_seed}.npz",
             sys_sens_couplings=phase_sensor['sys_sens_couplings'],
             sens_couplings=phase_sensor['sens_couplings'],
             accuracy=training_history['accuracy'],
             cost=training_history['cost'],
             training_cost=training_history['training_cost'],
             product_accuracy=training_history['product_accuracy'],
             single_shot_accuracy=training_history['single_shot_accuracy'],
             max_choice_accuracy=training_history['max_choice_accuracy'],
             y_pred_grid=y_pred_grid)
        

print("total time needed (sec):", time.time()-start)
    




