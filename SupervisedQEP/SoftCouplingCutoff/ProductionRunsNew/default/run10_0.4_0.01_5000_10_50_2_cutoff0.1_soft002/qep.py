import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from tqdm import tqdm
import optax

Xop=sp.sparse.csr_array([[0.,1.],[1.,0.]]) # Pauli X as a sparse matrix
Yop=sp.sparse.csr_array([[0,-1j],[1j,0.]])
Zop=sp.sparse.csr_array([[1.,0.],[0.,-1]])

def operator(op,idx,N):
  """
  Produce an operator that has single-qubit operator op acting
  on qubit with index idx, and the rest is identity. N is the total number of qubits.
  idx count starts at 0 and runs up to N-1.

  Example: Z_3
  >> operator(Zop,3,10)

  Example: X_0 X_2
  >>  operator(Xop,0,10) @ operator(Xop,2,10)
  """
  return sp.sparse.kron( sp.sparse.identity(2**idx), sp.sparse.kron(op, sp.sparse.identity(2**(N-idx-1))) )

def get_identity(N):
  """
  Return identity operator for an N-qubit system
  """
  return sp.sparse.identity(2**N)

def expectation(psi,op):
  """
  Calculate the expectation value of observable operator op with respect to state psi.
  This takes the real part automatically, assuming op is Hermitean !
  """
  return np.real( np.sum( np.conj(psi) * (op @ psi) ) )

def fluctuating_expectation(psi,op, M):
  """
  Calculate the expectation value of op with respect to psi. Also
  calculate the variance of op. Then imagine M measurement shots
  are conducted and return a Gaussian-sampled value of the empirically
  observed average, i.e. for op=A:
      A_observed ~ NormalDist(mu=<A>,sigma=sqrt(Var(A)/M))
  This is a reasonable approximation for 'large' M, where in practice
  M=10 is already large. It is not OK for M=1.
  
  Note: we assume op is Hermitean.
  """
  op_avg=np.real( np.sum( np.conj(psi) * (op @ psi) ) )
  op2_avg=np.real( np.sum( np.conj(psi) * (op @ (op @ psi) ) ) )
  sigma=np.sqrt((op2_avg-op_avg**2)/M)
  return sigma*np.random.randn()+op_avg

def empty_matrix(n_dim):
  """
  Return the empty sparse matrix for a system of Hilbert space dimension n_dim.
  """
  return sp.sparse.csr_matrix((n_dim, n_dim), dtype='complex')

def produce_XYZ(N):
  """
  Return X,Y,Z = a list of all Pauli X operators, all Pauli Y, all Pauli Z,
  for given qubit number N
  """
  return ( [operator(Xop,idx,N) for idx in range(N)] ,
            [operator(Yop,idx,N) for idx in range(N)] ,
            [operator(Zop,idx,N) for idx in range(N)] )


def ground_state(H):
  """
  Return ground state of H, using sparse Lanczos.
  Returns eigenvalue, eigenvector
  """
  evals,evecs=sp.sparse.linalg.eigsh(H, k=1, which='SA')
  return evals[0], evecs[:,0]


def QEP_Hamiltonian(H0,x,H_x_terms,theta,H_theta_terms,nu,H_nu_terms):
  """
  Construct a QEP Hamiltonian from the given components.

  H_0 - a parameter-independent Hamiltonian

  x   - a vector (list or array) of input values
  H_x_terms - a list of corresponding terms in the Hamiltonian

  theta - a vector (list or array) of trainable parameters
  H_theta_terms - a list of corresponding terms in the Hamiltonian

  nu - a vector (list or array) of nudge parameters, applied to output observables
  H_nu_terms - a list of corresponding terms in the Hamiltonian

  Note: nu may be None, in which case no H_nu_terms are added.

  Returns the Hamiltonian H as a sum of all these terms.
  """
  H=H0

  for x_term, H_term in zip(x,H_x_terms):
    H+= x_term * H_term

  for theta_term, H_term in zip(theta,H_theta_terms):
    H+= theta_term * H_term

  if nu is not None:
    for nu_term, H_term in zip(nu,H_nu_terms):
      H+= nu_term * H_term

  return H

def get_gradient_at_x(H, E, psi, error_signal, H_nu_terms, H_theta_terms, eta, 
                      nonsparse=False,finite_nudging=None,shot_noise=None,
                     symmetric_nudging=False):
  """
  Return the cost-function gradient, for updating the trainable theta parameters,
  for one fixed input x.

  Given:
    - full Hamiltonian H (provided for a particular input x)
    - its ground state energy E
    - its ground state psi
    - the H_nu_terms (list of output operators)
    - the H_theta_terms (list of trainable operators)
    - error_signal (vector of length len(H_nu_terms)
            entries given as error_j = d C / d <output_j>)
    - eta: regularization parameter for inverting E-H during perturbation
           theory (we will solve (E-H + 1j eta) x = b, with b orthogonal
           to the nullspace of E-H).
    - nonsparse: if True, use non-sparse matrix routine (is not useful)
    - finite_nudging: if not None, use this value for simulating the finite
            nudging (actual physical shift to a new ground state) instead
            of the linear response calculation
    - shot_noise: if not None, interpret this as the number of measurement
            shots and use fluctuating empirical expectation values instead
            of the idealized expectation values. Important: Can only be
            used together with finite_nudging! (see fluctuating_expectation
            for explanation of how it uses a Gaussian approximation; 
            shot_noise should be larger than say 5 to have a reasonable
            approximation to the underlying true distribution)
    - symmetric_nudging: if True, use symmetric nudging

  Return the derivative of the cost function C with respect to theta parameters,
  calculated via perturbation theory in the nudging terms (= output operators).

  gradient_k = sum_j (d C / d <output_j>) * (d <output_j> / d theta_k)

  This will be calculated via perturbation theory, using the output operators
  as perturbation. It reflects the result of the QEP procedure for
  infinitesimal nudging. For maximum efficiency, we consider the response to
  the linear superposition output operator that is
    OUT = sum_j dC/d<output_j> * output_j
  (instead of first calculating the response to each output_j operator and
  only then summing afterwards)

  Note the result returned here still has to be averaged over x, e.g. in a mini-batch.
  """
  n_dim = np.shape(H)[0] # Hilbert space dimension
  Out_operator = empty_matrix(n_dim)

  for weight,term in zip(error_signal,H_nu_terms):
    Out_operator += weight * term

  Out_expectation = expectation(psi, Out_operator)

  if finite_nudging is None:
      # Result will involve dpsi = (E-H)^(-1) @ Out_op @ psi
      # For efficiency, never invert a matrix when solving a linear system will do
      # Here, we try to subtract the projection of Out_operator @ psi onto
      # psi itself. Will it work?
      if nonsparse:
        M_full = ((E+1j*eta)*sp.sparse.identity(n_dim) - H).todense()
        dpsi = np.linalg.solve( M_full, Out_operator @ psi - Out_expectation * psi)
      else:
        dpsi = sp.sparse.linalg.spsolve( (E+1j*eta)*sp.sparse.identity(n_dim) - H ,
                                      Out_operator @ psi - Out_expectation * psi)

      # for any trainable coupling operator 'term', the response
      # will now be  <psi| term | dpsi> + c.c.
      # Thus, return the gradient dC/dtheta_j as follows:
      return np.array( [2*np.real( np.sum( np.conj(psi) *  (term @ dpsi) ) )
                          for term in H_theta_terms] )
  else: # simulate the actual physical nudging
    # first calculate old expectation values of theta terms:
    if shot_noise is None:
        get_expectation=expectation
    else:
        get_expectation=lambda psi,op: fluctuating_expectation(psi,op,shot_noise)
    
    def theta_expectations(state):
        return np.array([ get_expectation(state,term)
                            for term in H_theta_terms])
    old_expectations = theta_expectations(psi)
    H_new = H + finite_nudging * Out_operator # add the nudge
    _,psi_new = ground_state(H_new)
    new_expectations = theta_expectations(psi_new)
    if symmetric_nudging:
        H_new = H - finite_nudging * Out_operator # add the nudge
        _,psi_new = ground_state(H_new)
        negative_new_expectations = theta_expectations(psi_new)
        return (new_expectations - negative_new_expectations)/(2*finite_nudging)
    else:
        return (new_expectations - old_expectations)/finite_nudging
    
    
def gradient_for_batch(x_batch, y_batch, dC, theta, H0, H_x_terms, 
                       H_nu_terms, H_theta_terms,nonsparse=False,finite_nudging=None,
                      shot_noise=None,symmetric_nudging=False):
  """
  Get the gradient of the cost function with respect to theta, averaged
  over a mini-batch of input vectors x.

  - x_batch: batch of input vectors --
    array of shape (batchsize, x_dimension), where x_dimension equals len(H_x_terms)
  - y_batch: batch of output vectors --
    array of shape (batchsize, y_dimension), where y_dimension equals len(H_nu_terms)
  - dC: a function with two arguments dC(y,y_target) that is the derivative of the
    cost function C with respect to y, at given y_target. For a quadratic cost function,
    dC(y,y_target) = 2 * (y-y_target)
  - theta: current array of trainable parameters
  - H0: fixed part of Hamiltonian
  - H_x_terms: list (or array) of terms in Hamiltonian to be multiplied with x vector entries
  - H_nu_terms: list of output operators
  - H_theta_terms: list of terms that go along with trainable parameters (same length as theta array)
  - nonsparse: if True, use non-sparse matrix routine (is not useful)
  - finite_nudging: if not None, use this value for simulating the finite
        nudging (actual physical shift to a new ground state) instead
        of the linear response calculation
  - shot_noise: if not None, this is the number of measurement shots. Only
        valid together with finite_nuding. See get_gradient_at_x.
  - symmetric_nudging: see get_gradient_at_x
  """
  grad=np.zeros(len(theta))
  batchsize=np.shape(x_batch)[0]

  for idx in range(batchsize):
    H=QEP_Hamiltonian(H0, x_batch[idx], H_x_terms,
                      theta, H_theta_terms, None, None) # nu terms set to None
    E,psi=ground_state(H) # for this particular x
    y=np.array( [expectation(psi, out_op) for out_op in H_nu_terms] ) # all outputs
    error_signal=dC(y,y_batch[idx]) # gradient of C with respect to output
    grad+=get_gradient_at_x(H,E,psi,error_signal,H_nu_terms,H_theta_terms,
                            nonsparse,finite_nudging,shot_noise,symmetric_nudging=symmetric_nudging)

  return grad / len(x_batch)


def prepare_general_phase_sensing(N_sys,N_sens,H0_sys,H0_sens,H_sys_terms,
                                  sys_operators,sens_operators,
                                  H_sens_terms,output_terms,paulis):
  """
  Prepare all contributions for a phase recognition scenario.

  N_sys: number of qubits in system
  N_sens: number of qubits in sensor
  H0_sys: fixed part of system Hamiltonian (None if not present)
  H0_sens: fixed part of sensor Hamiltonian (None if not present)
  H_sys_terms: tuneable terms in system Hamiltonian (to move through phases)
  sys_operators: the system operators to which the sensor can couple
                (e.g. a few qubit operators in some region)
  sens_operators: the sensor operators that can couple to the system operators
                This routine will implement an all-to-all coupling, i.e.
                all pairs of sys_operators and sens_operators will be produced
  H_sens_terms: sensor operators that will enter the sensor Hamiltonian
  output_terms: the operators whose expectation values constitute the output
  paulis: tuple (X,Y,Z) containing all Pauli operators

  Returns a dictionary phase_sensor that keeps all the contributions and info.
  This can then be used to call QEP_Hamiltonian with parameter values
  to go along with the terms.

  Entries of phase_sensor:
  'N_sys': N_sys,
  'N_sens': N_sens,
  'paulis': paulis,
  'H0': H0,
  'H_x_terms': H_sys_terms,
  'H_sys_sens_terms': H_sys_sensor_terms,
  'H_sens_terms': H_sens_terms,
  'H_theta_terms': H_sys_sensor_terms + H_sens_terms,
  'output_terms': output_terms
  """

  H0 = 0*paulis[0][0] # just empty initialization
  if H0_sys is not None:
    H0 += H0_sys
  if H0_sens is not None:
    H0 += H0_sens

  # all pairwise combinations of system and sensor operators
  # -- 'fully connected' sensor
  H_sys_sensor_terms=[op_sys @ op_sens for op_sys in sys_operators
                        for op_sens in sens_operators]

  return {'N_sys': N_sys,
          'N_sens': N_sens,
          'paulis': paulis,
          'H0': H0,
          'H_x_terms': H_sys_terms,
          'H_sys_sens_terms': H_sys_sensor_terms,
          'H_sens_terms': H_sens_terms,
          'H_theta_terms': H_sys_sensor_terms + H_sens_terms,
          'output_terms': output_terms
          }

def get_H_sys_terms_for_cluster_Ising(N_sys,paulis):
  """
  Prepare all the terms needed for the cluster Ising Hamiltonian.

  N_sys: number of qubits (length of chain)
  paulis: (X,Y,Z) tuple containing previously prepared Pauli operators

  Returns a tuple containing all these terms (each of which can
  be controlled by a parameter):

  (H_ZXZ,H_ZZ,H_X)
  """
  X,Y,Z=paulis

  H_ZXZ=0*Z[0] # just to initialize proper shape
  H_ZZ=0*Z[0]
  H_X=0*Z[0]

  for j in range(N_sys):
    H_ZXZ += Z[j] @ X[(j+1)%N_sys] @ Z[(j+2)%N_sys]
    H_ZZ  -= Z[j] @ Z[(j+1)%N_sys]
    H_X   -= X[j]

  return (H_ZXZ,H_ZZ,H_X)


def QEP_Hamiltonian_for_phase_sensing(phase_sensor,x,theta):
  """
  Return the QEP Hamiltonian for the phase sensing scenario.

  phase_sensor: as returned from prepare_general_phase_sensing,
                contains all the required operators
  x: input values (i.e. the location in the phase diagram)
  theta: trainable parameters (i.e. sys-sensor coupling and sensor terms)
  """
  return QEP_Hamiltonian(phase_sensor['H0'],x,phase_sensor['H_x_terms'],
                         theta,
                         phase_sensor['H_theta_terms'],
                         None,None)


def initialize_phase_sensor_parameters(phase_sensor,use_optimizer=None,coupling_bound=None):
  """
  Initialize parameters for a given phase sensor model.

  phase_sensor:  dictionary
  returned by a routine like prepare_cluster_Ising_phase_sensor or
  prepare_general_phase_sensing.

  use_optimizer: optionally, an optax optimizer, as obtained from a call
        to optax.adam(1e-3) or similar. Its state will be stored inside
        phase_sensor and automatically used in the train routine.

  Returns: an updated phase_sensor dictionary that also contains
  trainable parameters, 'sys_sens_couplings' and 'sens_couplings'
  (both of them are viewed together as the trainable parameters theta).

  These are vectors initialized with normal Gaussian random numbers.
  """
  if coupling_bound is None:
    phase_sensor['sys_sens_couplings']=np.random.randn(len(phase_sensor['H_sys_sens_terms']))
  else:
    phase_sensor['sys_sens_couplings']=np.random.uniform(low=-coupling_bound,
                                                         high=+coupling_bound,
                                                         size= (len(phase_sensor['H_sys_sens_terms']),) )    
  phase_sensor['sens_couplings']=np.random.randn(len(phase_sensor['H_sens_terms']))
  if use_optimizer is None:
    phase_sensor['optimizer']=None
  else:
    optimizer=use_optimizer
    theta_init=np.concatenate((phase_sensor['sys_sens_couplings'],phase_sensor['sens_couplings']))
    opt_state=optimizer.init(theta_init)
    phase_sensor['optimizer']=optimizer
    phase_sensor['opt_state']=opt_state
  return phase_sensor

def evaluate_phase_sensor_on_batch(phase_sensor,x_batch,operators=None):
  """
  Evaluate phase_sensor on a batch of samples,
  x_batch of shape (batchsize,x_dim)

  Returns: y_batch of shape (batchsize,y_dim), where y_dim
      is the output dimension, and these are the expectation
      values of the output operators stored inside
      phase_sensor['output_terms']

  Optionally, operators is a list of operators whose expectation values should
  be evaluated. In that case, this function returns
    y_batch, op_values
  where op_values is of shape (batchsize,num_ops) and num_ops is the number
  of operators.
  """
  theta=np.concatenate((phase_sensor['sys_sens_couplings'],
                        phase_sensor['sens_couplings']))
  y_dim=len(phase_sensor['output_terms'])
  batch_size=np.shape(x_batch)[0]

  y_batch=np.zeros((batch_size,y_dim))
  if operators is not None:
    num_ops=len(operators)
    op_values=np.zeros((batch_size,num_ops))

  for idx,x_sample in enumerate(x_batch):
    if operators is None:
      _,_,_,y_pred_sample = evaluate_phase_sensor(phase_sensor,x_sample,theta)
    else:
      _,_,_,y_pred_sample,op_pred = evaluate_phase_sensor(phase_sensor,
                                      x_sample,theta,operators=operators)
      op_values[idx,:]=op_pred
    y_batch[idx,:] = y_pred_sample

  if operators is None:
    return y_batch
  else:
    return y_batch,op_values

def evaluate_phase_sensor(phase_sensor,x_sample,theta,operators=None):
  """
  Evaluate the phase_sensor on a single x_sample, with
  theta replacing the trainable parameters stored inside
  phase_sensor.

  Returns H,E,psi,y_pred

  If operators is a list of operators, then evaluate their expectation
  values, yielding a vector op_pred, and return H,E,psi,y_pred,op_pred
  """
  # prepare Hamiltonian for this sample:
  H=QEP_Hamiltonian_for_phase_sensing(phase_sensor,x_sample,theta)
  # get ground state:
  E,psi=ground_state(H)
  # obtain expectation values of output terms:
  y_pred=np.array([expectation(psi,op) for op in phase_sensor['output_terms']])

  if operators is None:
    return H,E,psi,y_pred
  else:
    op_pred=np.array([expectation(psi,op) for op in operators])
    return H,E,psi,y_pred,op_pred

def eval_test_phase_sensor(phase_sensor,theta,cost_function,test_samples,
                          alternative_accuracies=False):
    """
    Obtain measures of accuracy etc. 
    for a phase_sensor and a given set of test samples.
    
    phase_sensor: contains all the data of the phase sensor
    theta: can be set to None, to use the trainable parameters inside the phase sensor.
    cost_function: the cost function to be used
    alternative_accuracies: if True, 
        now also calculate the following three versions of the accuracy:
        (i) product-accuracy:
          assuming the phase indicators are independently measured (! which they
          are not in the present example !),
          what is the probability of getting the right answer in a single measurement,
          i.e. (1-P(phase1_indicator))*P(phase2_indicator)*(1-P(phase3_indicator)),
          if phase2 is the right phase!
        (ii) single-shot:
          assuming the phase indicators are measured in a single shot (as in the
          present example: ZZ is measured, and the different outcomes indicate
          different phases):
          what is the probability of getting the right answer in a single measurement,
          i.e. P(phase_j_indicator), where j was the correct choice
        (iii) max-choice:
          does one get the right answer for infinitely many measurements, where
          one simply checks that the probability for the correct outcome combination
          (e.g. 0,1,0) is higher than any other probability?
        accuracy(default),cost_value,accuracy(product),accuracy(single-shot),accuracy(max-choice)
    
    test_samples=(samples_input,samples_output), with 
    samples_input.shape=(batch_size,dim_input) and similar for samples_output.
    
    the default accuracy value is based on:
       counting probabilities>0.5 as "1" and else "0", then comparing against
       desired output pattern (e.g. 0,1,0 for phase 2) and checking whether
       the prediction matches the desired output pattern.
    
    Returns:
        accuracy,cost_value
    
    """
    correct_matches=0
    cost_value=0.0
    if alternative_accuracies:
        value_product_accuracy=0.0
        value_single_shot_accuracy=0.0
        correct_max_choice=0
        
    if theta is None:
        theta=np.concatenate((phase_sensor['sys_sens_couplings'],
                        phase_sensor['sens_couplings']))
    for x_sample,y_sample in zip(*test_samples):
      # "forward pass" (inference)
      H,E,psi,y_pred = evaluate_phase_sensor(phase_sensor,x_sample,theta)
      if alternative_accuracies:
        value_product_accuracy += np.prod(y_sample*y_pred + (1-y_sample)*(1-y_pred))
        value_single_shot_accuracy += np.prod(y_sample*y_pred + (1-y_sample))
        correct_max_choice += (np.argmax(y_sample)==np.argmax(y_pred))
      digitize_pred=y_pred>0.5
      correct_matches += (digitize_pred == y_sample).all()
      cost_value +=cost_function(y_pred,y_sample)
    n_test_samples=len(test_samples[0])
    correct_matches/=(1.0*n_test_samples)
    cost_value/=n_test_samples
    if alternative_accuracies:
      value_product_accuracy/=n_test_samples
      value_single_shot_accuracy/=n_test_samples
      correct_max_choice/=(1.0*n_test_samples)
      return correct_matches,cost_value,value_product_accuracy,value_single_shot_accuracy,correct_max_choice
    else:
      return correct_matches,cost_value
    


def train_phase_sensor(phase_sensor,training_data_sampler,
                       cost_function,grad_cost_function,
                       learning_rate,batch_size,num_batches,
                       test_samples=None,test_step=None,
                       SMALL_NUMBER=1e-8,nonsparse=False,
                      finite_nudging=None,shot_noise=None,
                      check_nudging_accuracy=False,
                      symmetric_nudging=False,
                      check_shot_noise_influence=False,
                      record_error_signal=False,
                      return_gradient=False,
                      provide_training_samples=None,
                      no_progress_bar=False,
                      alternative_accuracies=False,
                       coupling_bound=None,
                       soft_coupling_cutoff_slope=None):
  """
  Train a phase sensor via QEP. You can call this repeatedly.
  Returns the updated phase_sensor dictionary.

  phase_sensor: phase sensor dictionary as returned by a routine
      like prepare_cluster_Ising_phase_sensor or
      prepare_general_phase_sensing, and augmented with randomly initialized
      trainable parameters, via initialize_phase_sensor_parameters.

  training_data_sampler: a function of the type return_batch(batchsize)
      that will return two lists, x_samples and y_samples, that define
      the training data. Each training sample
      is of the form (x,y), where x are the input parameters (here: defining
      the location in the phase diagram) and y are the desired output values
      (here: related to the expectation values of the 'output_terms')

  cost_function: a function of the type cost(y_pred,y_target), see below.

  grad_cost_function: a function of the type grad_cost(y_pred,y_target), which
      returns a vector that is the gradient of the cost function with
      respect to y_pred. Both y_pred, y_target are vectors, of length given
      by the number of output terms, i.e. len(phase_sensor['output_terms']).

  test_samples (if not None) is equal to (test_x,test_y), where test_x
      are the inputs for the test samples and test_y the outputs
      
  test_step: number of training batches after which one test evaluation is performed.
  
  SMALL_NUMBER is used in evaluating linear response, in 1/(E-H+i SMALL_NUMBER)

  finite_nudging: if not None, use this value for simulating the finite
        nudging (actual physical shift to a new ground state) instead
        of the linear response calculation
        
  shot_noise: if not None, number of measurement shots. See get_gradient_at_x
        for explanation. Only valid together with finite_nudging.
        
  check_nudging_accuracy: if True, also record gradients evaluated at half the nudging
      parameter, for later inspection. They will be stored in training_history under
      the key 'theta_gradient_half'.
    
  symmetric_nudging: if True, use symmetric nudging
  
  check_shot_noise_influence: if True, also record what the ideal gradient without
      shot noise would have been. Record it under 'theta_gradient_no_shot_noise'.
  
  record_error_signal: if True, record norm of error vector as 'error_signal'
  
  return_gradient: if True, return only the gradient instead of performing the update
  
  provide_training_samples: if not None, this is (x_samples,y_samples); alternative to
      providing the training_data_sampler
      
  no_progress_bar: if True, do not show progress bar.
  
  alternative_accuracies: if True, also calculate three additional accuracy measures,
      adding them to training_history: product_accuracy, single_shot_accuracy, max_choice_accuracy
      (see eval_phase_sensor for their meaning)
      
  coupling_bound: if not None, this is the bound on the
      absolute magnitude of the sys_sens_couplings. 
      Anything beyond will be cut off during gradient descent.

  soft_coupling_cutoff_slope: if not None, a soft coupling cutoff will
      be imposed, where a cost function rises with this slope for any
      sys_sens_coupling beyond the coupling_bound.
      
  Returns:

  phase_sensor, training_history

  where phase_sensor is updated with the final trainable parameters,
  and training_history is a dictionary with the following entries, each a list of
  length num_batches:

  cost -- list of batch-averaged cost values
  theta -- list of theta vectors
  theta_gradient -- list of batch-averaged theta gradients
  accuracy -- list of batch-averaged default accuracies
  training_cost -- list of batch-averaged training costs (evaluated on training batches)
  
  if alternative_accuracies is True, also add:
  product_accuracy, single_shot_accuracy, max_choice_accuracy
  
  """
  LARGE_NUMBER = 1e10
  
  # trainable parameters (all of them together):
  theta=np.concatenate((phase_sensor['sys_sens_couplings'],
                        phase_sensor['sens_couplings']))
  num_sys_sens = len(phase_sensor['sys_sens_couplings'])
  if coupling_bound is not None:
    cut = np.full( (num_sys_sens,), coupling_bound )
  else:
    cut = np.full( (num_sys_sens,), LARGE_NUMBER )
  
  optimizer=phase_sensor['optimizer']
  if optimizer is not None:
    opt_state=phase_sensor['opt_state']

  output_terms=phase_sensor['output_terms']
  H_theta_terms=phase_sensor['H_theta_terms']
  num_theta_terms=len(phase_sensor['H_theta_terms'])

  # prepare training history:
  training_history={'cost': [], 'accuracy': [], 
                    'training_cost': [], 'theta': [], 'theta_gradient': []}

  if alternative_accuracies:
    training_history['single_shot_accuracy']=[]
    training_history['product_accuracy']=[]
    training_history['max_choice_accuracy']=[]
    
  if check_nudging_accuracy:
    training_history['theta_gradient_half']=[]
  if check_shot_noise_influence:
    training_history['theta_gradient_no_shot_noise']=[]
  if record_error_signal:
    training_history['error_signal']=[]
    
  if test_samples is not None:
    n_test_samples=len(test_samples[0])

  if no_progress_bar:
    training_range=range(num_batches)
  else:
    training_range=tqdm(range(num_batches))
    
  for training_step in training_range: # training loop
    theta_gradient=np.zeros((num_theta_terms,)) # set to zero initially
    if check_nudging_accuracy:
        theta_gradient_half=np.zeros((num_theta_terms,))
    if check_shot_noise_influence:
        theta_gradient_no_shot_noise=np.zeros((num_theta_terms,))
    if record_error_signal:
        abs_error_signal=0.0
        
    cost_value=0.0

    if provide_training_samples is None:
      training_samples = training_data_sampler(batch_size)
    else:
      training_samples = provide_training_samples
    
    for x_sample,y_sample in zip(*training_samples):
      # "forward pass" (inference)
      H,E,psi,y_pred = evaluate_phase_sensor(phase_sensor,x_sample,theta)
      cost_value +=cost_function(y_pred,y_sample)

      # "backward pass" (gradients)

      # obtain gradient:
      error_signal=grad_cost_function(y_pred,y_sample)
      theta_gradient+=get_gradient_at_x(H,E,psi,error_signal,output_terms,
                        H_theta_terms,SMALL_NUMBER,nonsparse,
                                       finite_nudging=finite_nudging,
                                       shot_noise=shot_noise, symmetric_nudging=symmetric_nudging)
      if check_nudging_accuracy:
            theta_gradient_half+=get_gradient_at_x(H,E,psi,error_signal,output_terms,
                        H_theta_terms,SMALL_NUMBER,nonsparse,
                                       finite_nudging=0.5*finite_nudging,
                                       shot_noise=shot_noise, symmetric_nudging=symmetric_nudging)
      if check_shot_noise_influence:
            theta_gradient_no_shot_noise+=get_gradient_at_x(H,E,psi,error_signal,output_terms,
                        H_theta_terms,SMALL_NUMBER,nonsparse,
                                       finite_nudging=0.5*finite_nudging,
                                       shot_noise=None, symmetric_nudging=symmetric_nudging)
      if record_error_signal:
        abs_error_signal+=np.sqrt(np.sum(error_signal**2))
        
    cost_value/=batch_size # average
    theta_gradient/=batch_size # average

    if soft_coupling_cutoff_slope is not None:
      cost_value += np.sum( np.maximum( soft_coupling_cutoff_slope * (np.abs(theta[:num_sys_sens])-coupling_bound), 0.0 ) )
      theta_gradient[:num_sys_sens] += soft_coupling_cutoff_slope * np.sign(theta[:num_sys_sens]) * (np.abs(theta[:num_sys_sens])>=coupling_bound)
      
    training_history['training_cost'].append(cost_value)
    training_history['theta'].append(theta)
    training_history['theta_gradient'].append(theta_gradient)

    if check_nudging_accuracy:
        theta_gradient_half/=batch_size
        training_history['theta_gradient_half'].append(theta_gradient_half)

    if check_shot_noise_influence:
        theta_gradient_no_shot_noise/=batch_size
        training_history['theta_gradient_no_shot_noise'].append(theta_gradient_no_shot_noise)
        
    if record_error_signal:
        abs_error_signal/=batch_size
        training_history['error_signal'].append(abs_error_signal)
    
    if test_samples is not None:
        if training_step%test_step==0:
            if alternative_accuracies:
                accuracy,cost_value,product_accuracy,single_shot_accuracy,max_choice_accuracy=eval_test_phase_sensor(phase_sensor,theta,
                                        cost_function,test_samples,alternative_accuracies=True)
                training_history['single_shot_accuracy'].append(single_shot_accuracy)
                training_history['product_accuracy'].append(product_accuracy)
                training_history['max_choice_accuracy'].append(max_choice_accuracy)
            else:
                accuracy,cost_value=eval_test_phase_sensor(phase_sensor,theta,
                                        cost_function,test_samples)
            training_history['cost'].append(cost_value)
            training_history['accuracy'].append(accuracy)
            training_range.set_description(f"Acc: {100*accuracy} %",refresh=True)

    # update trainable parameters:
    if return_gradient:
        return theta_gradient
    else:
        if optimizer is None:
          theta -= learning_rate * theta_gradient # standard SGD
        else:
          updates, opt_state = optimizer.update(theta_gradient, opt_state)
          theta = np.array(optax.apply_updates(theta, updates))
        if soft_coupling_cutoff_slope is None:
          # implement hard cutoff:
          theta[:num_sys_sens] = theta[:num_sys_sens]*(np.abs(theta[:num_sys_sens])<cut) + np.sign(theta[:num_sys_sens])*(np.abs(theta[:num_sys_sens])>=cut)*cut
  
  

  # split parameters again into sys-sens couplings and sensor parameters:
  phase_sensor['sys_sens_couplings']=theta[:num_sys_sens]
  phase_sensor['sens_couplings']=theta[num_sys_sens:]

  if optimizer is not None:
    phase_sensor['opt_state']=opt_state # record the final opt_state

  return phase_sensor, training_history


def prepare_cluster_Ising_phase_sensor(N_sys,N_sens):
  N=N_sys+N_sens
  paulis=produce_XYZ(N)
  X,Y,Z=paulis

  # prepare Hamiltonian terms for cluster Ising
  H_sys_terms=get_H_sys_terms_for_cluster_Ising(N_sys,paulis)

  # the sensor may couple to these two neighboring spins:
  sys_operators=[X[0],Y[0],Z[0],X[1],Y[1],Z[1]]

  # the sensor can couple via all its single-qubit operators:
  sens1=[X[N_sys],Y[N_sys],Z[N_sys]] # all ops for sensor qubit 1
  sens2=[X[N_sys+1],Y[N_sys+1],Z[N_sys+1]]
  sens_operators=sens1 + sens2

  # the sensor itself can have arbitrary couplings for all possible
  # single- and two-qubit terms:
  H_sens_terms=sens_operators + [op1 @ op2 for op1 in sens1 for op2 in sens2]

  one=get_identity(N)

  # the two output 'variables' of the sensor:
  ZA=Z[N_sys]
  ZB=Z[N_sys+1]

  # output terms will be the PROJECTORS onto Z,Z=+1,+1 and
  # Z,Z=+1,-1 and Z,Z=-1,-1
  output_terms=[ 0.25*(one+ZA) @ (one+ZB),
                 0.25*(one+ZA) @ (one-ZB),
                 0.25*(one-ZA) @ (one-ZB)]

  # now get the dictionary that summarizes the sensor:
  phase_sensor=prepare_general_phase_sensing(N_sys,N_sens,None,None,H_sys_terms,
                                  sys_operators,sens_operators,
                                  H_sens_terms,output_terms,paulis)

  return phase_sensor


def visualize_cluster_Ising_phase_sensor_couplings(sys_sens_couplings,sens_couplings,
                                                   scale=1.0,titles=True,show=True,
                                                   savefig=None
                                                   ):
  """
  Visualize all the couplings. Scale is the cutoff for the
  color bar (at -scale...+scale).
  """
  A=sys_sens_couplings
  B=sens_couplings[6:]
  C=sens_couplings[:6]

  fig,ax=plt.subplots(ncols=3,width_ratios=[3,.5,1.5])
  img=ax[0].matshow(np.reshape(A,(6,6)),vmin=-scale,vmax=+scale,
                    cmap='seismic')
  ax[0].set_xticks(ticks=np.arange(6),labels=[r"$X_1'$",r"$Y_1'$",r"$Z_1'$",r"$X_2'$",r"$Y_2'$",r"$Z_2'$"])
  ax[0].set_yticks(ticks=np.arange(6),labels=[r"$X_1$",r"$Y_1$",r"$Z_1$",r"$X_2$",r"$Y_2$",r"$Z_2$"])
  if titles:
    ax[0].set_title("system-sensor couplings")
  ax[1].matshow(np.reshape(C,(6,1)),vmin=-scale,vmax=+scale,cmap='seismic')
  ax[1].set_xticks(ticks=[],labels=[])
  ax[1].set_yticks(ticks=(0,1,2,3,4,5),labels=(r"$X_1'$",r"$Y_1'$",r"$Z_1'$",r"$X_2'$",r"$Y_2'$",r"$Z_2'$"))
  ax[2].matshow(np.reshape(B,(3,3)),vmin=-scale,vmax=+scale,cmap='seismic')
  ax[2].set_xticks(ticks=(0,1,2),labels=(r"$X_2'$",r"$Y_2'$",r"$Z_2'$"))
  ax[2].set_yticks(ticks=(0,1,2),labels=(r"$X_1'$",r"$Y_1'$",r"$Z_1'$"))
  if titles:
    ax[2].set_title("sensor couplings")
  plt.colorbar(img,ax=ax[2],location='bottom',cmap='seismic')
  fig.tight_layout()
  if savefig is not None:
    fig.savefig(savefig)
  if show:
    plt.show()
    
# define training sampler: uniform sampling of the triangle
def sample_batch_cluster_Ising_phase_sensor(batch_size):
  """
  Returns tuple (x_samples, y_samples), for uniformly sampled
  locations x_samples inside the triangle, and denoting
  each of them with a 2D vector according to the phase it is in.
  """
  gX=np.random.uniform(high=4.0,size=batch_size)
  gZZ=np.random.uniform(high=4.0,size=batch_size)
  gZXZ=4.0-gX-gZZ
  # now flip around all parts that would yield negative gZXZ
  # (stay within triangle)
  neg=gZXZ<0
  gX_new=neg*(4-gZZ) + (1-neg)*gX
  gZZ_new=neg*(4-gX) + (1-neg)*gZZ
  gZXZ_new=4.0-gX_new-gZZ_new

  # the three phases, as they are known analytically:
  middle_boundary=(2*gX_new<4-gZZ_new)
  phaseI=(gZZ_new<2)*middle_boundary
  phaseII=gZZ_new>=2
  phaseIII=(gZZ_new<2)*(1-middle_boundary)

  output=np.stack([phaseI,phaseII,phaseIII],axis=-1)

  return (np.stack([gZXZ_new,gZZ_new,gX_new], axis=-1), output)

def cost_function_mse(y_pred,y_target):
  return np.sum( (y_pred-y_target)**2 )

def grad_cost_function_mse(y_pred,y_target):
  return 2*(y_pred-y_target)


def plot_cluster_Ising_results_inside_phase_triangle(params,values,title="",show=""):
  plt.figure(figsize=(3,2.5))
  plt.scatter(params[:,2]+0.5*params[:,1],params[:,1],c=values,s=5**2)
  plt.axis('off')
  plt.plot([.95,3.05],[2,2],color="black",linewidth=1)
  plt.plot([2,2],[-0.05,2],color="black",linewidth=1)
  plt.text(1.8,-0.6,r"$g_X$")
  plt.text(3.3,2,r"$g_{ZZ}$")
  plt.title(title)
  plt.plot([0,4],[-0.2,-0.2],color="gray",linewidth=1)
  plt.text(-0.1,-0.6,"0")
  plt.text(3.9,-0.6,"4")
  plt.plot([4.2,2.2],[+0,4],color="gray",linewidth=1)
  plt.text(4.2,0.1,"0")
  plt.text(2.4,3.8,"4")
  if show:
    plt.show()
    
# define training sampler: sampling of the triangle at defined locations
def sample_batch_cluster_Ising_phase_sensor_around_corners(batch_size,
                              sampling_locations=np.array([[0.0,0.0,0.2],
                               [4.0,0.0,0.2],
                               [0.0,4.0,0.2]]) ):
  """
  Returns tuple (x_samples, y_samples), with random
  locations x_samples inside the triangle, where
  these are equally sampled from around each of the given
  sampling_locations, each of which denotes
      [gX,gZZ,extent]
  where extent denotes the extent of the sampling region around
  the given location.

  Each of the samples comes with a 3D vector according to the phase it is in.
  (one-hot encoding, for phases I, II, and III)
  """
  # pick the available sampling locations at random:
  loc=np.random.randint(low=0,high=len(sampling_locations),size=batch_size)

  # ascribe gX,gZZ according to the location:
  location=sampling_locations[loc]

  gX=location[:,0] + location[:,2]*np.random.uniform(-0.5,+0.5,size=batch_size)
  gZZ=location[:,1] + location[:,2]*np.random.uniform(-0.5,+0.5,size=batch_size)
  gZXZ=4.0-gX-gZZ

  # the three phases, as they are known analytically:
  middle_boundary=(2*gX<4-gZZ)
  phaseI=(gZZ<2)*middle_boundary
  phaseII=gZZ>=2
  phaseIII=(gZZ<2)*(1-middle_boundary)

  output=np.stack([phaseI,phaseII,phaseIII],axis=-1)

  return (np.stack([gZXZ,gZZ,gX], axis=-1), output)

def get_grad(M,nudge):
    return train_phase_sensor(phase_sensor,None,
                       cost_function_mse,grad_cost_function_mse,
                       learning_rate=0.1,batch_size=batch_size,num_batches=1,
                       test_samples=None,test_step=None,
                       SMALL_NUMBER=1e-8,nonsparse=False,
                      finite_nudging=nudge,shot_noise=M,
                      check_nudging_accuracy=False,
                      symmetric_nudging=True,
                      check_shot_noise_influence=False,
                      record_error_signal=False,
                      return_gradient=True,
                      training_samples=training_samples,
                      no_progress_bar=True)

def try_phase_sensor_training(learning_rate=0.01,finite_nudging=0.1,
                    M_msmt_samples=None,N_sys=8,N_sens=2,
                    n_test_samples=200,batch_size=10,num_batches=1000,
                    test_step=10,optimizer=optax.adam,**kwargs):
# check_nudging_accuracy=False,
#                             symmetric_nudging=False,record_error_signal=False,
#                             check_shot

    # set up phase sensor (Hamiltonian terms etc)    
    phase_sensor = prepare_cluster_Ising_phase_sensor(N_sys,N_sens)
    phase_sensor = initialize_phase_sensor_parameters(phase_sensor,optimizer(learning_rate))

    
    x_test_samples,y_test_samples=sample_batch_cluster_Ising_phase_sensor(n_test_samples)

    phase_sensor,training_history=train_phase_sensor(phase_sensor,
                                sample_batch_cluster_Ising_phase_sensor,
                                cost_function_mse,grad_cost_function_mse,
                                    learning_rate,batch_size,num_batches,
                                test_step=test_step,test_samples=(x_test_samples,y_test_samples),
                                nonsparse=False,
                                finite_nudging=finite_nudging,
                                shot_noise=M_msmt_samples,
                                **kwargs)

    # output to preserve it for later:
    return (phase_sensor['sens_couplings'],phase_sensor['sys_sens_couplings'],
         np.array(training_history['cost']), np.array(training_history['accuracy']),
         np.array(training_history['training_cost'])) , phase_sensor, training_history
  

# shot noise effects:
def try_run_shotnoise(training_seed,M,optimizer=optax.sgd, learning_rate=0.1,
                     num_batches=1000,finite_nudging=0.1):
    np.random.seed(training_seed)
    return try_phase_sensor_training(optimizer=optimizer,
                                                                 test_step=50,
                                                                 finite_nudging=finite_nudging,
                                                                 check_nudging_accuracy=False,
                                                                  check_shot_noise_influence=False,
                                                             num_batches=num_batches,
                                                                  M_msmt_samples=M,
                                                                 symmetric_nudging=True,
                                                                 learning_rate=learning_rate,
                                                                 record_error_signal=False)

  

def get_cluster_Ising_grid_samples(M):
  # sweep through the phase diagram:
  pars=[]
  for jX,gX in enumerate(np.linspace(0,4,M)):
    for jZXZ,gZXZ in enumerate(np.linspace(0,4,M)):
      gZZ=4-gX-gZXZ
      if gZZ>=-1e-5: # only keep positive values of gZZ (following the original papers)
        parameters=(gZXZ,gZZ,gX)
        pars.append(parameters)
  return np.array(pars)

