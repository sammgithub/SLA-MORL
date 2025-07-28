# SLA-MORL: Cloud Resource Optimization System 

## Configuration Classes

### 1. **SLAConfig**
**Purpose:** This class manages the configuration of the Service Level Agreements (SLAs) in the cloud environment, defining thresholds for time, cost, throughput, and resource utilization.

- **Attributes:**
  - `time_max`: Maximum training time (in seconds).
  - `cost_max`: Maximum cost (in dollars).
  - `throughput_min`: Minimum throughput (samples per second).
  - `memory_max`: Maximum memory usage (in GB).
  - `gpu_util_min`: Minimum GPU utilization percentage.
  - `cpu_util_min`: Minimum CPU utilization percentage.

- **Functions:**
  - **`__post_init__`**: Initializes default values if not provided. These defaults are set for time, cost, throughput, memory, and GPU/CPU utilization.
  - **`_get_current_gpu_utilization`**: Attempts to fetch the current GPU utilization via `nvidia-smi`.
  - **`forecast_sla_values`**: Forecasts the SLA values (time, cost, throughput) based on baseline history (previous training logs).
  - **`update_from_baseline_history`**: Updates the SLA configuration using the predicted values from the baseline history.

### 2. **ResourceConfig**
**Purpose:** Defines the available resources (e.g., GPU, CPU, memory) and their respective costs for the cloud-based ML training environment.

- **Attributes:**
  - `gpu_count_max`, `gpu_count_min`: Maximum and minimum number of GPUs.
  - `cpu_count_max`, `cpu_count_min`: Maximum and minimum number of CPUs.
  - `memory_max`, `memory_min`: Maximum and minimum memory in GB.
  - `cost_per_gpu_hour`, `cost_per_cpu_hour`, `cost_per_gb_hour`: Costs per hour for GPU, CPU, and memory.

- **Functions:**
  - **`__post_init__`**: Automatically detects the available resources (GPUs, CPUs, memory) and assigns them if they are not provided.

### 3. **RLConfig**
**Purpose:** Contains the configuration for reinforcement learning (RL), defining hyperparameters used for training the RL agent.

- **Attributes:**
  - `hidden_dim`: Hidden dimension for the actor and critic neural networks.
  - `gamma`: Discount factor for future rewards.
  - `tau`: Target network update rate.
  - `lr_actor`, `lr_critic`: Learning rates for the actor and critic.
  - `batch_size`: Batch size used during training.
  - `buffer_size`: Size of the replay buffer.
  - `exploration_noise`: Noise used for exploration in the action space.
  - `train_freq`: Frequency of training updates.
  - `reset_interval`, `reset_probability`: Intervals and probability of resetting the critic network during training.

### 4. **MultiObjectiveConfig**
**Purpose:** Manages the configuration for multi-objective reinforcement learning (MORL), which focuses on optimizing multiple goals (e.g., cost, time, resources) simultaneously.

- **Attributes:**
  - `num_policies`: Number of policies for Pareto front approximation (multi-objective optimization).
  - `preference_vectors`: List of preference vectors that represent different trade-offs between cost, time, and resources.

- **Functions:**
  - **`__post_init__`**: Initializes the preference vectors if they are not provided. These vectors define the priorities for different optimization goals (e.g., cost-focused, time-focused, balanced).

---

## SLA Evaluation Helper

### 5. **`compute_sla_violations`**
**Purpose:** This function calculates SLA violations based on the provided metrics and compares them against predefined SLA thresholds.

- **Arguments:**
  - `metrics_df`: A pandas DataFrame containing training logs.
  - `sla_thresholds`: A dictionary of SLA thresholds for metrics like time, cost, throughput, etc.

- **Returns:**
  - A dictionary where each key is a metric, and the value is `True` if the SLA is met, or `False` if violated.

---

## Offline Data Analysis

### 6. **OfflineDataAnalyzer**
**Purpose:** This class analyzes historical training logs to predict future resource needs based on model and dataset characteristics.

- **Attributes:**
  - `dataframes`: Stores all loaded CSV log files.
  - `combined_data`: Combines all log data for analysis.
  - `model_characteristics`: Stores details about the model and dataset (e.g., parameters, layers, dataset type).

- **Functions:**
  - **`add_log`**: Adds a CSV log file to the analyzer, extracting and storing model characteristics.
  - **`analyze_all_logs`**: Analyzes all logs to compute basic statistics (e.g., average training time, cost) and find the best configurations per model.
  - **`predict_resources`**: Predicts optimal resource allocation for a given model and dataset based on historical logs.
  - **`_default_prediction`**: Provides a default resource prediction using heuristics if no similar models are found.
  - **`update_with_result`**: A placeholder for incorporating new training results (not yet implemented).

---

## CSV Monitor

### 7. **CSVMonitor**
**Purpose:** Monitors CSV log files in real-time, tracking updates and appending reinforcement learning metrics.

- **Attributes:**
  - `csv_path`: Path to the CSV file being monitored.
  - `update_interval`: Interval at which the CSV is checked for updates.
  - `last_modified_time`, `last_row_count`: Track the last modified time and row count to identify new data.
  - `extended_columns`: List of additional columns to track in the CSV, such as RL metrics and SLA compliance.

- **Functions:**
  - **`check_for_updates`**: Checks if the CSV file has been updated since the last check.
  - **`load_latest_data`**: Loads the latest data from the CSV.
  - **`get_new_rows`**: Retrieves any newly added rows since the last check.
  - **`append_rl_metrics`**: Appends RL metrics (e.g., policy used, rewards, SLA compliance) to the CSV file.
  - **`create_extended_csv`**: Creates a new CSV file combining the original data and RL metrics for further analysis.

---

## Training Module Management

### 1. **TrainingModuleManager**
**Purpose:** This class manages the loading, execution, and tracking of external training modules (scripts), allowing for resource management and integration with RL-based optimization.

- **Attributes:**
  - `module`: Stores the loaded training module.
  - `module_path`: Path to the module.
  - `training_process`: Stores the training process (if running).
  - `csv_monitor`: Monitors CSV files for training logs.
  - `output_csv_path`: Path to the output CSV file containing training results.
  - `resource_config`: Holds the resource configuration (like GPUs, CPUs, memory).

- **Functions:**
  - **`load_module`**: Loads a Python training module from a file path and attempts to find a valid training function (e.g., `train`, `train_model`, etc.).
  - **`analyze_module`**: Scans the module's source code to extract details about GPU usage, model type (e.g., ResNet, VGG), and dataset type (e.g., CIFAR, ImageNet).
  - **`run_training`**: Runs the training module with controlled resource allocation (GPU, CPU, batch size). It sets environment variables for resource management and training configuration, and manages the process of executing the training.
  - **`get_model_info`**: Extracts model and dataset information from the CSV file generated during training, such as model name, parameters, dataset type, and other relevant details.
  - **`generate_summary`**: Generates a summary of the training run from the CSV file, capturing key metrics like time, cost, throughput, GPU utilization, etc.

---

## Core MDP Components

### 1. **CloudEnvironmentState**
**Purpose:** Represents the state of the cloud environment, including resource allocation, utilization, training progress, cost, time, and SLA compliance.

- **Attributes:**
  - `resource_config`: Resource configuration (GPUs, CPUs, memory).
  - `sla_config`: SLA configuration (time, cost, throughput, etc.).
  - `state_components`: Dictionary holding various state metrics, including resource utilization, SLA compliance, cost, etc.
  - `state_dim`: The dimension of the state vector (length of `state_components`).

- **Functions:**
  - **`__init__`**: Initializes the state using the provided configuration and sets up the state components.
  - **`update_from_metrics`**: Updates the state from provided metrics (e.g., resource usage, training progress).
  - **`get_vector`**: Converts the state components into a normalized vector format for the RL agent.
  - **`get_sla_status`**: Returns the current SLA compliance status for all defined SLAs (time, cost, throughput, etc.).
  - **`get_overall_sla_compliance`**: Computes the overall SLA compliance rate by averaging the compliance status of all SLAs.

### 2. **ResourceAction**
**Purpose:** Represents the action space for resource allocation. It defines actions that modify the GPU, CPU, and memory allocation.

- **Attributes:**
  - `resource_config`: The resource configuration (limits for GPU, CPU, memory).
  - `gpu_actions`, `cpu_actions`, `memory_actions`: Possible actions for each resource (decrease, no change, increase).
  - `action_dim`: Total number of possible actions (combinations of GPU, CPU, and memory actions).

- **Functions:**
  - **`action_to_changes`**: Converts an action index into resource changes (GPU, CPU, memory).
  - **`compute_new_allocation`**: Computes a new resource allocation based on the action index and current allocation.
  - **`get_action_description`**: Returns a human-readable description of the action (e.g., "Increase GPUs by 1").

### 3. **CloudEnvironment**
**Purpose:** Represents the cloud environment in the context of reinforcement learning (MDP formulation). It manages the state, action space, resource allocation, and rewards.

- **Attributes:**
  - `config`: The configuration dictionary (including resource and SLA configuration).
  - `state`: The current state of the cloud environment (an instance of `CloudEnvironmentState`).
  - `action_space`: The action space for resource allocation (an instance of `ResourceAction`).
  - `state_dim`, `action_dim`: Dimensions of the state and action spaces.
  - `epoch_counter`, `epoch_adjustment_frequency`: Tracks epochs and controls resource adjustment frequency.
  - `current_allocation`: Current resource allocation (GPU, CPU, memory).
  - `metrics_history`: List of historical training metrics.
  - `episode_reward`, `episode_steps`, `episode_start_time`: Tracks reward and steps for the current episode.
  - `user_preference`: User-defined preference for balancing optimization goals (e.g., cost, time, resource usage).

- **Functions:**
  - **`__init__`**: Initializes the environment with the provided configuration and sets up the state and action space.
  - **`reset`**: Resets the environment to the initial state, including resource allocation and metrics history.
  - **`step`**: Takes an action (resource adjustment), updates the state, and returns the next state, reward, done flag, and additional information.
  - **`update_metrics`**: Updates the environment with new metrics from training and updates the state.
  - **`_calculate_reward`**: Calculates the reward for the action taken, considering factors like cost, performance, efficiency, and SLA compliance.
  - **`_check_done`**: Checks if the episode should terminate based on training completion or SLA violations.
  - **`estimate_optimal_resources`**: Estimates optimal resource allocations for different priorities (time, cost, resource efficiency).
  - **`log_adjustment_details`**: Logs detailed information about a resource adjustment, including the action taken, changes in resources, reward, and SLA compliance.
  - **`_get_reward_components`**: Retrieves the components of the reward (cost, performance, efficiency, SLA compliance).

---

## Neural Network Models with Dropout for Uncertainty Estimation

### 1. **Actor**
**Purpose:** The actor network approximates the policy for resource allocation, using dropout layers to estimate uncertainty in the actions.

- **Attributes:**
  - `net`: A sequential neural network consisting of two hidden layers with ReLU activations and dropout.
  - `dropout_rate`: The dropout rate applied after each hidden layer.
  - `enable_dropout`: A flag to enable or disable dropout during inference (evaluation mode).

- **Functions:**
  - **`__init__`**: Initializes the actor network with the specified state dimension, action dimension, hidden layer size, and dropout rate.
  - **`forward`**: Defines the forward pass for the actor. If dropout is enabled, it trains with dropout; otherwise, it uses evaluation mode (no dropout) and outputs the action probabilities via a softmax function.

### 2. **Critic**
**Purpose:** The critic network approximates the value function (Q-value) for a given state-action pair, also using dropout for uncertainty estimation.

- **Attributes:**
  - `l1`, `l2`, `l3`: Fully connected layers for the Q-value estimation.
  - `drop1`, `drop2`: Dropout layers applied after the first and second hidden layers.
  - `dropout_rate`: The dropout rate applied after each hidden layer.
  - `enable_dropout`: A flag to enable or disable dropout during inference.

- **Functions:**
  - **`__init__`**: Initializes the critic network with the specified state dimension, action dimension, hidden layer size, and dropout rate.
  - **`forward`**: Defines the forward pass for the critic. It concatenates state and action inputs, applies ReLU activations, and computes the Q-value, using dropout during training and evaluation during inference.

---

## Simple Policy Gradient Agent

### 3. **SimplePolicy**
**Purpose:** Implements a simple policy gradient agent for resource optimization, using a fully connected neural network to approximate the policy.

- **Attributes:**
  - `config`: Configuration dictionary, including RL parameters.
  - `state_dim`, `action_dim`: Dimensions of the state and action spaces.
  - `device`: The device to use for computation (GPU or CPU).
  - `policy_network`: The neural network representing the policy.
  - `optimizer`: The optimizer (Adam) used to train the policy network.
  - `states`, `actions`, `rewards`, `log_probs`: Buffers to store experiences for training.
  - `train_steps`: Number of training steps taken.

- **Functions:**
  - **`__init__`**: Initializes the policy network and the optimizer, and sets up experience buffers for training.
  - **`select_action`**: Selects an action based on the policy network's output. It includes exploration via noise addition or a random action.
  - **`add_experience`**: Adds experience to the buffer for batch training, also storing the log probabilities of the actions taken.
  - **`train`**: Trains the policy network using the collected experiences, performing the policy update with the REINFORCE algorithm.
  - **`save`**: Saves the trained policy network and optimizer state to a specified path.
  - **`load`**: Loads the trained policy network and optimizer state from a specified path.
  - **`reset`**: Placeholder for resetting the agent (not yet implemented).

---

## Multi-Objective Reinforcement Learning Agent

### 4. **MultiObjectiveRLAgent**
**Purpose:** Implements a multi-objective RL agent with multiple policies for different objectives (e.g., cost, time, resource efficiency).

- **Attributes:**
  - `config`: Configuration dictionary for the agent.
  - `state_dim`, `action_dim`: Dimensions of the state and action spaces.
  - `device`: The device to use for computation (GPU or CPU).
  - `rl_config`: RL configuration (including learning rates, gamma, etc.).
  - `multi_objective_config`: Configuration for multi-objective optimization (e.g., number of policies).
  - `actors`, `critics`: Lists of actor and critic networks, one for each policy.
  - `target_actors`, `target_critics`: Target networks for each policy.
  - `actor_optimizers`, `critic_optimizers`: Optimizers for the actor and critic networks.
  - `replay_buffers`: Experience replay buffers for each policy.
  - `train_steps`: Number of training steps taken.
  - `reset_counters`: Counters for the number of adaptive resets.

- **Functions:**
  - **`__init__`**: Initializes the multi-objective RL agent with multiple policies for different objectives, including the creation of actor-critic pairs and optimizers.
  - **`select_policy`**: Selects the policy to use based on the SLA status and user preference (e.g., cost-focused, time-focused).
  - **`select_action`**: Selects an action using the chosen policy.
  - **`add_experience`**: Adds experience to the replay buffer for a specified policy.
  - **`train`**: Trains the specified policy using the replay buffer and updates both the actor and critic networks.
  - **`check_adaptive_reset`**: Checks whether the critic network should be reset based on SLA violations, adapting to changing conditions.
  - **`save`**: Saves all policies to disk.
  - **`load`**: Loads all policies from disk.

---

## Replay Buffer

### 5. **ReplayBuffer**
**Purpose:** Stores transitions (state, action, reward, next state, done) to facilitate experience replay during training.

- **Attributes:**
  - `max_size`: The maximum size of the buffer.
  - `ptr`, `size`: Pointers and size to manage the buffer.
  - `state`, `action`, `reward`, `next_state`, `done`: Arrays storing the transitions.
  - `device`: The device to use for storing and sampling data (GPU or CPU).

- **Functions:**
  - **`__init__`**: Initializes the replay buffer with specified maximum size and state/action dimensions.
  - **`add`**: Adds a new transition to the buffer.
  - **`sample`**: Samples a batch of transitions from the buffer for training.
  - **`__len__`**: Returns the current size of the buffer.

---

## RL Metrics Tracker

### 6. **RLMetricsTracker**
**Purpose:** Tracks and stores metrics during RL optimization for later plotting and analysis.

- **Attributes:**
  - `output_dir`: Directory where metrics will be saved.
  - DataFrames to store various metrics, including `allocation_history`, `reward_components`, `sla_metrics`, `pareto_front`, and `performance_metrics`.

- **Functions:**
  - **`__init__`**: Initializes the tracker and sets up DataFrames for storing metrics.
  - `**add_adjustment_metrics**

Adds resource adjustment metrics to the history (e.g., action, allocation changes, reward).
  - **`add_performance_metrics`**: Adds general training performance metrics (e.g., throughput, cost).
  - **`add_pareto_front`**: Adds the current Pareto front (optimization options) to the history.
  - **`save_all_metrics`**: Saves all metrics to CSV files for later analysis.

---

## CloudResourceOptimizer Class

### 1. **Purpose:**
This class manages the cloud resource optimization process using reinforcement learning (RL). It includes configuration handling, training management, environment setup, and metrics tracking for optimizing cloud resources such as GPU, CPU, and memory.

### 2. **Attributes:**
- `config`: A configuration dictionary, either loaded from a file or passed directly, that includes resource, SLA, RL, and multi-objective configurations.
- `offline_analyzer`: An instance of `OfflineDataAnalyzer` used to analyze offline logs for further optimization insights.
- `training_manager`: Manages the training module, including loading and running training scripts.
- `cloud_env`: The cloud environment where resource allocation will be optimized (e.g., training model).
- `agent`: The reinforcement learning agent used to optimize resource allocation.
- `user_preference`: User-defined preference for optimization strategies (e.g., balanced, cost, time, or resource efficiency).
- `rl_metrics`: A DataFrame to store metrics related to RL optimization.
- `metrics_tracker`: An instance of `RLMetricsTracker` used to track and store all metrics during training.

### 3. **Functions:**
- **`__init__`**: Initializes the optimizer with the provided configuration, offline data analyzer, and training module manager.
- **`save_all_metrics`**: Saves all RL metrics collected during training into a CSV file.
- **`validate_gpu_configuration`**: Validates the GPU configuration by checking CUDA availability, visible devices, and actual GPU device count.
- **`monitor_cost_time_efficiency`**: Monitors the cost, time, and efficiency during training by comparing baseline metrics with current metrics. It flags issues like increased cost and time.
- **`set_user_preference`**: Sets the user preference for the optimization strategy (e.g., balanced, cost, time, or resource-focused).
- **`optimize_training`**: Main method to perform cloud resource optimization by:
  - Loading the training module.
  - Analyzing baseline metrics and running multiple baseline samples.
  - Initializing the RL agent and training the model.
  - Running optimized training based on RL results and user preference.
- **`save_optimization_metrics`**: Saves optimization metrics (e.g., user preference, baseline, optimized results) in a CSV format suitable for plotting.
- **`generate_report`**: Generates a detailed report of the optimization process, including baseline and optimized results, SLA compliance, RL decisions, and resource changes.

---

## Key Sections of the `optimize_training` Method:

1. **Baseline Sampling:**
   - Runs multiple baseline samples to estimate resource usage (e.g., GPU, CPU, memory) and training metrics like throughput, cost, and time.
   - Analyzes the performance of the model under different resource configurations.

2. **RL Agent Initialization:**
   - After baseline sampling, the `CloudResourceOptimizer` sets up the cloud environment and RL agent (`SimplePolicy`).
   - The agent learns how to adjust resources dynamically based on the metrics collected during baseline runs.

3. **Optimized Training:**
   - Uses the RL agent to make decisions about resource allocation (GPU, CPU, memory) for optimized training.
   - Considers user preferences such as cost, time, or resource utilization, adjusting configurations accordingly.

4. **Optimization Metrics:**
   - Saves and tracks metrics, including resource usage, cost, time, and throughput, during the optimized training run.
   - Generates detailed metrics for further analysis and plotting.

5. **Final Report Generation:**
   - Generates a comprehensive report summarizing the optimization process, improvements in performance (time, cost, throughput), and SLA compliance.
   - Includes a detailed analysis of resource changes, SLA compliance, and optimization options based on RL learning.

---

### Summary of Additional Methods:

- **`save_optimization_metrics`**: This method saves key metrics related to the optimization results (e.g., baseline vs. optimized configurations, cost changes, and SLA compliance).
  
- **`generate_report`**: This method generates a detailed textual report that summarizes the training, baseline, and optimized runs, including performance improvements and SLA compliance.

---

## ResourceManager Class

### 1. **Purpose:**
The `ResourceManager` class handles the application of resource configurations for cloud environments, ensuring that the required GPU, CPU, and batch size settings are correctly applied to the system. It also manages cost-related configurations, making sure they are set for training sessions.

### 2. **Functions:**
- **`apply_configuration`**: A static method that:
  - Configures the GPU settings using `CUDA_VISIBLE_DEVICES` environment variable based on the desired `gpu_count`.
  - Sets CPU thread environment variables (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, etc.).
  - Configures the training batch size by setting the `OPTIMIZER_BATCH_SIZE` environment variable.
  - Applies resource cost settings (e.g., `COST_PER_GPU_HOUR`, `COST_PER_CPU_HOUR`, `COST_PER_GB_HOUR`) based on the `resource_config` dictionary.
  - Returns the applied configuration for verification.

---

## ResourceCostCalculator Class

### 1. **Purpose:**
This helper class calculates the resource costs for cloud training jobs and provides methods for comparing different configurations to validate the tradeoffs between cost, time, and resource usage.

### 2. **Functions:**
- **`__init__`**: Initializes the cost calculator with customizable cost parameters for GPU, CPU, and memory usage.
- **`calculate_hourly_cost`**: Calculates the hourly cost for a given configuration based on the resource count (GPU, CPU, memory) and the per-hour cost rates.
- **`compare_configurations`**: Compares two resource configurations (baseline vs optimized) and provides a detailed analysis of:
  - Hourly cost comparison.
  - Total cost change considering training time.
  - CPU, GPU, and memory changes between the two configurations.
  - Validates whether the cost-time tradeoff is acceptable.
- **`validate_gpu_batch_relationship`**: Ensures that the batch size changes align with GPU count changes. This is important to verify that scaling up or down the GPUs is done sensibly.
- **`diagnose_zero_gpu_utilization`**: Diagnoses potential issues if GPU utilization is zero, even though GPUs are allocated. It checks for issues such as GPU memory usage being zero or potential misconfigurations in the environment.

---

### Key Concepts of `ResourceManager` and `ResourceCostCalculator`:

1. **Resource Allocation:**
   - `ResourceManager` is used to set up the environment for training by configuring GPUs, CPUs, and batch sizes.
   - It ensures that the training job will run with the desired resource configuration, and it logs the details of any changes made to the environment.

2. **Cost Calculation:**
   - `ResourceCostCalculator` helps calculate the total resource cost based on usage (GPU, CPU, memory) and allows for comparisons between configurations. This is particularly useful when optimizing for cost vs performance.

3. **Cost-Time Tradeoffs:**
   - The calculator also validates the tradeoff between cost and time. For example, if increasing resources leads to faster training at a higher cost, the method will determine if the tradeoff is justified by comparing the time savings to the cost increase.

4. **Validation:**
   - `validate_gpu_batch_relationship` ensures that batch size changes are appropriate when changing the number of GPUs (e.g., scaling batch size with the number of GPUs).
   - `diagnose_zero_gpu_utilization` helps detect if GPUs are being underutilized and flags potential configuration issues like GPU memory not being used or high CPU load.

---

## Main Execution Flow in `main()`

1. **Argument Parsing:**
   - The `main()` function starts by parsing the command-line arguments, which specify training file paths, dataset paths, offline logs, configuration files, baseline samples, and user preferences.

2. **Configuration Loading:**
   - The configuration is loaded either from a file or set to default values, which includes resource configurations, SLA constraints, and reinforcement learning (RL) configurations.
   - The SLA settings are overridden by the command-line arguments if provided (e.g., time, cost, throughput limits).

3. **Optimizer Initialization:**
   - The `CloudResourceOptimizer` is initialized with the configuration, and the user preference for optimization (balanced, cost, time, resource) is set.

4. **Optimization Process:**
   - The optimizer runs the `optimize_training` method for each training file provided in the arguments. It generates results and reports for each training session.

5. **Report Generation:**
   - After optimization, a report is generated summarizing the baseline and optimized configurations, performance improvements, SLA compliance, and other optimization metrics.

6. **Results Compilation:**
   - The results are returned as a list, which includes the optimization details for each training file processed.

---

## Summary of Key Methods and Their Responsibilities:

- **`apply_configuration`** in `ResourceManager` applies GPU, CPU, and batch size configurations, and sets cost factors in the environment variables.
- **`calculate_hourly_cost`** in `ResourceCostCalculator` calculates the hourly cost based on resource usage.
- **`compare_configurations`** in `ResourceCostCalculator` compares baseline and optimized configurations, providing a cost and time comparison and validating tradeoffs.
- **`main()`** manages the execution of the cloud resource optimization, including parsing inputs, running optimization, and generating reports.


## Usage Examples

### Time Priority Optimization

```bash
# Minimize time (no specific target)
python main-rl-slamorl.py --training_file train-cifar10-resnet50.py --preference time --epochs 200 --log_dir ./historical_logs/

# Time with specific target (e.g., finish within 30 minutes)
python main-rl-slamorl.py --training_file train-cifar10-resnet50.py --preference time --time_target 30 --epochs 200 --log_dir ./historical_logs/

# Time optimization with skip baseline
python main-rl-slamorl.py --training_file train-cifar10-resnet50.py --preference time --epochs 200 --log_dir ./historical_logs/ --skip_baseline

# Minimize cost (no specific target)
python main-rl-slamorl.py --training_file train-cifar10-resnet50.py --preference cost --epochs 200 --log_dir ./historical_logs/

# Cost with specific target (e.g., stay under $15)
python main-rl-slamorl.py --training_file train-cifar10-resnet50.py --preference cost --cost_target 15 --epochs 200 --log_dir ./historical_logs/

# Cost optimization with skip baseline
python main-rl-slamorl.py --training_file train-cifar10-resnet50.py --preference cost --epochs 200 --log_dir ./historical_logs/ --skip_baseline

# Minimize cost (no specific target)
python main-rl-slamorl.py --training_file train-cifar10-resnet50.py --preference cost --epochs 200 --log_dir ./historical_logs/

# Cost with specific target (e.g., stay under $15)
python main-rl-slamorl.py --training_file train-cifar10-resnet50.py --preference cost --cost_target 15 --epochs 200 --log_dir ./historical_logs/

# Cost optimization with skip baseline
python main-rl-slamorl.py --training_file train-cifar10-resnet50.py --preference cost --epochs 200 --log_dir ./historical_logs/ --skip_baseline


# Balanced with no targets (minimize both)
python main-rl-slamorl.py --training_file train-cifar10-resnet50.py --preference balanced --epochs 200 --log_dir ./historical_logs/

# Balanced with both targets
python main-rl-slamorl.py --training_file train-cifar10-resnet50.py --preference balanced --time_target 30 --cost_target 20 --epochs 200 --log_dir ./historical_logs/

# Balanced with skip baseline
python main-rl-slamorl.py --training_file train-cifar10-resnet50.py --preference balanced --time_target 30 --cost_target 20 --epochs 200 --log_dir ./historical_logs/ --skip_baseline

# Specify output directory
python main-rl-slamorl.py --training_file train-cifar10-resnet50.py --preference time --output_dir ./my_results/ --epochs 200

# Use different training files
python main-rl-slamorl.py --training_file train-resnet50.py --preference cost --epochs 100

# Quick test with fewer epochs
python main-rl-slamorl.py --training_file train-cifar10-resnet50.py --preference balanced --epochs 50 --skip_baseline
