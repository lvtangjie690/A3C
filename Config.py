
class Config:

    #########################################################################
    # Game configuration
    
    PLAY_MODE = False

    # Enable to train
    TRAIN_MODELS = True
    # Load old models. Throws if the model doesn't exist
    LOAD_CHECKPOINT = False
    # If 0, the latest checkpoint is loaded
    LOAD_EPISODE = 0 

    #########################################################################
    # Algorithm parameters

    # Discount factor
    DISCOUNT = 0.99
    
    # Tmax
    TIME_MAX = 10
    
    # Reward Clipping
    REWARD_MIN = -1
    REWARD_MAX = 1

    # Max size of the queue
    MAX_QUEUE_SIZE = 100
    PREDICTION_BATCH_SIZE = 128

    # Total number of episodes and annealing frequency
    EPISODES = 400000
    ANNEALING_EPISODE_COUNT = 400000

    # Entropy regualrization hyper-parameter
    BETA_START = 0.0001
    BETA_END = 0.0001

    # Learning rate
    LEARNING_RATE_START = 0.0003
    LEARNING_RATE_END = 0.0003

    # Optimizer (Adam or RMSProp)
    OPTIMIZER = 'RMSProp'

    # AdamOptimizer parameters
    ADAM_BETA1 = 0.9
    ADAM_BETA2 = 0.999
    ADAM_EPSILON = 1e-8

    # RMSProp parameters
    RMSPROP_DECAY = 0.99
    RMSPROP_MOMENTUM = 0.0
    RMSPROP_EPSILON = 0.1
    
    # Gradient clipping
    USE_GRAD_CLIP = True
    GRAD_CLIP_NORM = 40.0 

    #########################################################################
    # Log and save

    # Enable TensorBoard
    TENSORBOARD = False
    # Update TensorBoard every X training steps
    TENSORBOARD_UPDATE_FREQUENCY = 1000

    # Enable to save models every SAVE_FREQUENCY episodes
    SAVE_MODELS = True
    # Save every SAVE_FREQUENCY episodes
    SAVE_FREQUENCY = 1000
    
    # Print stats every PRINT_STATS_FREQUENCY episodes
    PRINT_STATS_FREQUENCY = 50
    # The window to average stats
    STAT_ROLLING_MEAN_WINDOW = 100

    # Results filename
    RESULTS_FILENAME = 'results.txt'
    # Network checkpoint name
    NETWORK_NAME = 'network'

    #########################################################################
    # More experimental parameters here
    
    # Minimum policy
    MIN_POLICY = 0.01

    # Learner's address
    MASTER_IP = 'localhost'
    MASTER_PORT = 8000
    
    # Worker's base port
    WORKER_BASE_PORT = 9000

    # Number of Agents
    AGENTS = 16

    # Number of Predictors
    PREDICTORS = 2

    # Number of Trainers
    TRAINERS = 1

    # Game name
    # GAME_NAME = 'PointGame'
    GAME_NAME = 'GymGame'

    # Game Push Alogorithm
    GAME_PUSH_ALGORITHM = False

    # DEVICE
    DEVICE = '/gpu:0' 
