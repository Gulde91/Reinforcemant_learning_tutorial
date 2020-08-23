
# https://cran.r-project.org/web/packages/ReinforcementLearning/vignettes/ReinforcementLearning.html
library(ReinforcementLearning)

### tictactoe example ----
# The following example utilizes the aforementioned dataset containing 406,541
# game states of Tic-Tac-Toe to learn the optimal actions for each state of the
# board (adapted from Sutton (1998)). All states are observed from the
# perspective of player X who is also assumed to have played first. The player
# who succeeds in placing three of their marks in a horizontal, vertical, or
# diagonal row wins the game. Reward for player X is +1 for 'win', 0 for 'draw',
# and -1 for 'loss'.

# The current state of the board is represented by a rowwise concatenation of
# the players' marks in a 3x3 grid. For example, "..X.B" denotes a board state
# in which player X has placed a mark in the first field of the third column
# whereas player B has placed a mark in the third field of the third column
# (see visualization below).

## ......X.B
## |  .  |  .  |  .   |
## |------------------|
## |  .  |  .  |  .   |
## |------------------|
## |  X  |  .  |   B  |

data("tictactoe")
head(tictactoe, 5)


# Define control object
control <- list(alpha = 0.2, gamma = 0.4, epsilon = 0.1)

# Pass learning parameters to reinforcement learning function
model <- ReinforcementLearning(data = tictactoe,
                               s = "State",
                               a = "Action",
                               r = "Reward",
                               s_new = "NextState",
                               iter = 2,
                               control = control)

# displays the policy that defines the best possible action in each state
head(computePolicy(model))

# Print state-action table
print(model)
View(cbind(row.names(model$Q), model$Q))
# viser den forventede reward ved at sætte X ud fra et givet stadie

# Print summary statistics
summary(model)

# predict
predict(model, "..B.X.X.B")

# retter fejltræk i modellen
## ......X.B
## |  .  |  .  |  B   |
## |------------------|
## |  .  |  X  |  .   |
## |------------------|
## |  X  |  .  |  B   |
# Modellen vil sætte næste træk i c4, men det bør være i c6!

data_ny <- data.frame(State = rep("..B.X.X.B", 100),
                  Action = "c4",
                  NextState = "..BXXBX.B",
                  Reward = -1, stringsAsFactors = FALSE)

model_ny <- ReinforcementLearning(data = data_ny,
                                  s = "State",
                                  a = "Action",
                                  r = "Reward",
                                  s_new = "NextState",
                                  iter = 2,
                                  control = control,
                                  model = model)

predict(model_ny, "..B.X.X.B")
# Nu har modellen lært at sætte næste træk i c6

### GridWorld exampel ----
# Our practical example aims at teaching optimal movements to a robot in a
# grid-shaped maze (adapted from Sutton (1998)). Here the agent must
# navigate from a random starting position to a final position on a
# simulated 2?2 grid (see figure below). Each cell on the grid reflects one
# state, yielding a total of 4 different states. In each state, the agent
# can perform one out of four possible actions, i.e. to move up, down, left,
# or right, with the only restriction being that it must remain on the grid.
# In other words, the grid is surrounded by a wall, which makes it
# impossible for the agent to move off the grid. A wall between s1 and s4
# hinders direct movements between these states. Finally, the reward
# structures is as follows: each movement leads to a negative reward of -1
# in order to penalize routes that are not the shortest path. If the agent
# reaches the goal position, it earns a reward of 10.
#|---------|
#| s1 | s4 |
#| s2   s3 |
#|---------|

# Define state and action sets
states <- c("s1", "s2", "s3", "s4")
actions <- c("up", "down", "left", "right")

# Load built-in environment function for 2x2 gridworld
env <- gridworldEnvironment
print(env)

# Sample N = 1000 random sequences from the environment
data <- sampleExperience(N = 1000,
                         env = env,
                         states = states,
                         actions = actions)
head(data)

# Define reinforcement learning parameters
control <- list(alpha = 0.1, gamma = 0.5, epsilon = 0.1)

# Perform reinforcement learning
model <- ReinforcementLearning(data,
                               s = "State",
                               a = "Action",
                               r = "Reward",
                               s_new = "NextState",
                               control = control)

# display the policy that defines the best possible action in each state
computePolicy(model)

# Alternatively, we can use print(model) in order to write the entire state-action
# table to the screen, i.e. the Q-value of each state-action pair
print(model)

# we see that the total reward in our sample (i.e. the sum of the rewards
# column r) is highly negative. This indicates that the random policy used
# to generate the state transition samples deviates from the optimal case.
# Hence, the next section explains how to apply and update a learned
# policy with new data samples.
summary(model)

## Applying a policy to unseen data ##
# We now apply an existing policy to unseen data in order to evaluate
# the out-of-sample performance of the agent. The following example
# demonstrates how to sample new data points from an existing policy.
# The result yields a column with the best possible action for each given state.

# Example data
data_unseen <- data.frame(State = c("s1", "s2", "s1", "s3"),
                          stringsAsFactors = FALSE)

# Pick optimal action
data_unseen$OptimalAction <- predict(model, data_unseen$State)
data_unseen


## Updating an existing policy ##
# Finally, one can update an existing policy with new observational data.
# This is beneficial when, for instance, additional data points become
# available or when one wants to plot the reward as a function of the
# number of training samples. For this purpose, the ReinforcementLearning()
# function can take an existing rl model as an additional input parameter.
# Moreover, it comes with an additional pre-defined action selection mode,
# namely ??-greedy, thereby following the best action with probability 1?????
# and a random one with ??.
# Sample N = 1000 sequences from the environment
# using epsilon-greedy action selection
data_new <- sampleExperience(N = 1000,
                             env = env,
                             states = states,
                             actions = actions,
                             actionSelection = "epsilon-greedy",
                             model = model,
                             control = control)

# Update the existing policy using new training data
model_new <- ReinforcementLearning(data_new,
                                   s = "State",
                                   a = "Action",
                                   r = "Reward",
                                   s_new = "NextState",
                                   control = control,
                                   model = model)

print(model_new)
plot(model_new)

