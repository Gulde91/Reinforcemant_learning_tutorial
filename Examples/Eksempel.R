install.packages('MDPtoolbox')
library(MDPtoolbox)

# Model-Based Learning

up <- matrix(c( 1  , 0  , 0  , 0   ,
                0.7, 0.2, 0.1, 0   ,
                0  , 0.1, 0.2, 0.7 ,
                0  ,   0,   0,   1),
             nrow=4, ncol=4, byrow=TRUE)


left <- matrix(c(0.9, 0.1, 0  , 0   , 
                 0.1, 0.9, 0  , 0   , 
                 0  , 0.7, 0.2, 0.1 ,
                 0  ,   0, 0.1, 0.9),
               nrow=4, ncol=4, byrow=TRUE)

down <- matrix(c(0.3, 0.7, 0  , 0   , 
                 0  , 0.9, 0.1, 0   , 
                 0  , 0.1, 0.9, 0   ,
                 0  ,   0, 0.7, 0.3),
               nrow=4, ncol=4, byrow=TRUE)

right <- matrix(c(0.9, 0.1, 0  , 0   ,
                  0.1, 0.2, 0.7, 0   ,
                  0  , 0  , 0.9, 0.1 ,
                  0  ,   0, 0.1, 0.9),
                nrow=4, ncol=4, byrow=TRUE)

# Aggregate previous matrices to create transition probabilities in T
T <- list(up=up, left=left, down=down, right=right)

# Create matrix with rewards
R <- matrix(c(-1, -1, -1, -1, 
              -1, -1, -1, -1, 
              -1, -1, -1, -1,
              10, 10, 10, 10),
            nrow=4, ncol=4, byrow=TRUE)

# Check if this provides a well-defined MDP
mdp_check(T, R)

# Run policy iteration with discount factor γ = 0.9
m <- mdp_policy_iteration(P=T, R=R, discount=0.9)


# Display optimal policy
m$policy

# Display value function
m$V


