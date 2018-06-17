# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util
import random

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        # Loop from start to end
        states = self.mdp.getStates()
        print('all states: ', states)
        for i in range(0, self.iterations):
            print('Iteration: ', i)
            values = self.values.copy()
            for state in states:
                actions = self.mdp.getPossibleActions(state)
                max_total_rewards = float('-inf')
                print('state: ' , state)
                for action in actions:
                    next_states = self.mdp.getTransitionStatesAndProbs(state, action)
                    # print('     action: ', action)
                    # total_rewards = 0
                    # for next_state in next_states:
                    #     reward = self.mdp.getReward(state, action, next_state[0])
                    #     print('         n_state: ', next_state)
                    #     print('         reward: ', reward)
                    #     if self.mdp.isTerminal(next_state[0]):
                    #         total_rewards +=  next_state[1] * reward
                    #     else:
                    #         total_rewards +=  next_state[1] * (reward + self.discount * self.values[next_state[0]])
                    #     print('         total_reward: ', total_rewards)
                    total_rewards = self.computeQValueFromValues(state, action)
                    if max_total_rewards < total_rewards:
                        max_total_rewards = total_rewards
                values[state] = max_total_rewards
            self.values.update(values)
        print(self.values)



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        print('DkS values: ' + str(self.values[state]))
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        next_states = self.mdp.getTransitionStatesAndProbs(state, action)
        total_reward = 0
        print('state: ', state, ' action: ', action)
        for next_state in next_states:
            reward = self.mdp.getReward(state, action, next_state)
            if self.mdp.isTerminal(next_state[0]):
                total_reward += next_state[1] * reward
            else:
                total_reward += next_state[1] * (reward + self.discount * self.getValue(next_state[0]))
        return total_reward


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        possible_actions = self.mdp.getPossibleActions(state)
        if len(possible_actions) <= 0:
            return None
        
        max_q_value = float('-inf')
        selected_action = None

        print('available action: ', possible_actions)

        for action in possible_actions:
            q_value = self.getQValue(state, action)
            print('action: ', action)
            print('     q_value: ', q_value)
            print('     max_q_value: ', max_q_value)
            if max_q_value < q_value:
                max_q_value = q_value
                selected_action = action
                print('assign max_q_value')
        # action = random.choice(possible_actions)
        print('action: ' + selected_action)
        return selected_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
