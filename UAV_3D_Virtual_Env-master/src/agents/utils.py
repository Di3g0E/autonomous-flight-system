import numpy as np

"""
MECHANICAL ENGINEERING POST-GRADUATE PROGRAM
UNIVERSIDADE FEDERAL DO ABC - SANTO ANDRÉ, BRASIL

NOME: RAFAEL COSTA FERNANDES
RA: 21201920754
E−MAIL: COSTA.FERNANDES@UFABC.EDU.BR

DESCRIPTION:
    Generates a T iterations delayed neural network input.
"""

class dl_in_gen():
    
    def __init__(self, T, state_size, action_size):
        # The trained policy expects 18 inputs: state(13) + action(4) + padding(1) OR state(14) + action(4)
        pass 
        self.hist_size = state_size+action_size+1
        self.deep_learning_in_size = self.hist_size*T
        self.reset()
        
    def reset(self):
        self.deep_learning_input = np.zeros(self.deep_learning_in_size)
        
    def dl_input(self, states, actions):
        
        for state, action in zip(states, actions):
            action = np.atleast_1d(action).flatten()
            state = np.atleast_1d(state).flatten()
            
            # Concatenate action and state
            state_t = np.concatenate((action, state))
            
            # Adjust to match self.hist_size (18)
            current_len = len(state_t)
            if current_len < self.hist_size:
                # Pad with zeros if too short (e.g. 17 -> 18)
                padding = np.zeros(self.hist_size - current_len)
                state_t = np.concatenate((state_t, padding))
            elif current_len > self.hist_size:
                # Truncate if too long (just in case)
                state_t = state_t[:self.hist_size]
                
            self.deep_learning_input = np.roll(self.deep_learning_input, -self.hist_size)
            self.deep_learning_input[-self.hist_size:] = state_t
        return self.deep_learning_input