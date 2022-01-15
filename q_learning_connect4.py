import copy
import random
import connect4_final
import numpy as np

rows = 6
cols = 5
class Q_learning_agent:
    def __init__(self,board,player,alpha,epsilon,gamma) :
        self.current_state = board
        self.player = player
        self.q_values ={}
        self.rewards = {}
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

    def chooseAction(self,state):
        num = np.random.random()
        if num>=self.epsilon:
            hashval=hash(state)
            if hashval not in self.q_values:
                checkTerminal = state.findWinner()
                if(checkTerminal!=0):
                    self.q_values[hashval]=np.zeros(cols)
                    if(checkTerminal==self.player):
                        self.rewards[hashval]=10    #win
                    elif(checkTerminal[hashval]==-self.player):
                        self.rewards[hashval]=-10   #lose
                    else:
                        self.rewards[hashval]=-5    #tie    
                else:
                    self.q_values[hashval]=np.random.randint(0,5)
                    self.rewards[hashval]=0
            action = np.argmax(self.q_values[hash(state)])
        else:
            possibleActions = state.getPossibleMoves()
            action = random.choice(possibleActions)
        return action

    def getNextState(self,state,action,player):
        new_state = copy.deepcopy(state)
        new_state.updateBoard(action,-player)
        new_state.updateRowsFilled(action)
        hashval=hash(new_state)
        if hashval not in self.q_values:
            checkTerminal = new_state.findWinner()
            if(checkTerminal!=0):
                self.q_values[hashval]=np.zeros(cols)
                if(checkTerminal==self.player):
                    self.rewards[hashval]=10    #win
                elif(checkTerminal[hashval]==-self.player):
                    self.rewards[hashval]=-10   #lose
                else:
                    self.rewards[hashval]=-5    #tie    
            else:
                self.q_values[hashval]=np.random.randint(0,5)
                self.rewards[hashval]=0
        return new_state

    def update_q_values(self,state,action,new_state):
        R = self.rewards[hash(new_state)]
        q_val_new = np.argmax(self.q_values[hash(new_state)])
        q_val_old = self.q_values[hash(state)][action]
        update = self.alpha*(R + self.gamma*q_val_new - q_val_old)
        self.q_values[hash(state)]+=update

    def play_game_ql(self):
        for episode in range(100):
            state = self.current_state
            player = self.player
            isTerminal =False
            while not isTerminal:
                if(state.findWinner()!=0):
                    isTerminal=True
                    break
                action = self.chooseAction(state)
                next_state = self.getNextState(state,action,player)
                self.update_q_values(state,action,next_state)
                player = -player
        best_action = np.argmax(self.q_values[hash(self.current_state)])
        return best_action




    

