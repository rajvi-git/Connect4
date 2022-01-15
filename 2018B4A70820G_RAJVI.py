import numpy as np
import math
import random
import copy

#The Connect4 class represents each game of connect4
class Connect4:
    def __init__(self,player1,player2):
        self.player1 = player1
        self.player2 = player2

        self.board, self.winner, self.player = self.newgame()
    
    #The players are given values 1 and -1 resprectively. 
    #Winner = 0 -> Game has not ended
    #Winner = 1 -> Player1 wins
    #Winner = -1 -> Player2 wins
    #Winner = 3 -> tie
    def newgame(self):
        board = Board()
        winner = 0
        player = 1
        return board,winner,player

    def updateGameBoard(self,move):
        board = self.board.updateBoard(move,self.player)
        return board

    def printGameBoard(self):
        self.board.printBoard()

    def changeTurn(self):
        if self.player == 1:
            self.player = -1
        else:
            self.player = 1

    def findGameWinner(self):
        self.winner=self.board.findWinner()
        return self.winner

#The Board class represents the current state of a board in a game along with the functions that can be performed on it
#It has a numpy array board which represents the values on a board at that instant
class Board:
    #The numpy array board which represents the values on a board at that instant
    #rows filled indicates the number of rows filled in a given column
    def __init__(self) :
        self.board = np.zeros((6,5),dtype=int)
        self.rowsfilled = np.zeros(5,dtype=int)

    def printBoard(self):
        print(self.board)

    #The board is updated after a move is the move is valid
    def updateBoard(self,move,player):
        if move is None or move<0 or move >4:
            return None       
        board = self.board
        row = 5-self.rowsfilled[move]
        if (board[row,move]==0 and row < 6 and row >=0):
            board[row,move]=player
        else:
            return None
        self.board=board
        return board

    #The number of rows filled must be updated after a move
    def updateRowsFilled(self,move):
        self.rowsfilled[move]=self.rowsfilled[move]+1

    #The winner is found by taking squares of size 4X4 and finding the sum of its rows, columns and diagonals.
    # If any of these values are 4(all 4 entries are 1) then player 1 is winner
    # If any of these values are -4(all 4 entries are -1) then player2 is the winner
    #If all the entries of the top row (row0) are 1 or -1, the grid is full with no winner hernce there is a tie
    def findWinner(self):
        for i in range(3):
            for j in range(2):
                square = self.board[i:i+4, j:j+4]
                checkrow =np.sum(square,axis=0)
                checkcol = np.sum(square,axis=1)
                checkdiag1=np.trace(square)
                checkdiag2=np.flip(square,axis=1).trace()
                if np.max([np.max(checkrow),np.max(checkcol),checkdiag2,checkdiag1])==4:
                    winner = 1
                    break
                elif np.min([np.min(checkrow),np.min(checkcol),checkdiag2,checkdiag1])==-4:
                    winner = -1
                    break
                else:
                    winner = 0
        if np.min(np.abs(self.board[0,:])) != 0:
            winner = 3 #tie
        return winner

    def getPossibleMoves(self):
        possibleMoves =[]
        for i in range(5):
            if(self.board[0,i]==0):
                possibleMoves.append(i)
        return possibleMoves
#The node class represents each node of the tree in MCTS
#board passed is an object of class Board
class Node:
    def __init__(self, board, player,id=0,parent=None):
        self.board = board
        self.player = player
        self.child = []
        self.parent = parent
        self.visits = 0
        self.wins = 0
        self.id =id     #index of the node in the mcts tree

#The MCTS class represents the tree in the MCTS algorithm
#board passed is an object
class MCTS:
    def __init__(self, board,player,simulations):
        self.board = board  #object of class Board
        self.simulations =simulations   #no of simulations
        self.player = player
        self.root = Node(board,player,0,None)   #root->represents the starting point of a move
        self.tree = [self.root]     #The tree is an array of nodes
        self.alpha = 3    #tunable parameter -> gives the exploitation coefficient
        self.numNodes = 1   #number of nodes in the tree
        self.hashboard ={hash(board):0}     #dictionary that stores the current state of a board to the index of a node that has that board. used to find the position of the start point in a tree before simulating a new move

    def ucb(self, nodeindex):
        currentNode = self.tree[nodeindex]
        if not currentNode.parent.visits:
            return math.inf
        else:
            exploit = currentNode.wins/(currentNode.visits+1)
            explore = math.log(currentNode.parent.visits)/(currentNode.visits +1)
            ucb = exploit+ self.alpha*math.sqrt(explore)
            return ucb

    def selection(self):
        isTerminal = False
        if hash(self.board) in self.hashboard:
            leaf_node_id = self.hashboard[hash(self.board)]
            self.root=self.tree[leaf_node_id]
        else:
            newroot= Node(self.board,self.player,self.numNodes,None)
            leaf_node_id = self.numNodes
            self.hashboard[hash(self.board)]=self.numNodes
            self.tree.append(newroot)
            self.numNodes+=1
            self.root=self.tree[leaf_node_id]
        while not isTerminal:
            node_id = leaf_node_id
            numChild = len((self.tree[node_id]).child)
            if not numChild:
                leaf_node_id=node_id
                isTerminal=True
            else:
                max_ucb_score = -math.inf
                best_action = leaf_node_id
                for i in range(numChild):
                    action = self.tree[node_id].child[i]
                    child_id = action.id
                    current_ucb = self.ucb(child_id)
                    if current_ucb>max_ucb_score:
                        max_ucb_score = current_ucb
                        best_action = action
                leaf_node_id = action.id
        return leaf_node_id

    def expansion(self, leaf_node):
        current_state = self.tree[leaf_node].board
        player = self.tree[leaf_node].player
        possibleMoves =[]
        for i in range(5):
            if(current_state.board[0,i]==0):
                possibleMoves.append(i)
        winner = current_state.findWinner()
        if(winner==0 and len(possibleMoves)>0):
            children=[]
            for move in possibleMoves:
                new_board=copy.deepcopy(current_state)
                new_board.updateBoard(move,player)
                new_board.updateRowsFilled(move)
                #
                self.hashboard[hash(new_board)]=self.numNodes
                child = Node(new_board,player,self.numNodes,self.tree[leaf_node])
                self.numNodes=self.numNodes+1
                self.tree.append(child)
                children.append(child)
            self.tree[leaf_node].child=children
            chosen=random.choice(children)
            return chosen.id
        return self.tree[leaf_node].id

    def simulation(self, child_node_id):
        current_state = self.tree[child_node_id].board
        prev_player = self.tree[child_node_id].player

        winner = current_state.findWinner()
        isTerminal = False
        if(winner!=0):
            isTerminal=True
        while not isTerminal:
            possibleMoves =[]
            for i in range(5):
                if(current_state.board[0,i]==0):
                    possibleMoves.append(i)
            if len(possibleMoves)==0 :
                winner = 0
                isTerminal=True
            else:
                if prev_player==1:
                    curr_player=-1
                else:
                    curr_player=1
                move = random.choice(possibleMoves)
                new_board=copy.deepcopy(current_state)
                new_board.updateBoard(move,curr_player)
                new_board.updateRowsFilled(move)
                result = new_board.findWinner()
                if result !=0:
                    isTerminal=True
                    winner=curr_player
                    break
            current_state=new_board
            prev_player=curr_player
        return winner  

    def backprop(self,child_node_id,winner):
        player=self.tree[self.root.id].player
        reward=0
        if winner==player:
            reward=1
        elif winner ==-player:
            reward=-1

        self.tree[child_node_id].visits +=1
        self.tree[child_node_id].wins +=reward 

    def play_game(self):
        numSimulations = 0
        while numSimulations<self.simulations:
            node_id = self.selection()
            node_id = self.expansion(node_id)
            winner = self.simulation(node_id)
            self.backprop(node_id,winner)
            numSimulations+=1

        curr_node_id=self.root.id
        action_candidates = self.tree[curr_node_id].child
        maxwins = -math.inf
        bestaction=None
        for action in action_candidates:
            actionwins = action.wins
            if actionwins>=maxwins:
                maxwins=actionwins
                bestaction = action
        return bestaction

def findMove(action,gameBoard):
    if not action:
        return -1
    for i,col in enumerate(action.board.rowsfilled):
        if col==gameBoard.rowsfilled[i]+1:
            return i
    return -1

def simulateGameA(player1,player2):
    game = Connect4(player1,player2)
    game.newgame()
    #game.printGameBoard()
    agent1 = MCTS(game.board,1,200)
    agent2= MCTS(game.board,-1,40)
    while True:
        if game.player==1:
            #print('mcts200')
            agent1.board=copy.deepcopy(game.board)
            agent1.player=1
            action = agent1.play_game()
            move = findMove(action,game.board)
            if(move== -1):
                continue
        else:
            #print('mcts40')
            agent2.board=copy.deepcopy(game.board)
            agent2.player=-1
            agent2= MCTS(game.board,-1,200)
            action=agent2.play_game()
            move = findMove(action,game.board)
            if(move== -1):
                continue    

        if game.updateGameBoard(move) is not None:
            winner = game.findGameWinner()
            game.winner=winner
            if(winner == 3):
                print("Tie")
                game.printGameBoard()
                return None
            elif(winner == 1):
                print("Winner: ", player1)
                game.printGameBoard()
                return player1
            elif(winner == -1):
                print("Winner: ",player2)
                game.printGameBoard()
                return player2
            game.board.updateRowsFilled(move)
            game.changeTurn()
            game.printGameBoard()
        else:
            print("play again")



'''PART B'''

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
                self.q_values[hashval]=np.random.randint(0,5,size = (cols))
                self.rewards[hashval]=0

        if num>=self.epsilon:
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
                elif(checkTerminal==-self.player):
                    self.rewards[hashval]=-10   #lose
                else:
                    self.rewards[hashval]=-5    #tie    
            else:
                self.q_values[hashval]=np.random.randint(0,5,size=(cols))
                self.rewards[hashval]=0
        return new_state

    def update_q_values(self,state,action,new_state):
        R = self.rewards[hash(new_state)]
        q_val_new = np.argmax(self.q_values[hash(new_state)])
        q_val_old = self.q_values[hash(state)][action]
        update = self.alpha*(R + self.gamma*q_val_new - q_val_old)
        self.q_values[hash(state)]=self.q_values[hash(state)]+update

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
                state = next_state


    def train(self):
        #pickle.load(gzip.open("C:\\Documents\\20180820_final.dat.gz", "rb"))

        i=0
        for episode in range(2000):
            game=Connect4('QLearning','MCn')
            agent1=MCTS(game.board,1,4)
            state = game.board
            self.player=-1
            while True:
                if game.player==1:
                    agent1.board = copy.deepcopy(game.board)
                    agent1.player=1
                    action = agent1.play_game()
                    move = findMove(action,game.board)
                    if(move==-1):
                        continue
                else:
                    self.current_state=game.board
                    move = self.chooseAction(game.board)
                    next_state = self.getNextState(state,move,self.player)
                    self.update_q_values(state,move,next_state)
                if game.updateGameBoard(move) is not None:
                    winner = game.findGameWinner()
                    game.board.updateRowsFilled(move)
                    game.changeTurn()
                    if(winner!=0):
                        if(winner!=3):
                            result[i]=winner
                        break
                else:
                    print("play again")
            i+=1
        pickle.dump(self.q_values, gzip.open("C:\\Users\\Jayesh Sampat\\OneDrive\\Documents\\4-1\\AI\\Assignments\\20180820_final.dat.gz", "wb"), protocol=pickle.HIGHEST_PROTOCOL)


def simulateGameB(player1,player2):
    game = Connect4(player1,player2)
    game.newgame()
    agent1 = MCTS(game.board,1,200)
    agent3 = Q_learning_agent(game.board,1,0.9,0.1,0.9)
    agent3.current_state = copy.deepcopy(game.board)
    agent3.player=1
    move = agent3.train()
    while True:
        if game.player==1:
            agent1.board=copy.deepcopy(game.board)
            agent1.player=1
            action = agent1.play_game()
            move = findMove(action,game.board)
            if(move== -1):
                #break
                continue
        else:
            agent3.current_state = copy.deepcopy(game.board)
            agent3.player=1
            move = agent3.chooseAction(game.board)
            if(move== -1):
                continue    

        if game.updateGameBoard(move) is not None:
            winner = game.findGameWinner()
            game.winner=winner
            if(winner == 3):
                print("Tie")
                game.printGameBoard()
                return None
            elif(winner == 1):
                print("Winner: ", player1)
                game.printGameBoard()
                return player1
            elif(winner == -1):
                print("Winner: ",player2)
                game.printGameBoard()
                return player2
            game.board.updateRowsFilled(move)
            game.changeTurn()
            game.printGameBoard()
        else:
            print("play again")

def main():
    A = "MCTS200"
    B = "MCTS40"
    simulateGameA(A,B)
    '''
    #part a 100 games    
    numWins=[0,0,0] #A,B,tie
    for i in range(100):
        winner = simulateGameA(A,B)
        if not winner:
            numWins[2]+=1
        elif winner==A:
            numWins[0]+=1
        elif winner==B:
            numWins[1]+=1
    print(numWins)
    '''
    A="MCTS200"
    B='Qlearning'
    simulateGameB(A,B)
if __name__ == '__main__':
    main()





        



