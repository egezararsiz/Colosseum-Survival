# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
import math
import time
import random
import copy

"""
State class for MCT
"""
class State:
    def __init__(self,board,my_pos,adv_pos,max_move):
        self.board = board
        self.my_pos = my_pos
        self.adv_pos = adv_pos
        self.max_move = max_move
        return

"""
Node class for MCT
"""
class Node:
    def __init__(self,state:State, parent:"Node"):
        self.state = state
        self.valActionState = 0 #this is Q
        self.parent = parent
        self.numsims = 0 #this is n
        self.numsucc = 0
        self.barrier = -1
        self.children = []
        self.player = 1
        return

    def addChild(self,child): 
        self.children.append(child)
        return

    """
    Upper confidence bound
    Help to balance exploitation vs exploration
    """
    def ucb(self): 
        if(self.neverExpanded()): return 1000 
        c = np.sqrt(2)
        q_star = self.valActionState + c * np.sqrt((np.log(self.parent.numsims)/self.numsims))
        return q_star
    
    def updateActionState(self):
        self.valActionState = self.numsucc / self.numsims 
        return
    
    def isLeaf(self):
        return (len(self.children) == 0)

    def maxUcbChild(self):
        maxchild = self.children[-1]
        for child in reversed(self.children):
            if(maxchild.ucb() == 1000): break
            elif (child.ucb() > maxchild.ucb()):
                maxchild = child
        return maxchild
    
    def maxWinRate(self):
        maxchild = self.children[-1]
        for child in reversed(self.children):
            if child.valActionState > maxchild.valActionState:
                maxchild = child
        return maxchild
    
    
    def minUcbChild(self):
        minchild = self.children[-1]
        for child in reversed(self.children):
            if child.ucb() < minchild.ucb():
                minchild = child
        return minchild
    
    def neverExpanded(self):
        return self.numsims == 0
    
    def setWall(self,dir):
        self.barrier = dir
        return

"""
MCT class
Tree structure for Monte Carlo Tree Search
"""
class MCT: 
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
    opposites = {0: 2, 1: 3, 2: 0, 3: 1}
    def __init__(self,node:Node):
        self.current = node     # Current is root.
        self.sims = 0           # Number of full simulations
        return
    
    def search(self,node:Node,firstrun,agent:"StudentAgent"):
        # Time threshold allowance
        if(firstrun):
            t_end = time.time() + 25
        else:
            t_end = time.time() + 1
        
        # Check if this is the first run. Run simulations for 25s (to find a decent first move)
        if (firstrun):
            while(time.time() < t_end):
                # Check if not a leaf, select the max/min depending on player
                if not node.isLeaf():
                    if(node.player == 1):
                        node = node.maxUcbChild()
                    else:
                        node = node.minUcbChild()
                # Leaf node
                else:
                    # Check if leaf never expanded (expand if condition true)
                    if (node.neverExpanded()):
                        res = self.rollout(node)
                        self.backProp(node,res)
                        if(node.parent):
                            node = node.parent
                    # Leaf already expanded
                    else:
                        # Check if game not finished
                        if not(StudentAgent.check_endgame(self,node.state.board, node.state.my_pos, node.state.adv_pos)[0]):
                            # Check if it's our turn
                            if(node.player == 1):
                                # Run alpha-beta pruning to eliminate bad moves and get ok/less worse moves
                                _, my_possible_moves, purgedMoves = agent.alphaBeta(0, True, StudentAgent.MIN_VAL, StudentAgent.MAX_VAL,
                                        node.state.board, node.state.my_pos, node.state.adv_pos, node.state.max_move, time.time())
                                # Check for too many options. We want a random distribution of up to 10 moves to allow for a
                                # better result from simulation. Too many options => too little simulation => less accurate
                                if len(my_possible_moves) > 10:
                                    random.shuffle(my_possible_moves)
                                    my_possible_moves = my_possible_moves[:10]
                                elif len(my_possible_moves) > 0 and len(my_possible_moves) <= 10:
                                    random.shuffle(my_possible_moves)
                                # If alphaBeta prunes all nodes (based on heuristics), use the list of purged moves.
                                # Purged moves might not losing, they just put you in a compromising position (i.e 3 walls around,
                                # or take you further from your opponent)
                                elif len(my_possible_moves) == 0:
                                    my_possible_moves = purgedMoves
                                    # Check if more than 10 possible moves to choose from
                                    if len(my_possible_moves) > 10:
                                        random.shuffle(my_possible_moves)
                                        my_possible_moves = my_possible_moves[:10]
                                    # Less than 10 options, take what you have
                                    elif len(my_possible_moves) > 0 and len(my_possible_moves) <= 10:
                                        random.shuffle(my_possible_moves)
                                    # Last resort, collect all valid moves and randomly select 10
                                    elif len(my_possible_moves) == 0:
                                        my_possible_moves = StudentAgent.posMoves(self,node.state.board,node.state.adv_pos,node.state.my_pos,
                                                                                                        node.state.max_move)
                                    random.shuffle(my_possible_moves)
                                    my_possible_moves = my_possible_moves[:20]
                                # For each good move, create a node and add as child (reverse players)                 
                                for move in my_possible_moves:
                                    updated_board = StudentAgent.set_barrier(self,copy.deepcopy(node.state.board),move[1][0],move[1][1],move[2])
                                    new_state = State(updated_board,(move[1][0],move[1][1]),node.state.adv_pos,self.current.state.max_move)
                                    new_node = Node(new_state,node)
                                    new_node.player = 0
                                    new_node.setWall(move[2])
                                    node.addChild(new_node)
                            # Same as above, but for adversary we do not use alpha-beta pruning since we need to consider all
                            # possible opponent moves and MCTS works better with more randomness
                            else:
                                adv_possible_moves = StudentAgent.posMoves(self,node.state.board,node.state.my_pos,node.state.adv_pos,self.current.state.max_move)
                                # For each move, create a node and add as child (reverse players)
                                for adv in adv_possible_moves:
                                    updated_board = StudentAgent.set_barrier(self,copy.deepcopy(node.state.board),adv[1][0],adv[1][1],adv[2])
                                    new_state = State(updated_board,node.state.my_pos,(adv[1][0],adv[1][1]),self.current.state.max_move)
                                    new_node = Node(new_state,node)
                                    new_node.setWall(adv[2])
                                    node.addChild(new_node)

                            # After adding children, select the last one to rollout (convenient to do so)
                            # Simulate and backpropogate (add relevant values)
                            res = self.rollout(new_node)
                            self.backProp(new_node,res)
                        # If terminal node (game over), backpropogate outcome value to parents until root reached
                        else:
                            game = StudentAgent.check_endgame(self,node.state.board, node.state.my_pos, node.state.adv_pos)
                            my_score = game[1]
                            opp_score = game[2]

                            if my_score > opp_score:
                                self.backProp(node,1)
                            elif my_score < opp_score:
                                self.backProp(node,0)
                            else:
                                self.backProp(node,0.5)
                        
                        # Go back to root see if we have a more promising node along the way
                        node = self.current 
        # Not first move => max 2s per move
        else:
            # Run alpha-beta pruning to eliminate bad moves and get ok/less worse moves
            _, my_possible_moves, purgedMoves = agent.alphaBeta(0, True, StudentAgent.MIN_VAL, StudentAgent.MAX_VAL,
                    node.state.board, node.state.my_pos, node.state.adv_pos, node.state.max_move, time.time())
            
            # Check for too many options. We want a random distribution of up to 10 moves to allow for a
            # better result from simulation. Too many options => too little simulation => less accurate
            if len(my_possible_moves) > 10:
                random.shuffle(my_possible_moves)
                my_possible_moves = my_possible_moves[:10]
            elif len(my_possible_moves) > 0 and len(my_possible_moves) <= 10:
                random.shuffle(my_possible_moves)
            # If alphaBeta prunes all nodes (based on heuristics), use the list of purged moves.
            # Purged moves might not losing, they just put you in a compromising position (i.e 3 walls around,
            # or take you further from your opponent)
            elif len(my_possible_moves) == 0:
                my_possible_moves = purgedMoves
                # Check if more than 10 possibly moves to choose from
                if len(my_possible_moves) > 10:
                    random.shuffle(my_possible_moves)
                    my_possible_moves = my_possible_moves[:10]
                # Less than 10 options, take what you have
                elif len(my_possible_moves) > 0 and len(my_possible_moves) <= 10:
                    random.shuffle(my_possible_moves)
                # Last resort, collect all valid moves
                elif len(my_possible_moves) == 0:
                    my_possible_moves = StudentAgent.posMoves(self,node.state.board,node.state.adv_pos,node.state.my_pos,
                                                                                    node.state.max_move)
                random.shuffle(my_possible_moves)
                my_possible_moves = my_possible_moves[:10]
            
            # Quickly add small subset of moves and hope they are good
            for move in my_possible_moves:
                updated_board = StudentAgent.set_barrier(self,copy.deepcopy(node.state.board),move[1][0],move[1][1],move[2])
                new_state = State(updated_board,(move[1][0],move[1][1]),node.state.adv_pos,self.current.state.max_move)
                new_node = Node(new_state,node)
                new_node.setWall(move[2])   
                node.addChild(new_node)

            # Simulate as many games as possible within time allotment
            while(time.time() < t_end):                     
                new_node = self.current.children[np.random.randint(len(self.current.children))]
                res = self.rollout(new_node)
                self.backProp(new_node,res)

    # Backpropogates the game outcome to all the parents of the leaf 
    def backProp(self,node:Node,res):
        node.numsims += 1
        node.numsucc += res
        node.updateActionState()
        if(node.parent):
            self.backProp(node.parent,res)    
        return

    # Gets the move with the highest win rate after running simulations
    def getBest(self,firstrun,agent):
        self.search(self.current,firstrun,agent)
        best = self.current.maxWinRate()
        return (best.state.my_pos,best.barrier)

    # Simulates a game until completion from the given position (node)
    def rollout(self,node:Node):
        new_board = copy.deepcopy(node.state.board)
        new_pos = node.state.my_pos
        new_adv = node.state.adv_pos 
        max_step = node.state.max_move
        my_turn = node.player

        # Keep applying random moves for both players while the game is not done
        while not (StudentAgent.check_endgame(self,new_board,new_pos,new_adv)[0]):
            # Apply move for our agent
            if(my_turn):
                move = self.randomstep(new_board,new_pos,new_adv,max_step)
                new_board = StudentAgent.set_barrier(self,new_board,move[0][0],move[0][1],move[1])
                new_pos = move[0]
                my_turn -= 1
            # Apply move for the adversary
            else:
                move = self.randomstep(new_board,new_adv,new_pos,max_step)
                new_board = StudentAgent.set_barrier(self,new_board,move[0][0],move[0][1],move[1])
                new_adv = move[0]
                my_turn += 1

        self.sims += 1
        game = StudentAgent.check_endgame(self,new_board, new_pos, new_adv)
        my_score = game[1]
        opp_score = game[2]

        # Return result of simulation
        if(my_score > opp_score):
            return 1
        elif(my_score < opp_score):
            return 0
        else:
            return 0.5
    
    # Random walk function
    def randomstep(self, chess_board, my_pos, adv_pos, max_step):
        # Moves: Up, Right, Down, Left
        ori_pos = copy.deepcopy(my_pos)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        steps = np.random.randint(0, max_step + 1)

        # Random walk
        for _ in range(steps):
            r, c = my_pos
            dir = np.random.randint(0, 4)
            m_r, m_c = moves[dir]
            my_pos = (r + m_r, c + m_c)

            # Special Case enclosed by Adversary
            k = 0
            while chess_board[r, c, dir] or my_pos == adv_pos:
                k += 1
                if k > 300:
                    break
                dir = np.random.randint(0, 4)
                m_r, m_c = moves[dir]
                my_pos = (r + m_r, c + m_c)

            if k > 300:
                my_pos = ori_pos
                break

        # Put Barrier
        dir = np.random.randint(0, 4)
        r, c = my_pos
        while chess_board[r, c, dir]:
            dir = np.random.randint(0, 4)

        return my_pos, dir


@register_agent("student_agent")
class StudentAgent(Agent):
 
    # Constants used
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
    opposites = {0: 2, 1: 3, 2: 0, 3: 1}
    
    # Calculate the Euclidean distance between the agent and adversary
    def evalDist(self, my_pos, adv_pos):
        return math.sqrt(math.pow(my_pos[0] - adv_pos[0], 2) + math.pow(my_pos[1] - adv_pos[1], 2))

    # Check if the step that p1 takes is valid (reachable and within max steps)
    def check_valid_step(self, chess_board, p2, max_step, p1Start, p1End, barrier_dir):
        # Endpoint already has barrier or is boarder
        r, c = p1End
        if chess_board[r, c, barrier_dir]:
            return False
        if np.array_equal(p1Start, p1End):
            return True

        # BFS
        state_queue = [(p1Start, 0)]
        visited = {tuple(p1Start)}
        is_reached = False
        while state_queue and not is_reached:
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos

            if cur_step == max_step:
                break

            for dir, move in enumerate(StudentAgent.moves):
                if chess_board[r, c, dir]:
                    continue

                next_pos = (cur_pos[0] + move[0], cur_pos[1] + move[1])

                if np.array_equal(next_pos, p2) or tuple(next_pos) in visited:
                    continue

                if np.array_equal(next_pos, p1End):
                    is_reached = True
                    break

                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))

        return is_reached

    # General function that is used to generate a list of valid moves for p1
    def posMoves(self, chess_board, p2, p1, max_step):
        n = chess_board.shape[0]
        validMoves = []

        # Loop through k * k area for reachable tiles
        for x in range(-max_step, max_step + 1):
            for y in range(-max_step, max_step + 1):
                # Check if x valid
                if p1[0] + x >= 0 and p1[0] + x < n:
                    # Check if y valid
                    if p1[1] + y >= 0 and p1[1] + y < n:
                        newPos = (p1[0] + x, p1[1] + y)

                        # Check if each move is valid
                        up = StudentAgent.check_valid_step(self, chess_board, p2, max_step, p1, newPos, 0)
                        right = StudentAgent.check_valid_step(self, chess_board, p2, max_step, p1, newPos, 1)
                        down = StudentAgent.check_valid_step(self, chess_board, p2, max_step, p1, newPos, 2)
                        left = StudentAgent.check_valid_step(self, chess_board, p2, max_step, p1, newPos, 3)

                        # Add moves if return is true
                        if up:
                            validMoves.append((p1, newPos, 0))
                        if right:
                            validMoves.append((p1, newPos, 1))
                        if down:
                            validMoves.append((p1, newPos, 2))
                        if left:
                            validMoves.append((p1, newPos, 3))

        return validMoves        

    # Checks to see if the game is over (if true), then computes the score of both players
    def check_endgame(self, chess_board, my_pos, adv_pos):
            board_size = chess_board.shape[0]

            # Union-Find
            father = dict()
            for r in range(board_size):
                for c in range(board_size):
                    father[(r, c)] = (r, c)

            def find(pos):
                if father[pos] != pos:
                    father[pos] = find(father[pos])
                return father[pos]

            def union(pos1, pos2):
                father[pos1] = pos2

            for r in range(board_size):
                for c in range(board_size):
                    for dir, move in enumerate(StudentAgent.moves[1:3]):  # Only check down and right
                        if chess_board[r, c, dir + 1]:
                            continue
                        pos_a = find((r, c))
                        pos_b = find((r + move[0], c + move[1]))
                        if pos_a != pos_b:
                            union(pos_a, pos_b)

            for r in range(board_size):
                for c in range(board_size):
                    find((r, c))

            p0_r = find(tuple(my_pos))
            p1_r = find(tuple(adv_pos))

            p0_score = list(father.values()).count(p0_r)
            p1_score = list(father.values()).count(p1_r)

            if p0_r == p1_r:
                return False, p0_score, p1_score
            
            return True, p0_score, p1_score

    # Set barrier (used for looking ahead with alpha-beta pruning)
    def set_barrier(self, chess_board, r, c, dir):
        # Set the barrier to True
        chess_board[r, c, dir] = True

         # Set the opposite barrier to True
        move = StudentAgent.moves[dir]
        chess_board[r + move[0], c + move[1], StudentAgent.opposites[dir]] = True

        return chess_board

    # Checks if there is 3 walls around the agent 
    # We want to avoid this position to eliminate the possibility the adversary
    # boxing us in
    def checkBox(self, chess_board, r, c):
        count = 0

        # Count the number of walls around the square (r, c)
        if chess_board[r, c, 0] == True:
            count += 1
        if chess_board[r, c, 1] == True:
            count += 1
        if chess_board[r, c, 2] == True:
            count += 1
        if chess_board[r, c, 3] == True:
            count += 1
        
        if count >= 3:
            return True
        
        return False

    # Constants used for alpha-beta pruning
    MAX_VAL = 10000
    MIN_VAL = -10000
    
     # alpha-beta pruning 
    def alphaBeta(self, depth, isAgent, alpha, beta, chess_board, my_pos, adv_pos, max_step, startTime):
        if (depth % 2 == 0):
            isOver, my_score, adv_score = StudentAgent.check_endgame(self, chess_board, my_pos, adv_pos)
        else:
            isOver, adv_score, my_score = StudentAgent.check_endgame(self, chess_board, adv_pos, my_pos) 

        # Check if game over (more useful for endgame moves)
        if isOver:
            # Return 1000 if you win, -1000 if you lose
            if my_score > adv_score:
                return 1, 1000, 0
            else:
                return 1, -1000, 0

         # Hard code search depth (only looks at your move and opponent's move)
        if depth == 2:
            return 0, 0, 0

        # Check if it's the agent's turn
        if isAgent:
            best = StudentAgent.MIN_VAL
            numRemoved = 0
            possibleMoves = StudentAgent.posMoves(self, chess_board, adv_pos, my_pos, max_step)

            # Need to shuffle list of possibleMoves to allow for even distribution of potential nodes
            # in the event that the 0.2s time limit is exceeded 
            random.shuffle(possibleMoves)

            # Variables used in return (list of moves left/pruned after running alpha-beta)
            if depth == 0:
                movesLeft = copy.deepcopy(possibleMoves)
                prunedMoves = []
                startPos = my_pos
            
            # Loop through possible moves for agent
            for i in range(len(possibleMoves)):
                move = (possibleMoves[i][1][0], possibleMoves[i][1][1], possibleMoves[i][2])

                # Update board (set barrier) + update position of agent
                new_board = StudentAgent.set_barrier(self, copy.deepcopy(chess_board), move[0], move[1], move[2])
                my_pos = (move[0], move[1])
                
                # Check if board large (many possible moves), want to prune the number of possible moves
                if chess_board.shape[0] >= 10:
                    dist = StudentAgent.evalDist(self, my_pos, adv_pos)
                    # Check if move puts you further than n/2 distance from opponent
                    if dist > chess_board.shape[0] / 2:
                        prunedMoves.append((startPos, (move[0], move[1]), move[2]))
                        movesLeft.remove((startPos, (move[0], move[1]), move[2])) 
                        numRemoved += 1
                        continue
                
                # Remove move if it threatens to create a box around your postion
                if StudentAgent.checkBox(self, new_board, move[0], move[1]):
                    prunedMoves.append((startPos, (move[0], move[1]), move[2]))
                    movesLeft.remove((startPos, (move[0], move[1]), move[2]))
                    numRemoved += 1
                    continue 

                # Check if we exceeded 0.2s allotment (allow extra time to find more nodes if board large,
                # due to high branching factor)
                if chess_board.shape[0] >= 10:
                    if time.time() > startTime + 0.2:
                        random.shuffle(prunedMoves)
                        return 0, movesLeft[0:((i-1)-numRemoved)], prunedMoves
                # Check if we exceeded 0.1s allotment (allow shorter time to find moves since alpha-beta
                # doesn't take that long)
                else:
                    if time.time() > startTime + 0.1:
                        random.shuffle(prunedMoves)
                        return 0, movesLeft[0:((i-1)-numRemoved)], prunedMoves

                # Pass info to adversary & check their moves
                _, res, _ = StudentAgent.alphaBeta(self, depth + 1, False, alpha, beta, new_board, 
                                             my_pos, adv_pos, max_step, startTime)
                
                # Bad move for us, remove it if we are back at the root level (depth = 0)
                if res == -1000:
                    if depth == 0:
                        movesLeft.remove((startPos, (move[0], move[1]), move[2]))
                        numRemoved += 1
                        continue
                # Ok move, keep searching
                elif res == 0 and res > best:
                    best = res
                    alpha = res
                # Good move, keep going 
                elif res == 1000:
                    best = res
                    alpha = res

                if beta <= alpha:
                    break
            
            # Reached end of search, return all OK moves
            return 0, movesLeft, prunedMoves

        # Adversary's turn
        else:
            best = StudentAgent.MAX_VAL
            possibleMoves = StudentAgent.posMoves(self, chess_board, my_pos, adv_pos, max_step)

            # Loop through possible moves for adversary
            for i in range(len(possibleMoves)):
                move = (possibleMoves[i][1][0], possibleMoves[i][1][1], possibleMoves[i][2])

                # Update board (set barrier) + update position of adversary
                new_board = StudentAgent.set_barrier(self, copy.deepcopy(chess_board), move[0], move[1], move[2])
                adv_pos = (move[0], move[1])

                # Pass info to agent for next move (since we only go to depth 2, get return)
                _, res, _ = StudentAgent.alphaBeta(self, depth + 1, True, alpha, beta, new_board, 
                                             my_pos, adv_pos, max_step, startTime)

                # Check if adversary has winning move, return right away & prune move
                if res == -1000:
                    return 1, res, 1
                # Ok move for opponent, keep searching
                elif res == 0 and res < best:
                    best = res
                    beta = res
                # Good move, keep going
                elif res == 1000 and res < best:
                    best = res
                    beta = res

                if beta <= alpha:
                    break 

            return 1, best, 1

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.autoplay = True
        self.firstrun = True
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    def step(self, chess_board, my_pos, adv_pos, max_step):
        validMoves = StudentAgent.posMoves(self, chess_board, adv_pos, my_pos, max_step)

        # Check through all moves quickly for a win
        for i in range (len(validMoves)):
            move = (validMoves[i][1][0], validMoves[i][1][1], validMoves[i][2])
            new_board = StudentAgent.set_barrier(self, copy.deepcopy(chess_board), move[0], move[1], move[2])
            new_my_pos = (move[0], move[1])

            # Check if move wins game
            isOver, myScore, advScore = StudentAgent.check_endgame(self, new_board, new_my_pos, adv_pos) 
            if isOver:
                if myScore > advScore:
                    return ((move[0], move[1]), move[2])

        cur_state = State(chess_board,my_pos,adv_pos,max_step)
        cur_node = Node(cur_state,None)
        cur_tree = MCT(cur_node)
        x = cur_tree.getBest(self.firstrun,self)

        self.firstrun = False

        return (x[0],x[1])

        
