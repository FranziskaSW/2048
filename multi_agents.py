import numpy as np
import abc
import util
from game import Agent, Action


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """

        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (GameState.py) and returns a number, where higher numbers are better.

        """

        # Useful information you can extract from a GameState (game_state.py)

        successor_game_state = current_game_state.generate_successor(action=action)
        board = successor_game_state.board
        rows = abs(board[:, 1:] - board[:, :-1]).sum(axis=1)
        cols = abs(board[1:, :] - board[:-1, :]).sum(axis=0)
        smoothness = rows.sum() + cols.sum()  # smaller value is better
        empty = sum(board == 0).sum()
        return 10000-smoothness + empty*100


def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.score


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        self.evaluation_function = util.lookup(evaluation_function, globals())
        self.depth = depth

    @abc.abstractmethod
    def get_action(self, game_state):
        return


class MinmaxAgent(MultiAgentSearchAgent):

    def max_value(self, game_state, depth, a_idx=0):
        if depth == 0 or (len(game_state.get_legal_actions(agent_index=a_idx)) == 0):
            return self.evaluation_function(game_state)
        value = -np.inf
        for act in game_state.get_legal_actions(agent_index=a_idx):
            successor = game_state.generate_successor(agent_index=a_idx, action=act)
            value = max(value, self.min_value(successor, depth-1))
        return value

    def min_value(self, game_state, depth, a_idx=1):
        if depth == 0 or (len(game_state.get_legal_actions(agent_index=a_idx)) == 0):
            return self.evaluation_function(game_state)
        value = np.inf
        for act in game_state.get_legal_actions(agent_index=a_idx):
            successor = game_state.generate_successor(agent_index=a_idx, action=act)
            value = min(value, self.max_value(successor, depth-1))
        return value

    def get_action(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1

        Action.STOP:
            The stop direction, which is always legal

        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """
        values = np.zeros(4)
        actions = [0] * 4
        for i, act in enumerate(np.random.permutation(game_state.get_legal_actions(agent_index=0))):
            successor = game_state.generate_successor(agent_index=0, action=act)
            values[i] = self.min_value(game_state=successor, depth=self.depth*2-1)
            actions[i] = act

        act_idx = np.argmax(values)
        action = actions[act_idx]

        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def max_value(self, game_state, alpha, beta, depth, a_idx=0):
        if depth == 0 or (len(game_state.get_legal_actions(agent_index=a_idx)) == 0):
            return self.evaluation_function(game_state)
        v = -np.inf
        for act in game_state.get_legal_actions(agent_index=a_idx):
            successor = game_state.generate_successor(agent_index=a_idx, action=act)
            v = max(v, self.min_value(successor, alpha, beta, depth-1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(self, game_state, alpha, beta, depth, a_idx=1):
        if depth == 0 or (len(game_state.get_legal_actions(agent_index=a_idx)) == 0):
            return self.evaluation_function(game_state)
        v = np.inf
        for act in game_state.get_legal_actions(agent_index=a_idx):
            successor = game_state.generate_successor(agent_index=a_idx, action=act)
            v = min(v, self.max_value(successor, alpha, beta, depth-1))
            if v <= alpha:
                return v
            beta = min(v, beta)
        return v

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        values = np.ones(4) * -np.inf
        actions = [0] * 4
        for i, act in enumerate(np.random.permutation(game_state.get_legal_actions(agent_index=0))):
            successor = game_state.generate_successor(agent_index=0, action=act)
            values[i] = self.min_value(successor, - np.inf, np.inf, self.depth*2-1)
            actions[i] = act

        act_idx = np.argmax(values)
        action = actions[act_idx]

        return action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """
    def max_value(self, game_state, depth, a_idx=0):
        if depth == 0 or (len(game_state.get_legal_actions(agent_index=a_idx)) == 0):
            return self.evaluation_function(game_state)
        value = -np.inf
        for act in game_state.get_legal_actions(agent_index=a_idx):
            successor = game_state.generate_successor(agent_index=a_idx, action=act)
            value = max(value, self.mean_value(successor, depth-1))
        return value

    def mean_value(self, game_state, depth, a_idx=1):
        if depth == 0 or (len(game_state.get_legal_actions(agent_index=a_idx)) == 0):
            return self.evaluation_function(game_state)
        values = []
        for act in game_state.get_legal_actions(agent_index=a_idx):
            successor = game_state.generate_successor(agent_index=a_idx, action=act)
            values.append(self.max_value(successor, depth-1))
        return np.mean(values)

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        values = np.ones(4) * -np.inf
        actions = [0] * 4
        for i, act in enumerate(np.random.permutation(game_state.get_legal_actions(agent_index=0))):
            successor = game_state.generate_successor(agent_index=0, action=act)
            values[i] = self.mean_value(successor, self.depth*2-1)
            actions[i] = act

        act_idx = np.argmax(values)
        action = actions[act_idx]

        return action


def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).
    DESCRIPTION:
    (1) smoothness - we "derive" rows and columns (by subtract an element from its neighbor).
                     we will prefer smaller smoothness which means neighbors elements are close to each other, so that
                     there are more chances they will be combined.
    (2) empty tiles - a board with more empty tiles allows more flexibility of movement and strongly
                      connected to blocking
    (3) merges - we will prefer a state in which there are more potentials merges.
    (4) highest_in_corner - we will prefer a state in which the highest tile is in the corner of the board.

    see the `return` statement to understand how we linearly combined all the above features.
    """
    board = current_game_state.board
    rows = abs(board[:, 1:] - board[:, :-1]).sum(axis=1)
    cols = abs(board[1:, :] - board[:-1, :]).sum(axis=0)
    smoothness = rows.sum() + cols.sum()  # smaller value is better

    empty = sum(board == 0).sum()

    merges = 0

    for i in range(board.shape[0]):
        row_i = board[i, (board != 0)[i,:]]
        if len(row_i) > 1:
            merge_i = sum(np.diff(row_i)==0)

        col_j = board[(board != 0)[:,i], i]
        if len(col_j) > 1:
            merge_j = sum((col_j[1:] - col_j[:-1])==0)

        if (len(row_i) > 1) and (len(col_j) > 1):
            merges += merge_i + merge_j
        elif (len(row_i) > 1) and (len(col_j) <= 1):
            merges += merge_i
        elif (len(row_i) <= 1) and (len(col_j) > 1):
            merges += merge_j

    max_tile = np.log2(current_game_state.max_tile)
    highest_in_corner = any([board[0,0] == max_tile,
                            board[-1, 0] == max_tile,
                            board[0, -1] == max_tile,
                            board[-1, -1] == max_tile])

    return 10000 - smoothness + 100 * empty + 200 * merges + 50 * highest_in_corner

# Abbreviation
better = better_evaluation_function