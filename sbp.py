import sys
import time
import numpy as np
from numpy._typing import NDArray

# Sliding Brick Puzzle Game

# 2D-Matrix values:
# -1: the goal
# 0: empty (available to move in)
# 1: wall (no move allowed)
# 2: master brick
# >2: all other bricks

class Game:
    def __init__(self):
        self.board_matrix = None

    def get_board_matrix(self) -> NDArray[np.int32]:
        return self.board_matrix

    def set_board_matrix(self, matrix) -> None:
        self.board_matrix = matrix

    def create_matrix(self,h,w) -> None:
        self.board_matrix = np.zeros((h, w), dtype=int)

    def print_matrix(self, matrix) -> None:
        print(f"{matrix.shape[1]}, {matrix.shape[0]},")
        for row in matrix:
            print(", ".join(row.astype(str)) + ",")

    # Imports a matrix from the file provided
    def import_matrix(self, filename) -> None:
        with open(filename, 'r') as f:
            lines = f.readlines()

        dim = lines[0].strip().rstrip(',')
        cols, rows = map(int, dim.split(','))

        self.create_matrix(rows,cols)

        for r in range(rows):
            line = lines[r + 1].strip().rstrip(',')
            values = list(map(int, line.split(',')))
            self.board_matrix[r] = values

    # Returns the deep clone of the original matrix
    def get_clone_matrix(self) -> NDArray[np.int32]:
        return self.board_matrix.copy()

    # Iterates over the matrix,
    # Returns False is "-1" found or True otherwise
    def is_puzzle_solved(self) -> bool:
        return not np.any(self.board_matrix == -1)

    # Returns all coordinates for the brick number
    def get_brick_coords(self, brick_num) -> list[tuple[int,int]]:
        brick_coords = []  # all coordinates for brickNum
        for i in range(self.board_matrix.shape[0]):
            for j in range(self.board_matrix.shape[1]):
                if self.board_matrix[i][j] == brick_num:
                    brick_coords.append((i, j))
        return brick_coords

    # Returns the array of brick numbers from the state in the order they appeared
    def get_brick_numbers(self) -> list[int]:
        flat = self.board_matrix.reshape(-1)
        seen = set()
        brick_numbers = []
        for x in flat:
            if x > 1 and x not in seen:
                seen.add(x)
                brick_numbers.append(x)
        brick_numbers = np.array(brick_numbers)
        return brick_numbers

    # Calculate all available moves for the matrix
    def get_moves(self) -> list[tuple[int, str]]:
        brick_numbers = self.get_brick_numbers()
        all_moves = []
        for n in brick_numbers:
            moves = self.get_brick_moves(n)
            all_moves.extend(moves)
        return all_moves

    # Calculate available moves for the brick number
    def get_brick_moves(self, brick_num) -> list[tuple[int, str]]:
        # Get all coordinates for the brick
        brick_coords = self.get_brick_coords(brick_num)

        # Check the allowed number of moves per the brick-cell
        up = 0
        right = 0
        down = 0
        left = 0
        for [i,j] in brick_coords:
            # up
            if (self.board_matrix[i-1][j] == 0 or self.board_matrix[i-1][j] == brick_num
                    or (self.board_matrix[i-1][j] == -1 and brick_num == 2)):
                up += 1
            # right
            if (self.board_matrix[i][j+1] == 0 or self.board_matrix[i][j+1] == brick_num
                    or (self.board_matrix[i][j+1] == -1 and brick_num == 2)):
                right += 1
            # down
            if (self.board_matrix[i+1][j] == 0 or self.board_matrix[i+1][j] == brick_num
                    or (self.board_matrix[i+1][j] == -1 and brick_num == 2)):
                down += 1
            # left
            if (self.board_matrix[i][j-1] == 0 or self.board_matrix[i][j-1] == brick_num
                    or (self.board_matrix[i][j-1] == -1 and brick_num == 2)):
                left += 1

        # Compare brick peaces with the allowed number of moves
        num_peaces = len(brick_coords)
        moves = []
        if up == num_peaces: moves.append((brick_num, "up"))
        if right == num_peaces: moves.append((brick_num, "right"))
        if down == num_peaces: moves.append((brick_num, "down"))
        if left == num_peaces: moves.append((brick_num, "left"))
        return moves

    # Finds the user move in the available moves for the brick number
    # Swaps coordinates for bricks if the move is available
    # Returns the cloned matrix or None move not available
    def move_brick(self, brick_num: int, user_move: str) -> None | NDArray[np.int32]:
        if brick_num < 2:
            print(f"Move for {brick_num} is not available.")
            return None

        # Check if the move is available
        brick_moves = self.get_brick_moves(brick_num)
        is_move_available = False
        for [_, move] in brick_moves:
            if user_move == move:
                is_move_available = True
                break

        if not is_move_available:
            print("move not available.")
            return None

        # Sort coordinates based on the move
        brick_coords = self.get_brick_coords(brick_num)
        match user_move:
            case "up":
                # smaller row first
                brick_coords.sort(key=lambda x: x[0])
            case "right":
                # larger col first
                brick_coords.sort(key=lambda x: -x[1])
            case "down":
                # larger row first
                brick_coords.sort(key=lambda x: -x[0])
            case "left":
                # smaller col first
                brick_coords.sort(key=lambda x: x[1])

        # Clone the matrix
        clone_matrix = self.get_clone_matrix()

        # Swap bricks based on the move
        for [i,j] in brick_coords:
            brickCoord = (i, j)
            match user_move:
                case "up":
                    if brick_num == 2 and clone_matrix[i-1][j] == -1:
                        temp = clone_matrix[brickCoord]
                        clone_matrix[brickCoord] = 0
                        clone_matrix[i-1,j] = temp
                    else:
                        self.swap_bricks(clone_matrix, brickCoord, (i-1, j))
                case "right":
                    if brick_num == 2 and clone_matrix[i][j+1] == -1:
                        temp = clone_matrix[brickCoord]
                        clone_matrix[brickCoord] = 0
                        clone_matrix[i,j+1] = temp
                    else:
                        self.swap_bricks(clone_matrix, brickCoord, (i, j+1))
                case "down":
                    if brick_num == 2 and clone_matrix[i+1][j] == -1:
                        temp = clone_matrix[brickCoord]
                        clone_matrix[brickCoord] = 0
                        clone_matrix[i+1,j] = temp
                    else:
                        self.swap_bricks(clone_matrix, brickCoord, (i+1, j))
                case "left":
                    if brick_num == 2 and clone_matrix[i][j-1] == -1:
                        temp = clone_matrix[brickCoord]
                        clone_matrix[brickCoord] = 0
                        clone_matrix[i,j-1] = temp
                    else:
                        self.swap_bricks(clone_matrix, brickCoord, (i, j-1))

        return clone_matrix

    # Mutates the matrix by swapping coordinates for bricks
    def swap_bricks(self, matrix, coord1: (int,int), coord2: (int,int)) -> (NDArray)[np.int32]:
        temp = matrix[coord1]
        matrix[coord1] = matrix[coord2]
        matrix[coord2] = temp
        return matrix

    # Compares an identity of the two states
    def compare_states(self, matrix1, matrix2) -> bool:
        # Check dimensions
        if matrix1.shape != matrix2.shape:
            return False

        # Compare the cells
        rows, cols = matrix1.shape
        for i in range(rows):
            for j in range(cols):
                if matrix1[i][j] != matrix2[i][j]:
                    return False

        return True

    # Normalizes the state for the board matrix
    # Assigns sequential numbers for all bricks with a number > 2
    def normalize_state(self):
        next_idx = 3 # first available number for a brick
        rows,cols = self.board_matrix.shape
        for i in range(rows):
            for j in range(cols):
                if self.board_matrix[i][j] == next_idx:
                    next_idx += 1
                elif self.board_matrix[i][j] > next_idx:
                    self.assign_index(next_idx, self.board_matrix[i][j])
                    next_idx += 1

    # Assigns new brick number for the provided brick number
    # Mutates the board matrix
    def assign_index(self, idx, brick_num):
        rows, cols = self.board_matrix.shape
        for i in range(rows):
            for j in range(cols):
                if self.board_matrix[i][j] == idx:
                    self.board_matrix[i][j] = brick_num
                elif self.board_matrix[i][j] == brick_num:
                    self.board_matrix[i][j] = idx

    # Implements Breadth First Search:
    # Node: a matrix state, Action: move applied,
    # Result: new state (normalized).
    # Stores the visited nodes and parents for the discovered states
    # to restore the path to the solution.
    def breadth_first_search(self):
        node = self.get_board_matrix()
        if self.is_puzzle_solved():
            return [], node, set()

        state_queue = [node]
        visited_states = set()
        visited_states.add(tuple(node.reshape(-1)))
        parent = {}
        actions = {}
        start_key = tuple(node.reshape(-1))

        while len(state_queue) > 0:
            node = state_queue.pop(0)
            self.set_board_matrix(node)

            # Actions
            moves = self.get_moves()
            for [brick_num, move] in moves:
                # Parent node is not changed after moving
                self.set_board_matrix(node)
                state_after_move = self.move_brick(brick_num, move)
                if state_after_move is None:
                    continue

                self.set_board_matrix(state_after_move)
                self.normalize_state()
                norm_state = self.get_board_matrix()

                # Record the visits and how the state was reached
                key = tuple(norm_state.reshape(-1))
                if key in visited_states:
                    continue
                visited_states.add(key)
                parent[key] = tuple(node.reshape(-1))
                actions[key] = (brick_num, move)

                if self.is_puzzle_solved():
                    # Find the path of actions done
                    path = []
                    curr_key = key
                    while curr_key != start_key:
                        path.append(actions[curr_key])
                        curr_key = parent[curr_key]
                    path.reverse()
                    return path, norm_state, visited_states

                state_queue.append(norm_state)

        return None

    # Implements Depth First Search (Preorder):
    # Same as BFS but the stack (LIFO queue) is used,
    # Moves are iterated in the reversed order, so the preorder traversal done.
    def depth_first_search(self):
        node = self.get_board_matrix()
        if self.is_puzzle_solved():
            return [], node, set()

        state_stack = [node]
        visited_states = set()
        visited_states.add(tuple(node.reshape(-1)))
        parent = {}
        actions = {}
        start_key = tuple(node.reshape(-1))

        while len(state_stack) > 0:
            node = state_stack.pop()
            self.set_board_matrix(node)

            moves = self.get_moves()
            for [brick_num, move] in reversed(moves):
                # Parent state persisted
                self.set_board_matrix(node)

                # Child state expanded
                state_after_move = self.move_brick(brick_num, move)
                if state_after_move is None:
                    continue
                self.set_board_matrix(state_after_move)
                self.normalize_state()
                norm_state = self.get_board_matrix()

                # Record the path to the child
                key = tuple(norm_state.reshape(-1))
                if key in visited_states:
                    continue
                visited_states.add(key)
                parent[key] = tuple(node.reshape(-1))
                actions[key] = (brick_num, move)

                # Check if the child is a solution
                if self.is_puzzle_solved():
                    # Find the path of actions done
                    path = []
                    curr_key = key
                    while curr_key != start_key:
                        path.append(actions[curr_key])
                        curr_key = parent[curr_key]
                    path.reverse()
                    return path, norm_state, visited_states

                state_stack.append(norm_state)

        return None

if __name__ == "__main__":
    game = Game()

    if len(sys.argv) == 3 and sys.argv[1] == "print":
        game.import_matrix(sys.argv[2])
        matrix = game.get_board_matrix()
        game.print_matrix(matrix)
    elif len(sys.argv) == 3 and sys.argv[1] == "done":
        game.import_matrix(sys.argv[2])
        print(game.is_puzzle_solved())
    elif len(sys.argv) == 3 and sys.argv[1] == "availableMoves":
        game.import_matrix(sys.argv[2])
        matrix = game.get_board_matrix()
        game.print_matrix(matrix)
        all_moves = game.get_moves()
        for [num, move] in all_moves:
            print(f"({num}, {move})")
    elif len(sys.argv) == 4 and sys.argv[1] == "applyMove":
        game.import_matrix(sys.argv[2])
        matrix = game.get_board_matrix()

        # Parse string
        string = sys.argv[3].strip().strip("()")  # "5,down"
        brick_str, move = string.split(",")

        # Move a brick
        brick_num = int(brick_str)
        matrix_after_move = game.move_brick(brick_num, move)
        if matrix_after_move is None:
            print("Illegal move")
        else:
            game.print_matrix(matrix_after_move)
    elif len(sys.argv) == 4 and sys.argv[1] == "compare":
        game.import_matrix(sys.argv[2])
        matrix1 = np.copy(game.get_board_matrix())
        game.import_matrix(sys.argv[3])
        matrix2 = np.copy(game.get_board_matrix())
        is_identical = game.compare_states(matrix1, matrix2)
        print(is_identical)
    elif len(sys.argv) == 3 and sys.argv[1] == "norm":
        game.import_matrix(sys.argv[2])
        game.normalize_state()
        matrix = game.get_board_matrix()
        game.print_matrix(matrix)
    elif len(sys.argv) == 4 and sys.argv[1] == "random":
        N = int(sys.argv[3])
        game.import_matrix(sys.argv[2])

        for i in range(N):
            # 1. Generate all the moves for the board
            all_moves = game.get_moves()

            # 2. Select a random move
            idx = np.random.randint(len(all_moves))
            brick_num, move = all_moves[idx]
            print(f"({brick_num}, {move})")

            # 3. Execute the selected move
            matrix_after_move = game.move_brick(brick_num, move)
            game.print_matrix(matrix_after_move)

            # 4. Normalize the resulting game state
            game.set_board_matrix(matrix_after_move)
            game.normalize_state()

            if game.is_puzzle_solved():
                print("puzzle is solved.")
                break
    elif len(sys.argv) == 3 and sys.argv[1] == "bfs":
        game.import_matrix(sys.argv[2])
        start_time = time.perf_counter()
        path, state, visits = game.breadth_first_search()
        end_time = time.perf_counter()

        if state is None:
            print("No solution found.")
        else:
            for num, move in path:
                print(f"({num}, {move})")
            game.print_matrix(state)
            print(f"Total search time: {(end_time - start_time):4f}ms.")
            print(f"Total nodes visited: {len(visits)}.")
            print(f"Solution length: {len(path)}.")
    elif len(sys.argv) == 3 and sys.argv[1] == "dfs":
        game.import_matrix(sys.argv[2])
        start_time = time.perf_counter()
        path, state, visits = game.depth_first_search()
        end_time = time.perf_counter()

        if state is None:
            print("No solution found.")
        else:
            for num, move in path:
                print(f"({num}, {move})")
            game.print_matrix(state)
            print(f"Total search time: {(end_time - start_time):4f}ms.")
            print(f"Total nodes visited: {len(visits)}.")
            print(f"Solution length: {len(path)}.")