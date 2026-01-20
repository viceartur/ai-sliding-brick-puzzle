import sys
import numpy as np
from numpy._typing import NDArray
from sympy.strategies.core import switch


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

    def get_board_matrix(self):
        return self.board_matrix

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

    # Calculate all available moves for the matrix
    def get_moves(self) -> list[tuple[int, str]]:
        brick_numbers = np.unique(self.board_matrix)
        brick_numbers = brick_numbers[brick_numbers > 1]
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
            if (self.board_matrix[i-1][j] == 0
                    or self.board_matrix[i-1][j] == -1
                    or self.board_matrix[i-1][j] == brick_num):
                up += 1
            # right
            if (self.board_matrix[i][j+1] == 0
                    or self.board_matrix[i-1][j] == -1
                    or self.board_matrix[i][j+1] == brick_num):
                right += 1
            # down
            if (self.board_matrix[i+1][j] == 0
                    or self.board_matrix[i - 1][j] == -1
                    or self.board_matrix[i+1][j] == brick_num):
                down += 1
            # left
            if (self.board_matrix[i][j-1] == 0
                    or self.board_matrix[i - 1][j] == -1
                    or self.board_matrix[i][j-1] == brick_num):
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
        self.clone_matrix = self.get_clone_matrix()

        # Swap bricks based on the move
        for [i,j] in brick_coords:
            match user_move:
                case "up":
                    self.swap_bricks(self.clone_matrix, (i, j), (i-1, j))
                case "right":
                    self.swap_bricks(self.clone_matrix, (i, j), (i, j+1))
                case "down":
                    self.swap_bricks(self.clone_matrix, (i, j), (i+1, j))
                case "left":
                    self.swap_bricks(self.clone_matrix, (i, j), (i, j-1))

        return self.clone_matrix

    # Mutates the matrix by swapping coordinates for bricks
    def swap_bricks(self, matrix, coord1: (int,int), coord2: (int,int))\
            ->(NDArray)[np.int32]:
        temp = matrix[coord1]
        matrix[coord1] = matrix[coord2]
        matrix[coord2] = temp
        return matrix

    # Compares an identity of the two states
    def compare_states(self, filename1: str, filename2: str) -> bool:
        self.import_matrix(filename1)
        matrix1 = np.copy(self.board_matrix)
        self.import_matrix(filename2)
        matrix2 = np.copy(self.board_matrix)

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
        isIdentical = game.compare_states(sys.argv[2], sys.argv[3])
        print(isIdentical)

