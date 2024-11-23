from validator import Validator
from board import Board, Destination, Tile

validator = Validator()
board = Board()
board.place('żałość', 7, 3, True)
board.place('stary', 5, 4, False)
board.print()

print(validator.points_of_interest(board, Destination.of(Tile.of_word('wyrwie'), 9, 3, True)))
print(validator.points_of_interest(board, Destination.of(Tile.of_word('wyrwie'), 9, 3, False)))
print(validator.points_of_interest(board, Destination.of(Tile.of_word('osocze'), 6, 7, True)))
print(validator.points_of_interest(board, Destination.of(Tile.of_word('las'), 4, 5, False)))
print(validator.points_of_interest(board, Destination.of(Tile.of_word('lololo'), 5, 1, False)))
print(validator.points_of_interest(board, Destination.of(Tile.of_word('lololo'), 3, 2, True)))
print(validator.points_of_interest(board, Destination.of(Tile.of_word('lololo'), 4, 10, False)))
