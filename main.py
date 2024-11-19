from board import Board
from board import Tile
from scores import Score

board = Board()
board.place('masakra', 1, 1, True)
board.place('ameba', 0, 1, False)
board.place('walka', 4, 0, True)
word_with_blank = [Tile('.', 'Ż'), Tile('A', 'A'), Tile('B', 'B'), Tile('A', 'A')]
board.place_tiles(word_with_blank, 3, 4, False)
board.print()