# all new tiles must be in one line
# this line must be adjacent to (or crossing) some existing tiles (except the first move)
# all newly formed words must exist

from board import Destination

class Validator:

    def __init__(self):
        words = set()
        with open('dictionary.txt', 'r', encoding='utf-8') as file:
            for line in file:
                words.add(line.strip())
        self.dict = words

    def exists(self, word):
        return word.lower() in self.dict


    def points_of_interest(self, board, destination):
        potential = set()
        if destination.horizontal:
            if destination.row > 0:
                potential.update([(destination.row-1, x) for x in range(destination.col, destination.col + len(destination.tiles))])
            if destination.row < 14:
                potential.update([(destination.row+1, x) for x in range(destination.col, destination.col + len(destination.tiles))])
            if destination.col > 0:
                potential.add((destination.row, destination.col-1))
            if destination.col + len(destination.tiles) < 15:
                potential.add((destination.row, destination.col+len(destination.tiles)))
            potential.update([(destination.row, x) for x in range(destination.col, destination.col + len(destination.tiles))])
        else:
            if destination.col > 0:
                potential.update([(y, destination.col-1) for y in range(destination.row, destination.row + len(destination.tiles))])
            if destination.col < 14:
                potential.update([(y, destination.col+1) for y in range(destination.row, destination.row + len(destination.tiles))])
            if destination.row > 0:
                potential.add((destination.row-1, destination.col))
            if destination.row + len(destination.tiles) < 15:
                potential.add((destination.row+len(destination.tiles), destination.col))
            potential.update(
                [(y, destination.col) for y in range(destination.row, destination.row + len(destination.tiles))])

        return [(x, y) for x, y in potential if board.get_tile(x, y) is not None]

    def is_valid(self, board, destination):
        size = len(destination.tiles)
        if destination.horizontal:
            if destination.col + size > 15:
                return False
        else:
            if destination.row + size > 15:
                return False
        word = ''.join(tile.char for tile in destination.tiles)
        if not self.exists(word):
            return False
