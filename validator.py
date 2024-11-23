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


    def points_of_interest(self, board, tiles, row, col, horizontal):
        return 

    def validate(self, board, destination):
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
