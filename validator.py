# all new tiles must be in one line
# this line must be adjacent to (or crossing) some existing tiles (except the first move)
# all newly formed words must exist

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

    def validate(self, board, tiles, row, col, horizontal):
        size = len(tiles)
        if horizontal:
            if col + size > 15:
                return False
        else:
            if row + size > 15:
                return False
        word = ''.join(tile.char for tile in tiles)
        if not self.exists(word):
            return False
