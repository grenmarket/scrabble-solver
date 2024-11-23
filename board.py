class IllegalMove(Exception):
    pass


class Tile:
    def __init__(self, tile, char):
        # tile is what is written on the tile, char is the underlying character
        self.tile = tile
        self.char = char

    @staticmethod
    def of(char):
        return Tile(char.upper(), char.upper())

    @staticmethod
    def of_word(word):
        return [Tile.of(ch) for ch in word]

class Destination:

    def __init__(self, tiles, row, col, horizontal):
        self.tiles = tiles
        self.row = row
        self.col = col
        self.horizontal = horizontal

    @staticmethod
    def of(tiles, row, col, horizontal):
        return Destination(tiles, row, col, horizontal)

    def covers(self, row, col):
        if self.horizontal:
            is_col = self.col <= col <= self.col + len(self.tiles)
            return self.row == row and is_col
        else:
            is_row = self.row <= row <= self.row + len(self.tiles)
            return self.col == col and is_row


class Board:

    def __init__(self):
        self.matrix = [[None for i in range(15)] for i in range(15)]
        self.is_empty = True


    def copy(self):
        board = Board()
        board.matrix = self.matrix
        return board

    def get_tile(self, row, col):
        return self.matrix[row][col]

    def place_tile(self, tile, row: int, col: int):
        curr = self.matrix[row][col]
        if curr is None or curr.char == tile.char:
            self.matrix[row][col] = tile
        else:
            raise IllegalMove

    def place_tiles(self, destination):
        size = len(destination.tiles)
        if destination.horizontal:
            if destination.col + size > 15:
                raise IllegalMove
            for i in range(size):
                self.place_tile(destination.tiles[i], destination.row, destination.col + i)
        else:
            if destination.row + size > 15:
                raise IllegalMove
            for i in range(size):
                self.place_tile(destination.tiles[i], destination.row + i, destination.col)
        self.is_empty = False

    def place(self, word, row, col, horizontal):
        tiles = Tile.of_word(word)
        self.place_tiles(Destination.of(tiles, row, col, horizontal))

    def print(self):
        print(' ' + ' _ ' * 15)
        for row in self.matrix:
            row_string = '|' + ''.join('   ' if ch is None else f' {ch.tile} ' for ch in row) + '|'
            print(row_string)
        print(' ' + ' ‾ ' * 15)
