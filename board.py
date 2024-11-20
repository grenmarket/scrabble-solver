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


class Board:

    def __init__(self):
        self.matrix = [[None for i in range(15)] for i in range(15)]

    def copy(self):
        board = Board()
        board.matrix = self.matrix
        return board

    def place_tile(self, tile, row: int, col: int):
        curr = self.matrix[row][col]
        if curr is None or curr.char == tile.char:
            self.matrix[row][col] = tile
        else:
            raise IllegalMove

    def place_tiles(self, tiles, row, col, horizontal):
        size = len(tiles)
        if horizontal:
            if col + size > 15:
                raise IllegalMove
            for i in range(size):
                self.place_tile(tiles[i], row, col + i)
        else:
            if row + size > 15:
                raise IllegalMove
            for i in range(size):
                self.place_tile(tiles[i], row + i, col)

    def place(self, word, row, col, horizontal):
        tiles = [Tile.of(ch) for ch in word]
        self.place_tiles(tiles, row, col, horizontal)

    def print(self):
        print(' ' + ' _ ' * 15)
        for row in self.matrix:
            row_string = '|' + ''.join('   ' if ch is None else f' {ch.tile} ' for ch in row) + '|'
            print(row_string)
        print(' ' + ' ‾ ' * 15)
