class IllegalMove(Exception):
    pass

class Board:

    def __init__(self):
        self.matrix = [[None for i in range(15)] for i in range(15)]

    def copy(self):
        board = Board()
        board.matrix = self.matrix
        return board

    def place_letter(self, letter, row: int, col: int):
        upper = letter.upper()
        curr = self.matrix[row][col]
        if curr is None or curr == upper:
            self.matrix[row][col] = upper
        else:
            raise IllegalMove

    def place(self, word, row, col, horizontal):
        size = len(word)
        if horizontal:
            if col + size > 15:
                raise IllegalMove
            for i in range(size):
                self.place_letter(word[i], row, col+i)
        else:
            if row + size > 15:
                raise IllegalMove
            for i in range(size):
                self.place_letter(word[i], row+i, col)

    def print(self):
        print(' ' + ' _ ' * 15)
        for row in self.matrix:
            row_string = '|' + ''.join('   ' if ch is None else f' {ch} ' for ch in row) + '|'
            print(row_string)
        print(' ' + ' ‾ ' * 15)
