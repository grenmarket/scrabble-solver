class Score:
    def __init__(self):
        self.letter_multipliers = [[1 for i in range(15)] for i in range(15)]
        self.letter_multipliers[0][3] = 2
        self.letter_multipliers[0][11] = 2
        self.letter_multipliers[2][6] = 2
        self.letter_multipliers[2][8] = 2
        self.letter_multipliers[3][0] = 2
        self.letter_multipliers[3][7] = 2
        self.letter_multipliers[3][14] = 2
        self.letter_multipliers[6][2] = 2
        self.letter_multipliers[6][6] = 2
        self.letter_multipliers[6][8] = 2
        self.letter_multipliers[6][12] = 2
        self.letter_multipliers[7][3] = 2
        self.letter_multipliers[7][11] = 2
        self.letter_multipliers[8][2] = 2
        self.letter_multipliers[8][6] = 2
        self.letter_multipliers[8][8] = 2
        self.letter_multipliers[8][12] = 2
        self.letter_multipliers[11][0] = 2
        self.letter_multipliers[11][7] = 2
        self.letter_multipliers[11][14] = 2
        self.letter_multipliers[12][6] = 2
        self.letter_multipliers[12][8] = 2
        self.letter_multipliers[14][3] = 2
        self.letter_multipliers[14][11] = 2
        self.letter_multipliers[1][5] = 3
        self.letter_multipliers[1][9] = 3
        self.letter_multipliers[5][1] = 3
        self.letter_multipliers[5][5] = 3
        self.letter_multipliers[5][9] = 3
        self.letter_multipliers[5][13] = 3
        self.letter_multipliers[9][1] = 3
        self.letter_multipliers[9][5] = 3
        self.letter_multipliers[9][9] = 3
        self.letter_multipliers[9][13] = 3
        self.letter_multipliers[13][5] = 3
        self.letter_multipliers[13][9] = 3
        self.word_multipliers = [[1 for i in range(15)] for i in range(15)]
        self.word_multipliers[0][0] = 3
        self.word_multipliers[0][7] = 3
        self.word_multipliers[0][14] = 3
        self.word_multipliers[7][0] = 3
        self.word_multipliers[7][14] = 3
        self.word_multipliers[14][0] = 3
        self.word_multipliers[14][7] = 3
        self.word_multipliers[14][14] = 3
        self.word_multipliers[1][1] = 2
        self.word_multipliers[1][13] = 2
        self.word_multipliers[2][2] = 2
        self.word_multipliers[2][12] = 2
        self.word_multipliers[3][3] = 2
        self.word_multipliers[3][11] = 2
        self.word_multipliers[4][4] = 2
        self.word_multipliers[4][10] = 2
        self.word_multipliers[7][7] = 2
        self.word_multipliers[10][4] = 2
        self.word_multipliers[10][10] = 2
        self.word_multipliers[11][3] = 2
        self.word_multipliers[11][11] = 2
        self.word_multipliers[12][2] = 2
        self.word_multipliers[12][12] = 2
        self.word_multipliers[13][1] = 2
        self.word_multipliers[13][13] = 2
        self.weights = {
            'A': 1,
            'Ą': 5,
            'B': 3,
            'C': 2,
            'Ć': 6,
            'D': 2,
            'E': 1,
            'Ę': 5,
            'F': 5,
            'G': 3,
            'H': 3,
            'I': 1,
            'J': 3,
            'K': 2,
            'L': 2,
            'Ł': 3,
            'M': 2,
            'N': 1,
            'Ń': 7,
            'O': 1,
            'Ó': 5,
            'P': 2,
            'R': 1,
            'S': 1,
            'Ś': 5,
            'T': 2,
            'U': 3,
            'W': 1,
            'Y': 2,
            'Z': 1,
            'Ź': 9,
            'Ż': 5,
            '.': 0
        }

