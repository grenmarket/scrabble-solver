import string
from collections import namedtuple
from typing import List, Set, Tuple, Optional

Move = namedtuple('Move', ['row', 'col', 'direction', 'word', 'score'])

class Tile:
    def __init__(self, letter: str, is_blank: bool = False):
        self.letter = letter.upper() if letter else None
        self.is_blank = is_blank

    def __repr__(self):
        return f"Tile('{self.letter}', {self.is_blank})"

class BoardSquare:
    def __init__(self, tile: Optional[Tile] = None, premium: str = None):
        self.tile = tile
        self.premium = premium  # 'DL', 'TL', 'DW', 'TW', or None

    def __repr__(self):
        return f"BoardSquare({self.tile}, '{self.premium}')"

class GADDAGNode:
    def __init__(self):
        self.arcs = {}  # Maps characters to other nodes
        self.is_terminal = False  # Indicates if a valid word ends here

class GADDAG:
    def __init__(self):
        self.root = GADDAGNode()
        self.delimiter = '>'  # Separator character

    def add_word(self, word: str) -> None:
        """Add a word to the GADDAG structure."""
        word = word.upper()
        # For each position in the word
        for i in range(len(word)):
            # Create the reversed prefix + delimiter + suffix
            gaddag_str = word[i::-1] + self.delimiter + word[i + 1:]

            # Add this string to the GADDAG
            node = self.root
            for char in gaddag_str:
                if char not in node.arcs:
                    node.arcs[char] = GADDAGNode()
                node = node.arcs[char]
            node.is_terminal = True

    def is_valid_word(self, word: str) -> bool:
        """Check if a word exists in the dictionary."""
        word = word.upper()
        # Check each way of breaking the word
        for i in range(len(word)):
            # Form the GADDAG string
            gaddag_str = word[i::-1] + self.delimiter + word[i + 1:]

            # Check if this string exists in the GADDAG
            node = self.root
            valid = True
            for char in gaddag_str:
                if char not in node.arcs:
                    valid = False
                    break
                node = node.arcs[char]

            if valid and node.is_terminal:
                return True

        return False

    def build_from_dictionary(self, dictionary_path: str) -> None:
        with open(dictionary_path, 'r') as file:
            for line in file:
                word = line.strip()
                if word and all(c.isalpha() for c in word):
                    self.add_word(word)

class ScrabbleBoard:
    def __init__(self, size: int = 15):
        self.size = size
        self.board = [[BoardSquare() for _ in range(size)] for _ in range(size)]
        self._set_premium_squares()
        self.gaddag = GADDAG()

    def load(self, dict_path: str = './dictionary.txt'):
        self.gaddag.build_from_dictionary(dict_path)

    def _set_premium_squares(self) -> None:
        """Set the premium squares on the Scrabble board.

        Sets up all premium squares on a standard 15x15 Scrabble board:
        - DL: Double Letter Score
        - TL: Triple Letter Score
        - DW: Double Word Score
        - TW: Triple Word Score
        """
        # Double letter scores
        dl_positions = [
            (0, 3), (0, 11),
            (2, 6), (2, 8),
            (3, 0), (3, 7), (3, 14),
            (6, 2), (6, 6), (6, 8), (6, 12),
            (7, 3), (7, 11),
            (8, 2), (8, 6), (8, 8), (8, 12),
            (11, 0), (11, 7), (11, 14),
            (12, 6), (12, 8),
            (14, 3), (14, 11)
        ]

        # Triple letter scores
        tl_positions = [
            (1, 5), (1, 9),
            (5, 1), (5, 5), (5, 9), (5, 13),
            (9, 1), (9, 5), (9, 9), (9, 13),
            (13, 5), (13, 9)
        ]

        # Double word scores
        dw_positions = [
            (1, 1), (1, 13),
            (2, 2), (2, 12),
            (3, 3), (3, 11),
            (4, 4), (4, 10),
            (7, 7),  # Center square
            (10, 4), (10, 10),
            (11, 3), (11, 11),
            (12, 2), (12, 12),
            (13, 1), (13, 13)
        ]

        # Triple word scores
        tw_positions = [
            (0, 0), (0, 7), (0, 14),
            (7, 0), (7, 14),
            (14, 0), (14, 7), (14, 14)
        ]

        # Set premium values on the board
        for row, col in dl_positions:
            self.board[row][col].premium = 'DL'

        for row, col in tl_positions:
            self.board[row][col].premium = 'TL'

        for row, col in dw_positions:
            self.board[row][col].premium = 'DW'

        for row, col in tw_positions:
            self.board[row][col].premium = 'TW'

    def _is_empty(self) -> bool:
        """Check if the board is empty."""
        for row in self.board:
            for square in row:
                if square.tile is not None:
                    return False
        return True

    def _get_all_anchors(self) -> List[Tuple[int, int]]:
        anchors = []
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r][c].tile is None:  # Empty square
                    # Check if adjacent to a tile
                    has_neighbor = False
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < self.size and 0 <= nc < self.size and
                                self.board[nr][nc].tile is not None):
                            has_neighbor = True
                            break

                    if has_neighbor:
                        anchors.append((r, c))

        # If board is empty, center square is the only anchor
        if self._is_empty():
            anchors.append((self.size // 2, self.size // 2))

        return anchors

    def _get_cross_checks(self, row: int, col: int, is_horizontal: bool) -> Set[str]:
        """
        Determine which letters can be legally placed at (row, col)
        considering perpendicular words.
        """
        if self.board[row][col].tile is not None:
            return set()  # Square already occupied

        # Direction to check for cross words
        dr, dc = (0, 1) if is_horizontal else (1, 0)
        cross_dr, cross_dc = (1, 0) if is_horizontal else (0, 1)

        # If no tiles in the perpendicular direction, all letters are valid
        has_perpendicular = False
        r, c = row - cross_dr, col - cross_dc
        while 0 <= r < self.size and 0 <= c < self.size and self.board[r][c].tile is not None:
            has_perpendicular = True
            r -= cross_dr
            c -= cross_dc

        r, c = row + cross_dr, col + cross_dc
        while 0 <= r < self.size and 0 <= c < self.size and self.board[r][c].tile is not None:
            has_perpendicular = True
            r += cross_dr
            c += cross_dc

        if not has_perpendicular:
            return set(string.ascii_uppercase)  # All letters valid

        # Check each letter
        valid_letters = set()
        for letter in string.ascii_uppercase:
            # Build the cross word
            cross_word = []

            # Collect letters before
            r, c = row - cross_dr, col - cross_dc
            prefix = []
            while 0 <= r < self.size and 0 <= c < self.size and self.board[r][c].tile is not None:
                prefix.append(self.board[r][c].tile.letter)
                r -= cross_dr
                c -= cross_dc
            cross_word = prefix[::-1]  # Reverse the prefix

            # Add the current letter
            cross_word.append(letter)

            # Collect letters after
            r, c = row + cross_dr, col + cross_dc
            while 0 <= r < self.size and 0 <= c < self.size and self.board[r][c].tile is not None:
                cross_word.append(self.board[r][c].tile.letter)
                r += cross_dr
                c += cross_dc

            # Check if this forms a valid word
            cross_word_str = ''.join(cross_word)
            if len(cross_word) > 1 and self.gaddag.is_valid_word(cross_word_str):
                valid_letters.add(letter)
            elif len(cross_word) == 1 and self.gaddag.is_valid_word(letter):
                # Single letter must be a valid word by itself
                valid_letters.add(letter)

        return valid_letters

    def _find_moves_from_anchor(self, row: int, col: int, rack: List[Tile], is_horizontal: bool = True) -> List[Move]:
        """Find all legal moves that include the anchor square."""
        moves = []

        # Skip if square is already occupied
        if self.board[row][col].tile is not None:
            return moves

        # Direction vectors
        dr, dc = (0, 1) if is_horizontal else (1, 0)

        # Convert rack tiles to letters for easier processing
        rack_letters = []
        for tile in rack:
            if tile.is_blank:
                rack_letters.append('')  # Empty string indicates a blank tile
            else:
                rack_letters.append(tile.letter)

        # Get cross-checks for the anchor position
        cross_checks = self._get_cross_checks(row, col, is_horizontal)
        if not cross_checks:  # No valid letters can be placed here
            return moves

        # Find the prefix (tiles already on board before the anchor)
        prefix = []
        r, c = row, col
        while True:
            r -= dr
            c -= dc
            if 0 <= r < self.size and 0 <= c < self.size and self.board[r][c].tile is not None:
                prefix.append(self.board[r][c].tile.letter)
            else:
                break
        prefix = prefix[::-1]  # Reverse to get correct order

        # Function to extend right (forward) from a position
        def extend_right(node, path, pos, remaining_rack, used_blanks=None):
            if used_blanks is None:
                used_blanks = []  # Track positions where blanks are used

            r, c = pos

            # Check if we're still on the board
            if r < 0 or r >= self.size or c < 0 or c >= self.size:
                return

            # Get cross-checks for the current position (not just using anchor's cross-checks)
            current_cross_checks = self._get_cross_checks(r, c, is_horizontal)
            if not current_cross_checks:  # No valid letters can be placed here
                return

            # If the current square has a tile, incorporate it and continue
            if self.board[r][c].tile is not None:
                letter = self.board[r][c].tile.letter
                if letter in node.arcs:
                    next_node = node.arcs[letter]
                    new_path = path + [letter]

                    # If we've made a valid word, record it
                    if next_node.is_terminal:
                        # Find the complete word
                        complete_word = ''.join(prefix + new_path)

                        # Apply lowercase to indicate blanks
                        word_with_blanks = list(complete_word.upper())
                        for idx in used_blanks:
                            if 0 <= idx < len(word_with_blanks):
                                word_with_blanks[idx] = word_with_blanks[idx].lower()

                        formatted_word = ''.join(word_with_blanks)

                        # Calculate starting position
                        start_row = row - dr * len(prefix)
                        start_col = col - dc * len(prefix)

                        # Add the move
                        moves.append(Move(
                            row=start_row,
                            col=start_col,
                            direction='across' if is_horizontal else 'down',
                            word=formatted_word,
                            score=self._calculate_score(start_row, start_col, complete_word, is_horizontal)
                        ))

                    # Continue extending - make sure to increment position correctly
                    extend_right(next_node, new_path, (r + dr, c + dc), remaining_rack, used_blanks)
            else:
                # Empty square - try letters from rack that satisfy cross checks
                for i, letter in enumerate(remaining_rack):
                    is_blank = letter == ''

                    # For regular tiles, check if the letter is valid
                    if not is_blank:
                        if letter in node.arcs and letter in current_cross_checks:
                            next_node = node.arcs[letter]
                            new_path = path + [letter]
                            new_rack = remaining_rack.copy()
                            new_rack.pop(i)

                            # If we've made a valid word, record it
                            if next_node.is_terminal:
                                complete_word = ''.join(prefix + new_path)

                                start_row = row - dr * len(prefix)
                                start_col = col - dc * len(prefix)

                                moves.append(Move(
                                    row=start_row,
                                    col=start_col,
                                    direction='across' if is_horizontal else 'down',
                                    word=complete_word,
                                    score=self._calculate_score(start_row, start_col, complete_word, is_horizontal)
                                ))

                            # Continue extending - make sure to increment position correctly
                            extend_right(next_node, new_path, (r + dr, c + dc), new_rack, used_blanks)

                    # For blank tiles, try all valid cross-check letters
                    else:
                        for cross_letter in current_cross_checks:
                            if cross_letter in node.arcs:
                                next_node = node.arcs[cross_letter]
                                # Use lowercase to mark it was a blank
                                blank_pos = len(prefix) + len(path)
                                new_path = path + [cross_letter]
                                new_rack = remaining_rack.copy()
                                new_rack.pop(i)
                                new_used_blanks = used_blanks + [blank_pos]

                                # If we've made a valid word, record it
                                if next_node.is_terminal:
                                    complete_word = ''.join(prefix + new_path)

                                    # Mark blanks as lowercase in the final word
                                    word_with_blanks = list(complete_word.upper())
                                    for idx in new_used_blanks:
                                        if 0 <= idx < len(word_with_blanks):
                                            word_with_blanks[idx] = word_with_blanks[idx].lower()

                                    formatted_word = ''.join(word_with_blanks)

                                    start_row = row - dr * len(prefix)
                                    start_col = col - dc * len(prefix)

                                    moves.append(Move(
                                        row=start_row,
                                        col=start_col,
                                        direction='across' if is_horizontal else 'down',
                                        word=formatted_word,
                                        score=self._calculate_score(start_row, start_col, complete_word, is_horizontal)
                                    ))

                                # Continue extending - make sure to increment position correctly
                                extend_right(next_node, new_path, (r + dr, c + dc), new_rack, new_used_blanks)

        # Handle different cases based on prefix existence
        if prefix:
            # GADDAG pattern: go through reversed prefix to delimiter, then forward
            current_node = self.gaddag.root

            # Try to navigate through the reversed prefix
            for letter in reversed(prefix):
                if letter in current_node.arcs:
                    current_node = current_node.arcs[letter]
                else:
                    # Invalid prefix path in GADDAG
                    return moves

            # Check if we can cross the delimiter
            if self.gaddag.delimiter in current_node.arcs:
                # Start extending from after the delimiter
                delimiter_node = current_node.arcs[self.gaddag.delimiter]
                extend_right(delimiter_node, [], (row, col), rack_letters)
        else:
            # No prefix - we need to place the first letter at the anchor
            for i, letter in enumerate(rack_letters):
                is_blank = letter == ''

                if not is_blank:
                    # Regular tile
                    if letter in cross_checks:
                        # Check if placing this letter starts a valid word path
                        if letter in self.gaddag.root.arcs:
                            next_node = self.gaddag.root.arcs[letter]
                            new_rack = rack_letters.copy()
                            new_rack.pop(i)

                            # Check if a single letter is a valid word
                            if next_node.is_terminal:
                                moves.append(Move(
                                    row=row,
                                    col=col,
                                    direction='across' if is_horizontal else 'down',
                                    word=letter,
                                    score=self._calculate_score(row, col, letter, is_horizontal)
                                ))

                            # Try to extend this beginning
                            if self.gaddag.delimiter in next_node.arcs:
                                after_delimiter = next_node.arcs[self.gaddag.delimiter]
                                extend_right(after_delimiter, [letter], (row + dr, col + dc), new_rack)
                else:
                    # Blank tile - try all valid cross-check letters
                    for cross_letter in cross_checks:
                        if cross_letter in self.gaddag.root.arcs:
                            next_node = self.gaddag.root.arcs[cross_letter]
                            new_rack = rack_letters.copy()
                            new_rack.pop(i)

                            # Mark as lowercase to indicate blank
                            blank_letter = cross_letter.lower()

                            # Check if a single letter is a valid word
                            if next_node.is_terminal:
                                moves.append(Move(
                                    row=row,
                                    col=col,
                                    direction='across' if is_horizontal else 'down',
                                    word=blank_letter,
                                    score=self._calculate_score(row, col, cross_letter, is_horizontal)
                                ))

                            # Try to extend this beginning
                            if self.gaddag.delimiter in next_node.arcs:
                                after_delimiter = next_node.arcs[self.gaddag.delimiter]
                                extend_right(after_delimiter, [cross_letter], (row + dr, col + dc), new_rack, [0])

        return moves

    def _calculate_score(self, row: int, col: int, word: str, is_horizontal: bool) -> int:
        # Letter values in Scrabble
        letter_values = {
            'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1, 'F': 4, 'G': 2, 'H': 4, 'I': 1,
            'J': 8, 'K': 5, 'L': 1, 'M': 3, 'N': 1, 'O': 1, 'P': 3, 'Q': 10, 'R': 1,
            'S': 1, 'T': 1, 'U': 1, 'V': 4, 'W': 4, 'X': 8, 'Y': 4, 'Z': 10
        }

        word = word.upper()  # Convert to uppercase to match letter_values keys
        word_score = 0
        word_multiplier = 1  # Default, will be updated based on premium squares

        tiles_from_rack = 0
        for i, letter in enumerate(word):
            # Calculate the current position
            current_row = row
            current_col = col

            if is_horizontal:
                current_col += i
            else:
                current_row += i

            # Check if position is within board boundaries
            if (current_row < 0 or current_row >= 15 or
                    current_col < 0 or current_col >= 15):
                raise ValueError(f"Position ({current_row}, {current_col}) is outside the board")

            # Get letter value
            letter_value = letter_values.get(letter, 0)
            letter_multiplier = 1  # Default multiplier

            # Apply premium square effects
            premium = self.board[current_row][current_col].premium

            # Check if the square already has a tile (premium doesn't apply)
            tile_exists = self.board[current_row][current_col].tile is not None

            if not tile_exists:
                tiles_from_rack += 1

            if premium and not tile_exists:
                if premium == 'DL':
                    letter_multiplier = 2
                elif premium == 'TL':
                    letter_multiplier = 3
                elif premium == 'DW':
                    word_multiplier *= 2
                elif premium == 'TW':
                    word_multiplier *= 3

            # Add letter score with any letter multiplier
            word_score += letter_value * letter_multiplier

        # Apply word multiplier to get final score
        final_score = word_score * word_multiplier


        # Apply bingo bonus if all 7 tiles from rack were used
        if tiles_from_rack == 7:
            final_score += 50

        return final_score

    def is_connected_to_existing(self, row, col, word, is_horizontal):
        """Check if the word connects to existing tiles on the board."""
        dr, dc = (0, 1) if is_horizontal else (1, 0)

        # Check if any of the word's positions overlap with existing tiles
        for i in range(len(word)):
            r, c = row + i * dr, col + i * dc
            if 0 <= r < self.size and 0 <= c < self.size:
                # Direct overlap with existing tile
                if self.board[r][c].tile is not None:
                    return True

                # Check adjacent squares (orthogonal)
                for adj_dr, adj_dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    adj_r, adj_c = r + adj_dr, c + adj_dc
                    if (0 <= adj_r < self.size and 0 <= adj_c < self.size and
                            self.board[adj_r][adj_c].tile is not None):
                        return True

        return False

    def get_all_legal_moves(self, rack: List[Tile]) -> List[Move]:
        """Find all legal moves for the given rack."""
        if not self.gaddag:
            raise ValueError("Dictionary not loaded. Call load_dictionary() first.")

        moves = []
        anchors = self._get_all_anchors()

        for row, col in anchors:
            # Horizontal moves
            moves.extend(self._find_moves_from_anchor(row, col, rack, is_horizontal=True))

            # Vertical moves
            moves.extend(self._find_moves_from_anchor(row, col, rack, is_horizontal=False))

        # if not self._is_empty():
        #     moves = [move for move in moves if self.is_connected_to_existing(
        #         move.row, move.col, move.word, move.direction == 'across')]

        return sorted(moves, key=lambda m: m.score, reverse=True)

    def set_board(self, board: List[List[Tile]]):
        for row in range(self.size):
            for col in range(self.size):
                # Set the tile on the board, preserving premium squares
                if board[row][col] is not None:
                    self.board[row][col].tile = board[row][col]
                else:
                    self.board[row][col].tile = None

    def _add_horizontal(self, word: str, row, col):
        for i in range(len(word)):
            c = col + i
            self.board[row][c].tile = Tile(word[i])

    def _add_vertical(self, word: str, row, col):
        for i in range(len(word)):
            r = row + i
            self.board[r][col].tile = Tile(word[i])

    def __str__(self):
        result = ''
        for row in range(self.size):
            s = ''
            for col in range(self.size):
                tile = self.board[row][col].tile
                s += tile.letter if tile else ' '
            s += '\n'
            result += s
        return result