import string
from collections import defaultdict, namedtuple
from typing import List, Dict, Set, Tuple, Optional


# Define data structures
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


Move = namedtuple('Move', ['row', 'col', 'direction', 'word', 'score'])


# GADDAG node structure
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
        """Build the GADDAG from a dictionary file."""
        with open(dictionary_path, 'r') as file:
            for line in file:
                word = line.strip()
                if word and all(c.isalpha() for c in word):
                    self.add_word(word)


# Scrabble board and move generation
class ScrabbleBoard:
    def __init__(self, size: int = 15):
        self.size = size
        self.board = [[BoardSquare() for _ in range(size)] for _ in range(size)]
        self.set_premium_squares()
        self.gaddag = None

    def set_premium_squares(self) -> None:
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

        # For visual verification, you can add this debug output
        # self.print_premium_squares()

    def load_dictionary(self, dictionary_path: str) -> None:
        """Load the dictionary into a GADDAG structure."""
        self.gaddag = GADDAG()
        self.gaddag.build_from_dictionary(dictionary_path)

    def is_empty(self) -> bool:
        """Check if the board is empty."""
        for row in self.board:
            for square in row:
                if square.tile is not None:
                    return False
        return True

    def get_all_anchors(self) -> List[Tuple[int, int]]:
        """Find all anchor squares (adjacent to existing tiles)."""
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
        if self.is_empty():
            anchors.append((self.size // 2, self.size // 2))

        return anchors

    def get_cross_checks(self, row: int, col: int, is_horizontal: bool) -> Set[str]:
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
            if len(cross_word) == 1 or self.gaddag.is_valid_word(''.join(cross_word)):
                valid_letters.add(letter)

        return valid_letters

    def find_moves_from_anchor(self, row: int, col: int, rack: List[Tile], is_horizontal: bool = True) -> List[Move]:
        """Find all legal moves that include the anchor square."""
        moves = []

        # Skip if square is already occupied
        if self.board[row][col].tile is not None:
            return moves

        # Direction vectors
        dr, dc = (0, 1) if is_horizontal else (1, 0)

        # Get rack letters
        rack_letters = [tile.letter for tile in rack]

        # Get cross-checks
        cross_checks = self.get_cross_checks(row, col, is_horizontal)

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

        # Find possible suffix extensions from the anchor
        def extend_right(node: GADDAGNode, path: List[str], pos: Tuple[int, int], remaining_rack: List[str]):
            r, c = pos
            if r < 0 or r >= self.size or c < 0 or c >= self.size:
                return

            # If square is empty, try letters from rack
            if self.board[r][c].tile is None:
                # Try each letter in the rack
                for i, letter in enumerate(remaining_rack):
                    if letter in node.arcs and letter in cross_checks:
                        new_path = path + [letter]
                        new_remaining = remaining_rack[:i] + remaining_rack[i + 1:]

                        next_node = node.arcs[letter]
                        if next_node.is_terminal:
                            # Found a valid word
                            word = ''.join(prefix + new_path)
                            moves.append(Move(
                                row=row - dr * len(prefix),
                                col=col - dc * len(prefix),
                                direction='across' if is_horizontal else 'down',
                                word=word,
                                score=self.calculate_score(row - dr * len(prefix), col - dc * len(prefix), word,
                                                           is_horizontal)
                            ))

                        # Continue extending
                        extend_right(next_node, new_path, (r + dr, c + dc), new_remaining)

                # Try blanks in the rack (if any)
                blank_indices = [i for i, tile in enumerate(rack) if tile.is_blank]
                for blank_idx in blank_indices:
                    for letter in cross_checks:
                        if letter in node.arcs:
                            # Use blank as this letter
                            new_remaining = remaining_rack.copy()
                            del new_remaining[blank_idx]
                            new_path = path + [letter.lower()]  # Lowercase to indicate blank

                            next_node = node.arcs[letter]
                            if next_node.is_terminal:
                                word = ''.join(prefix + new_path)
                                moves.append(Move(
                                    row=row - dr * len(prefix),
                                    col=col - dc * len(prefix),
                                    direction='across' if is_horizontal else 'down',
                                    word=word,
                                    score=self.calculate_score(row - dr * len(prefix), col - dc * len(prefix), word,
                                                               is_horizontal)
                                ))

                            # Continue extending
                            extend_right(next_node, new_path, (r + dr, c + dc), new_remaining)

            # If square already has a tile, incorporate it
            else:
                letter = self.board[r][c].tile.letter
                if letter in node.arcs:
                    next_node = node.arcs[letter]
                    new_path = path + [letter]

                    if next_node.is_terminal:
                        word = ''.join(prefix + new_path)
                        moves.append(Move(
                            row=row - dr * len(prefix),
                            col=col - dc * len(prefix),
                            direction='across' if is_horizontal else 'down',
                            word=word,
                            score=self.calculate_score(row - dr * len(prefix), col - dc * len(prefix), word,
                                                       is_horizontal)
                        ))

                    # Continue extending
                    extend_right(next_node, new_path, (r + dr, c + dc), remaining_rack)

        # Start extending from each prefix possibility
        if not prefix:  # No prefix, start from root
            for letter in rack_letters:
                if letter in self.gaddag.root.arcs and letter in cross_checks:
                    # This is a valid starting letter
                    start_node = self.gaddag.root.arcs[letter]
                    new_rack = rack_letters.copy()
                    new_rack.remove(letter)

                    extend_right(start_node, [letter], (row + dr, col + dc), new_rack)
        else:
            # With prefix, navigate to the correct GADDAG node
            node = self.gaddag.root
            for letter in prefix[-1::-1]:  # Reverse the prefix for GADDAG
                if letter in node.arcs:
                    node = node.arcs[letter]
                else:
                    return moves  # Invalid prefix

            # Crossed the delimiter in GADDAG
            if self.gaddag.delimiter in node.arcs:
                node = node.arcs[self.gaddag.delimiter]
                extend_right(node, [], (row, col), rack_letters)

        return moves

    def calculate_score(self, row: int, col: int, word: str, is_horizontal: bool) -> int:
        """Calculate the score for a word placed on the board.

        Args:
            row: Starting row for the word
            col: Starting column for the word
            word: The word to calculate the score for
            is_horizontal: True if the word is placed horizontally, False for vertical

        Returns:
            The total score for the word, including premium square bonuses
        """
        # Letter values in Scrabble
        letter_values = {
            'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1, 'F': 4, 'G': 2, 'H': 4, 'I': 1,
            'J': 8, 'K': 5, 'L': 1, 'M': 3, 'N': 1, 'O': 1, 'P': 3, 'Q': 10, 'R': 1,
            'S': 1, 'T': 1, 'U': 1, 'V': 4, 'W': 4, 'X': 8, 'Y': 4, 'Z': 10
        }

        word = word.upper()  # Convert to uppercase to match letter_values keys
        word_score = 0
        word_multiplier = 1  # Default, will be updated based on premium squares

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
            tile_exists = hasattr(self.board[current_row][current_col], 'tile') and self.board[current_row][
                current_col].tile is not None

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

        # Bonus for using all 7 tiles (if applicable)
        if len(word) == 7:  # A "bingo" in Scrabble
            final_score += 50

        return final_score

    def get_all_legal_moves(self, rack: List[Tile]) -> List[Move]:
        """Find all legal moves for the given rack."""
        if not self.gaddag:
            raise ValueError("Dictionary not loaded. Call load_dictionary() first.")

        moves = []
        anchors = self.get_all_anchors()

        for row, col in anchors:
            # Horizontal moves
            moves.extend(self.find_moves_from_anchor(row, col, rack, is_horizontal=True))

            # Vertical moves
            moves.extend(self.find_moves_from_anchor(row, col, rack, is_horizontal=False))

        return moves


def find_all_moves(board_state: List[List[dict]], rack: List[dict], dictionary_path: str = "TWL06.txt") -> List[dict]:
    """
    Find all legal Scrabble moves given a board state and rack.

    Args:
        board_state: 2D array of dicts with 'letter' and 'isBlank' fields or None
        rack: List of dicts with 'letter' and 'isBlank' fields
        dictionary_path: Path to the dictionary file

    Returns:
        List of move dictionaries with position, direction, word, and score
    """
    # Convert input format to internal representation
    size = len(board_state)
    board = ScrabbleBoard(size)

    # Set up the board
    for r in range(size):
        for c in range(size):
            cell = board_state[r][c]
            if cell and cell.get('letter'):
                board.board[r][c].tile = Tile(cell['letter'], cell.get('isBlank', False))

    # Convert rack
    rack_tiles = [Tile(tile['letter'], tile.get('isBlank', False)) for tile in rack]

    # Load dictionary
    board.load_dictionary(dictionary_path)

    # Find moves
    moves = board.get_all_legal_moves(rack_tiles)

    # Convert to output format
    return [
        {
            'startRow': move.row,
            'startCol': move.col,
            'direction': move.direction,
            'word': move.word,
            'score': move.score
        }
        for move in sorted(moves, key=lambda m: m.score, reverse=True)
    ]


# Example usage
if __name__ == "__main__":
    # Example board state (empty 15x15 board)
    board_state = [[None for _ in range(15)] for _ in range(15)]

    # Example rack
    rack = [
        {'letter': 'A', 'isBlank': False},
        {'letter': 'B', 'isBlank': False},
        {'letter': 'C', 'isBlank': False},
        {'letter': 'D', 'isBlank': False},
        {'letter': 'E', 'isBlank': False},
        {'letter': 'F', 'isBlank': False},
        {'letter': 'G', 'isBlank': False}
    ]

    # Find all moves
    moves = find_all_moves(board_state, rack, "dictionary.txt")
    print(f"Found {len(moves)} possible moves.")
    for move in moves[:10]:  # Show top 10 moves
        print(
            f"{move['word']} at ({move['startRow']}, {move['startCol']}) {move['direction']} - Score: {move['score']}")