from collections import namedtuple, Counter
from typing import List, Set, Tuple, Optional, Dict

Move = namedtuple('Move', ['row', 'col', 'direction', 'word', 'score'])
letter_set = 'AĄBCĆDEĘFGHIJKLŁMNŃOÓPRSŚTUWYZŹŻ'
letter_values = {
            'A': 1, 'Ą': 9, 'B': 3, 'C': 2, 'Ć': 6, 'D': 2, 'E': 1, 'Ę': 5, 'F': 5, 'G': 3, 'H': 3, 'I': 1,
            'J': 3, 'K': 2, 'L': 2, 'Ł': 3, 'M': 2, 'N': 1, 'Ń': 7, 'O': 1, 'Ó': 5, 'P': 2, 'R': 1,
            'S': 1, 'Ś': 5, 'T': 2, 'U': 3, 'W': 1, 'Y': 2, 'Z': 1, 'Ź': 9, 'Ż': 5
        }

valid_letters_for_gaddag_and_crosscheck = frozenset(letter_values.keys()) - {'?'}


class Tile:
    def __init__(self, letter: str, is_blank: bool = False):
        self.letter = letter.upper() if letter else None
        self.is_blank = is_blank

    def __repr__(self):
        return f"Tile('{self.letter}', {self.is_blank})"

    def get_value_letter(self) -> Optional[str]:
        """Returns the letter used for GADDAG lookup and scoring base"""
        return self.letter.upper() if self.letter else None

    def get_score(self) -> int:
        """Gets the score, returning 0 for blanks"""
        if self.is_blank or not self.letter:
            return 0
        return letter_values.get(self.letter.upper(), 0)


class BoardSquare:
    def __init__(self, tile: Optional[Tile] = None, premium: str = None):
        self.tile = tile
        self.premium = premium  # 'DL', 'TL', 'DW', 'TW', or None

    def __repr__(self):
        return f"BoardSquare({self.tile}, '{self.premium}')"

    def is_empty(self) -> bool:
        return self.tile is None


class GADDAGNode:
    def __init__(self):
        self.arcs = {}  # Maps characters to other nodes
        self.is_terminal = False  # Indicates if a valid word ends here


class GADDAG:
    def __init__(self):
        self.root = GADDAGNode()
        self.delimiter = '>'  # Separator character

    def add_word(self, word: str) -> None:
        word = word.upper()
        for i in range(len(word)):
            # Create the reversed prefix + delimiter + suffix
            gaddag_str = word[i::-1] + self.delimiter + word[i + 1:]

            node = self.root
            for char in gaddag_str:
                if char not in node.arcs:
                    node.arcs[char] = GADDAGNode()
                node = node.arcs[char]
            node.is_terminal = True

    def is_valid_word(self, word: str) -> bool:
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
                if word:
                    self.add_word(word)


class ScrabbleBoard:
    def __init__(self, size: int = 15):
        self.size = size
        self.board = [[BoardSquare() for _ in range(size)] for _ in range(size)]
        self._set_premium_squares()
        self.gaddag = GADDAG()

    def load(self, dict_path: str = './dict.txt'):
        self.gaddag.build_from_dictionary(dict_path)

    def _is_valid_coord(self, r: int, c: int) -> bool:
        return 0 <= r < self.size and 0 <= c < self.size

    def _is_center(self, r: int, c: int) -> bool:
        return r == self.size // 2 and c == self.size // 2


    def _set_premium_squares(self) -> None:
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
        return self.board[7][7].is_empty()

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
            return set(letter_set)  # All letters valid

        # Check each letter
        valid_letters = set()
        for letter in letter_set:
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

    def _find_moves_from_anchor(self, anchor_row: int, anchor_col: int, rack: Counter, is_horizontal: bool) -> List[
        Tuple[str, List[Tuple[int, int, Tile]]]]:
        """
        Finds all valid word placements starting from a given anchor square,
        using the GADDAG. This function adapts the logic from scratch_6.py's gen/go_on.

        Args:
            anchor_row: The row of the anchor square.
            anchor_col: The column of the anchor square.
            rack: A Counter representing the player's tiles (e.g., {'A': 2, '?': 1}).
            is_horizontal: True if generating horizontal moves, False for vertical.

        Returns:
            A list of tuples: (word_string, placed_tiles_info).
            placed_tiles_info is a list of (row, col, tile_placed) tuples.
            'tile_placed' includes information if it was a blank and which letter it represents.
        """
        results = []
        (dr, dc) = (0, 1) if is_horizontal else (1, 0)  # Direction vector

        # Limit how far left we can place tiles from the anchor
        # `k` represents the number of squares to the left of the anchor we can potentially place tiles.
        # We can place at most len(rack)-1 tiles to the left, as one tile must be on the anchor.
        # We also cannot go off the board.
        limit = 0
        r, c = anchor_row - dr, anchor_col - dc
        while self._is_valid_coord(r, c) and self.board[r][c].is_empty():
            limit += 1
            r -= dr
            c -= dc
        max_left_extent = min(limit, len(rack) - 1 if len(
            rack) > 0 else 0)  # Can't place more tiles left than we have minus one for anchor

        # Inner recursive functions based on scratch_6.py's gen/go_on
        # We start by trying to place the *first* letter of the reversed prefix
        # to the *left* of the anchor, or on the anchor itself.

        def extend_left(current_r, current_c, current_node, current_rack, built_word_reversed, placed_tiles_so_far):
            """ Extends the word to the left (builds the reversed prefix in GADDAG terms)."""
            # Try placing a tile from the rack at (current_r, current_c)
            place_tile_at(current_r, current_c, current_node, current_rack, built_word_reversed, placed_tiles_so_far)

            # If the square to the left is empty and within limits, recurse left
            next_r, next_c = current_r - dr, current_c - dc
            # Check if we have tiles left *and* haven't gone too far left from the anchor
            distance_from_anchor = abs((next_r - anchor_row) * dr + (next_c - anchor_col) * dc)
            if distance_from_anchor <= max_left_extent and self._is_valid_coord(next_r, next_c) and self.board[next_r][
                next_c].is_empty():
                extend_left(next_r, next_c, current_node, current_rack, built_word_reversed, placed_tiles_so_far)

        def extend_right(current_r, current_c, current_node, current_rack, word_part_after_anchor, placed_tiles_so_far):
            """ Extends the word to the right (builds the suffix in GADDAG terms)."""
            # Base case: current_node is None (invalid path)
            if current_node is None:
                return

            # Check if the current path forms a valid word ending here
            if current_node.is_terminal and placed_tiles_so_far:  # Must place at least one tile
                # Reconstruct the full word
                anchor_pos_info = next((p for p in placed_tiles_so_far if p[0] == anchor_row and p[1] == anchor_col),
                                       None)
                if anchor_pos_info or not self.board[anchor_row][
                    anchor_col].is_empty():  # Ensure move involves the anchor
                    first_tile_info = min(placed_tiles_so_far,
                                          key=lambda x: (x[0] * dr + x[1] * dc))  # Find min row/col based on direction
                    start_r, start_c = first_tile_info[0], first_tile_info[1]
                    word = self._reconstruct_word(start_r, start_c, is_horizontal, placed_tiles_so_far)
                    if word:  # Check reconstruction worked
                        results.append((word, placed_tiles_so_far))

            # Check square to the right
            next_r, next_c = current_r + dr, current_c + dc
            if not self._is_valid_coord(next_r, next_c):
                return  # Off board

            square = self.board[next_r][next_c]
            if square.is_empty():
                # Try placing a tile from the rack
                place_tile_at(next_r, next_c, current_node, current_rack, word_part_after_anchor, placed_tiles_so_far,
                              extending_right=True)
            else:
                # Use existing tile on board
                tile = square.tile
                letter = tile.get_value_letter()
                if letter in current_node.arcs:
                    extend_right(next_r, next_c, current_node.arcs[letter], current_rack,
                                 word_part_after_anchor + letter, placed_tiles_so_far)

        def place_tile_at(r, c, prev_node, current_rack, built_word, placed_tiles_so_far, extending_right=False):
            """ Tries placing each possible tile from the rack at (r, c)."""
            cross_check_letters = self._get_cross_checks(r, c, is_horizontal)

            for tile_char, count in current_rack.items():
                if count == 0: continue

                is_blank = (tile_char == '?')
                possible_letters = valid_letters_for_gaddag_and_crosscheck if is_blank else [tile_char]

                for letter in possible_letters:
                    # Check GADDAG arc and cross-checks
                    if letter in prev_node.arcs and letter in cross_check_letters:
                        next_node = prev_node.arcs[letter]
                        new_rack = current_rack.copy()
                        new_rack[tile_char] -= 1

                        # Create the Tile object placed
                        placed_tile = Tile(letter, is_blank)  # Store the *assigned* letter for the blank
                        new_placed_tiles = placed_tiles_so_far + [(r, c, placed_tile)]

                        if extending_right:
                            extend_right(r, c, next_node, new_rack, built_word + letter, new_placed_tiles)
                        else:
                            # Check if we can transition to extending right (GADDAG delimiter)
                            if self.gaddag.delimiter in next_node.arcs:
                                delimiter_node = next_node.arcs[self.gaddag.delimiter]
                                # Start extending right from the anchor square
                                extend_right(anchor_row, anchor_col, delimiter_node, new_rack, "", new_placed_tiles)

                            # Continue extending left if possible
                            next_r, next_c = r - dr, c - dc
                            distance_from_anchor = abs((next_r - anchor_row) * dr + (next_c - anchor_col) * dc)
                            if distance_from_anchor <= max_left_extent and self._is_valid_coord(next_r, next_c) and \
                                    self.board[next_r][next_c].is_empty():
                                extend_left(next_r, next_c, next_node, new_rack, letter + built_word, new_placed_tiles)

        # --- Initial Call ---
        # The generation logic starts from the anchor square.
        # We either use an existing tile at the anchor or try placing a tile there.

        anchor_square = self.board[anchor_row][anchor_col]
        if not anchor_square.is_empty():
            # If anchor is occupied, start extending right using the existing letter
            tile = anchor_square.tile
            letter = tile.get_value_letter()
            if letter in self.gaddag.root.arcs:
                # If the anchor letter is a starting point in GADDAG (for right extension)
                # This case is complex with GADDAG, usually moves *must* place a tile.
                # A simpler approach might be to only anchor on empty squares adjacent to tiles.
                # However, the definition allows anchoring on occupied squares if extending.
                # For now, we focus on placing tiles *through* the anchor.
                pass  # Skip starting on occupied anchor for now, focus on placing tiles
        else:
            # Anchor is empty, try placing tiles on it, initiating the left/right extensions
            place_tile_at(anchor_row, anchor_col, self.gaddag.root, rack, "", [])

        # Post-processing: Ensure unique moves (same word, pos, dir can be found via different paths)
        # The `results` list currently holds (word_string, placed_tiles_info)
        # We need to derive the start position and direction to make them unique.
        unique_moves = {}  # Dict key: (start_r, start_c, direction, word), value: placed_tiles_info
        for word, placed_tiles in results:
            if not placed_tiles: continue
            min_r = min(r for r, c, t in placed_tiles)
            min_c = min(c for r, c, t in placed_tiles)
            start_r, start_c = (min_r, min_c)  # Approximation, needs refinement based on direction

            # Determine start coordinate accurately based on direction
            first_tile_info = min(placed_tiles, key=lambda x: (x[0] * dr + x[1] * dc))
            start_r, start_c = first_tile_info[0], first_tile_info[1]

            direction_char = 'H' if is_horizontal else 'V'
            move_key = (start_r, start_c, direction_char, word)
            if move_key not in unique_moves:
                unique_moves[move_key] = placed_tiles
            # Optional: Keep the one with more placed tiles if duplicates arise?

        # Convert back to the required format
        final_results = []
        for (r, c, d, w), p_tiles in unique_moves.items():
            final_results.append((w, p_tiles))  # Return word and placement info for scoring

        return final_results

    def _reconstruct_word(self, start_r, start_c, is_horizontal, placed_tiles_info):
        """ Reconstructs the full word string from the starting position and placed tiles. """
        word_letters = []
        (dr, dc) = (0, 1) if is_horizontal else (1, 0)
        r, c = start_r, start_c
        placed_dict = {(pr, pc): tile for pr, pc, tile in placed_tiles_info}

        while self._is_valid_coord(r, c):
            square = self.board[r][c]
            if (r, c) in placed_dict:
                tile = placed_dict[(r, c)]
                # Use the assigned letter if blank, otherwise the tile letter
                letter = tile.letter if not tile.is_blank else tile.letter  # Already assigned
                if not letter or letter == '?': return None  # Error case: blank wasn't assigned
                word_letters.append(letter)
            elif not square.is_empty():
                word_letters.append(square.tile.get_value_letter())
            else:
                # Found empty square, check if it's the end *after* the last placed tile
                max_pos = max(pr * dr + pc * dc for pr, pc, t in placed_tiles_info)
                current_pos = r * dr + c * dc
                if current_pos > max_pos:
                    break  # Word ends here
                else:
                    # Gap in the word - should not happen with GADDAG generation if correct
                    # print(f"Warning: Gap detected reconstructing word at {r},{c}")
                    return None  # Indicate error

            r += dr
            c += dc

        return "".join(word_letters) if word_letters else None

    def _calculate_score(self, placed_tiles: List[Tuple[int, int, Tile]], is_horizontal: bool) -> Tuple[int, str]:
        """
        Calculates the score of placing the given tiles.

        Args:
            placed_tiles: List of (row, col, tile_object) tuples for tiles placed from rack.
                          The tile_object stores the letter used (even for blanks).
            is_horizontal: True if the main word is horizontal, False if vertical.

        Returns:
            Tuple: (total_score, word_string)
                   Returns (0, "") if the placement is invalid or results in no word.
        """
        if not placed_tiles:
            return 0, ""

        total_score = 0
        main_word_multiplier = 1
        main_word_letter_score_sum = 0
        bingo_bonus = 50 if len(placed_tiles) == 7 else 0

        (dr, dc) = (0, 1) if is_horizontal else (1, 0)
        (cross_dr, cross_dc) = (1, 0) if is_horizontal else (0, 1)

        # Find the full extent of the main word
        placed_coords = set((r, c) for r, c, t in placed_tiles)
        min_coord = min(r * dr + c * dc for r, c, t in placed_tiles)
        max_coord = max(r * dr + c * dc for r, c, t in placed_tiles)

        # Find start of the main word by going backwards from the first placed tile
        first_placed_tile_info = min(placed_tiles, key=lambda x: x[0] * dr + x[1] * dc)
        start_r, start_c = first_placed_tile_info[0], first_placed_tile_info[1]
        while True:
            prev_r, prev_c = start_r - dr, start_c - dc
            if self._is_valid_coord(prev_r, prev_c) and not self.board[prev_r][prev_c].is_empty():
                start_r, start_c = prev_r, prev_c
            else:
                break

        # Iterate through the main word to calculate score
        current_r, current_c = start_r, start_c
        main_word_str_list = []
        while self._is_valid_coord(current_r, current_c):
            square = self.board[current_r][current_c]
            tile_info = next((t for r, c, t in placed_tiles if r == current_r and c == current_c), None)

            if tile_info:  # This is a newly placed tile
                tile_object = tile_info
                letter_value = 0 if tile_object.is_blank else letter_values[tile_object.letter.upper()]
                main_word_str_list.append(tile_object.letter)  # Use assigned letter

                premium = square.premium
                letter_multiplier = 1
                if premium == 'DL':
                    letter_multiplier = 2
                elif premium == 'TL':
                    letter_multiplier = 3

                current_letter_score = letter_value * letter_multiplier
                main_word_letter_score_sum += current_letter_score

                if premium == 'DW':
                    main_word_multiplier *= 2
                elif premium == 'TW':
                    main_word_multiplier *= 3

                # Calculate cross-word score IF a cross word is formed
                cross_word_letters = [tile_object.letter]  # Start with the placed letter
                cross_score = 0
                cross_word_multiplier = 1  # Multiplier for the cross-word itself
                has_cross_neighbors = False

                # Look perpendicularly backwards
                pr, pc = current_r - cross_dr, current_c - cross_dc
                prefix = []
                while self._is_valid_coord(pr, pc) and not self.board[pr][pc].is_empty():
                    has_cross_neighbors = True
                    cross_tile = self.board[pr][pc].tile
                    prefix.append(cross_tile.get_value_letter())
                    cross_score += cross_tile.get_score()  # Add score of existing tiles
                    pr -= cross_dr
                    pc -= cross_dc
                cross_word_letters = prefix[::-1] + cross_word_letters

                # Look perpendicularly forwards
                sr, sc = current_r + cross_dr, current_c + cross_dc
                suffix = []
                while self._is_valid_coord(sr, sc) and not self.board[sr][sc].is_empty():
                    has_cross_neighbors = True
                    cross_tile = self.board[sr][sc].tile
                    suffix.append(cross_tile.get_value_letter())
                    cross_score += cross_tile.get_score()  # Add score of existing tiles
                    sr += cross_dr
                    sc += cross_dc
                cross_word_letters.extend(suffix)

                # If a cross word longer than 1 letter was formed
                if has_cross_neighbors and len(cross_word_letters) > 1:
                    # Check validity (redundant if GADDAG/cross-checks worked, but good safeguard)
                    cross_word_str = "".join(cross_word_letters)
                    if not self.gaddag.is_valid_word(cross_word_str):
                        # print(f"Error: Invalid cross-word '{cross_word_str}' generated at {current_r},{current_c}")
                        return 0, ""  # Invalid move

                    # Apply premium from the *shared* square to the cross-word score sum
                    cross_score += current_letter_score  # Add score of the placed tile (already includes letter premium)
                    # Apply word multipliers from the *shared* square
                    if premium == 'DW':
                        cross_word_multiplier *= 2
                    elif premium == 'TW':
                        cross_word_multiplier *= 3
                    # Apply word multipliers from *other* squares in the cross word (already accounted for in cross_score for existing tiles?)
                    # No, need to re-evaluate premiums for existing tiles within the cross-word context if needed, but standard rules usually only apply premiums once per placement.
                    # The common rule: premiums on squares only count when a tile is first placed there.
                    # So, cross_score just sums the base values of existing tiles + the premium-adjusted value of the new tile.
                    # Word multipliers apply to the sum.

                    total_score += (cross_score * cross_word_multiplier)

            elif not square.is_empty():  # Existing tile, part of the main word
                tile_object = square.tile
                letter_value = tile_object.get_score()
                main_word_letter_score_sum += letter_value
                main_word_str_list.append(tile_object.get_value_letter())
            else:
                # Found an empty square - word should end here unless it's before the first placed tile
                current_pos = current_r * dr + current_c * dc
                if current_pos < min_coord or current_pos > max_coord:
                    break  # Reached end of the word
                else:
                    # Error: Gap in word
                    # print(f"Error: Gap found during scoring at {current_r},{current_c}")
                    return 0, ""

            # Move to the next square in the main word direction
            current_r += dr
            current_c += dc

        # Add score from the main word itself
        total_score += (main_word_letter_score_sum * main_word_multiplier)
        # Add bingo bonus
        total_score += bingo_bonus

        main_word_str = "".join(main_word_str_list)

        # Final check: main word must be valid
        if len(main_word_str) < 2:
            # This check might be too strict if single-letter words are allowed in the dict
            pass  # Allow potentially valid single-letter plays if dict supports it

        if not self.gaddag.is_valid_word(main_word_str):
            # print(f"Error: Invalid main word '{main_word_str}' generated.")
            return 0, ""

        return total_score, main_word_str

    def get_all_legal_moves(self, rack_tiles: List[Tile]) -> List[Move]:
        """
        Finds all valid moves for the given rack on the current board state.

        Args:
            rack_tiles: A list of Tile objects representing the player's rack.

        Returns:
            A list of Move namedtuples, sorted by score descending.
        """
        rack = Counter(t.letter if not t.is_blank else '?' for t in rack_tiles)
        anchors = self._get_all_anchors()
        potential_moves: Dict[Tuple[int, int, str, str], Tuple[
            int, List[Tuple[int, int, Tile]]]] = {}  # Key: (r, c, dir, word), Value: (score, placed_tiles)
        center_coord = (self.size // 2, self.size // 2)
        is_first = self._is_empty()

        if not anchors and not is_first:
            print("No anchors found on non-empty board.")
            return []  # No possible moves if no anchors

        if is_first and not anchors:  # Should not happen if _get_all_anchors is correct
            print("Error: Board empty but no center anchor found.")
            return []

        for r_anchor, c_anchor in anchors:
            # Generate Horizontal moves
            h_moves = self._find_moves_from_anchor(r_anchor, c_anchor, rack.copy(), is_horizontal=True)
            for word, placed_tiles in h_moves:
                if not placed_tiles: continue
                # First move must cross center
                if is_first and not any(r == center_coord[0] and c == center_coord[1] for r, c, t in placed_tiles):
                    continue

                score, word_validated = self._calculate_score(placed_tiles, is_horizontal=True)
                if score > 0 and word == word_validated:  # Check score is positive and word reconstruction matches
                    first_tile_info = min(placed_tiles, key=lambda x: x[1])  # Min col for H move
                    start_r, start_c = first_tile_info[0], first_tile_info[1]
                    move_key = (start_r, start_c, 'H', word)
                    if move_key not in potential_moves or score > potential_moves[move_key][0]:
                        potential_moves[move_key] = (score, placed_tiles)

            # Generate Vertical moves
            v_moves = self._find_moves_from_anchor(r_anchor, c_anchor, rack.copy(), is_horizontal=False)
            for word, placed_tiles in v_moves:
                if not placed_tiles: continue
                # First move must cross center
                if is_first and not any(r == center_coord[0] and c == center_coord[1] for r, c, t in placed_tiles):
                    continue

                score, word_validated = self._calculate_score(placed_tiles, is_horizontal=False)
                if score > 0 and word == word_validated:
                    first_tile_info = min(placed_tiles, key=lambda x: x[0])  # Min row for V move
                    start_r, start_c = first_tile_info[0], first_tile_info[1]
                    move_key = (start_r, start_c, 'V', word)
                    if move_key not in potential_moves or score > potential_moves[move_key][0]:
                        potential_moves[move_key] = (score, placed_tiles)

        # Convert to Move objects
        final_moves = []
        for (r, c, d, w), (score, placed) in potential_moves.items():
            # Optional: Modify word string to show blanks e.g., 'Wo(R)D' if 'R' was blank
            # For now, just use the standard word string.
            final_moves.append(Move(row=r, col=c, direction=d, word=w, score=score))

        # Sort by score descending
        final_moves.sort(key=lambda x: x.score, reverse=True)

        # After the first successful move, update the board state flag
        # This should ideally happen *after* a move is chosen and applied, not during generation.
        # if is_first and final_moves:
        #    self.is_first_move = False # Move this logic to where a move is actually placed on the board

        return final_moves

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

    def print_board(self):
        """Prints the current board state to the console."""
        print("   " + "  ".join(f"{i:<2}" for i in range(self.size)))
        print("  +" + "---+" * self.size)
        for r in range(self.size):
            row_str = f"{r:<2}|"
            for c in range(self.size):
                square = self.board[r][c]
                if square.tile:
                    content = square.tile.get_value_letter()
                else: content = ' ' # Empty non-premium

                # Pad content to 2 chars for alignment
                row_str += f" {content:<2}|" # Pad right
            print(row_str)
            print("  +" + "---+" * self.size)
