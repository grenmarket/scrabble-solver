import string
from unittest import TestCase

from board import GADDAG, ScrabbleBoard, Tile


class TestGADDAG(TestCase):

    def test_valid(self):
        gaddag = GADDAG()
        gaddag.add_word('zdrowy')
        gaddag.add_word('mama')
        gaddag.add_word('rowy')
        gaddag.add_word('wyro')
        gaddag.add_word('mam')
        gaddag.add_word('krowy')
        gaddag.add_word('omama')
        assert gaddag.is_valid_word('mam')
        assert gaddag.is_valid_word('rowy')
        assert gaddag.is_valid_word('zdrowy')
        assert gaddag.is_valid_word('wyro')
        assert gaddag.is_valid_word('mama')
        assert gaddag.is_valid_word('krowy')
        assert not gaddag.is_valid_word('ma')
        assert not gaddag.is_valid_word('row')
        assert not gaddag.is_valid_word('krowym')
        assert not gaddag.is_valid_word('krow')
        assert not gaddag.is_valid_word('omam')
        assert not gaddag.is_valid_word('wyromam')


class TestBoard(TestCase):

    def test_is_empty(self):
        board = ScrabbleBoard()
        assert board._is_empty()
        board.board[4][4].tile = Tile('L')
        assert not board._is_empty()

    def test_anchors_empty(self):
        board = ScrabbleBoard()
        anchors = board._get_all_anchors()
        assert anchors == [(7,7)]

    def test_anchors_1(self):
        board = ScrabbleBoard()
        board._add_horizontal('sto', 0, 0)
        anchors = set(board._get_all_anchors())
        assert {(1,0),(1,1),(1,2),(0,3)} == anchors

    def test_anchors_2(self):
        board = ScrabbleBoard()
        board._add_horizontal('krowa', 10, 10)
        board._add_vertical('kok', 9, 12)
        anchors = set(board._get_all_anchors())
        assert {(9,10),(9,11),(8,12),(9,13),(9,14),(10,9),(11,10),(11,11),(12, 12),(11,13),(11,14)} == anchors

    def test_anchors_3(self):
        board = ScrabbleBoard()
        board._add_horizontal('kok', 7, 12)
        board._add_horizontal('kok', 9, 12)
        board._add_vertical('kok', 7, 12)
        anchors = set(board._get_all_anchors())
        assert {(6,12),(6,13),(6,14),(10,12),(10,13),(10,14),(7,11),(8,11),(9,11),(8,13),(8,14)} == anchors

    def test_cross_checks_1(self):
        board = ScrabbleBoard()
        board.load('./test-dictionary.txt')
        board._add_horizontal('trowa', 8, 8)
        assert {'A'} == board._get_cross_checks(7, 8, True)
        assert {'O','A'} == board._get_cross_checks(7, 9, True)
        assert {'T','R'} == board._get_cross_checks(9, 12, True)
        assert {'R'} == board._get_cross_checks(8, 13, False)

    def test_cross_checks_2(self):
        board = ScrabbleBoard()
        board.load('./test-dictionary.txt')
        board._add_horizontal('at', 8, 8)
        board._add_vertical('nora', 1, 1)
        assert {'R','F'} == board._get_cross_checks(8, 7, False)
        assert {'A'} == board._get_cross_checks(0, 1, True)
        assert set(string.ascii_uppercase) == board._get_cross_checks(0, 2, True)

    def test_moves_1(self):
        board = ScrabbleBoard()
        board.load('./test-dictionary.txt')
        board._add_horizontal('at', 8, 8)
        tiles = [Tile(letter) for letter in 'UNCLEA']
        print(board._find_moves_from_anchor(9, 9, tiles, True))


