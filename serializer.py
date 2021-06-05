import chess
import numpy as np


class BoardSerializer:
    def __init__(
        self,
        is_pieces: bool = True,
        is_enpassant: bool = True,
        is_castling: bool = True,
        is_turn: bool = True,
        dtype=np.int8,  # TODO: add type
    ) -> None:
        self.is_pieces = is_pieces
        self.is_enpassant = is_enpassant
        self.is_castling = is_castling
        self.is_turn = is_turn
        self.dtype = dtype

    def _piece_type_to_index(self, piece_type: str) -> int:
        PIECE_TYPE_TO_INT = dict(
            zip(
                ["K", "Q", "R", "B", "N", "P", "k", "q", "r", "b", "n", "p"],
                range(0, 12),
            )
        )
        return PIECE_TYPE_TO_INT[piece_type]

    def _serialize_pieces(self, board: chess.Board) -> np.ndarray:
        if not self.is_pieces:
            return np.empty(shape=(8, 8, 0), dtype=self.dtype)

        pieces = np.zeros(shape=(8 * 8, 12), dtype=self.dtype)
        for sq_index, piece_type in board.piece_map().items():
            index = self._piece_type_to_index(piece_type.symbol())
            pieces[sq_index, index] = 1

        return pieces.reshape((8, 8, 12))

    def _serialize_enpassant(self, board: chess.Board) -> np.ndarray:
        if not self.is_enpassant:
            return np.empty(shape=(8, 8, 0), dtype=self.dtype)

        enpassant = np.zeros(shape=(8 * 8, 1), dtype=self.dtype)
        if board.ep_square:
            sq_index = board.ep_square
            enpassant[sq_index, 0] = 1

        return enpassant.reshape((8, 8, 1))

    def _serialize_castling(self, board: chess.Board) -> np.ndarray:
        if not self.is_castling:
            return np.empty(shape=(8, 8, 0), dtype=self.dtype)

        castling = np.zeros(shape=(8, 8, 4), dtype=self.dtype)
        castling[:, :, 0] = (
            1 if board.has_kingside_castling_rights(color=chess.WHITE) else 0
        )
        castling[:, :, 1] = (
            1 if board.has_queenside_castling_rights(color=chess.WHITE) else 0
        )
        castling[:, :, 2] = (
            1 if board.has_kingside_castling_rights(color=chess.BLACK) else 0
        )
        castling[:, :, 3] = (
            1 if board.has_queenside_castling_rights(color=chess.BLACK) else 0
        )

        return castling

    def _serialize_turn(self, board: chess.Board) -> np.ndarray:
        if not self.is_turn:
            return np.empty(shape=(8, 8, 0), dtype=self.dtype)

        turn = np.zeros(shape=(8, 8, 2), dtype=self.dtype)
        if board.turn == chess.WHITE:
            turn[:, :, 0] = 1
        else:
            turn[:, :, 1] = 1

        return turn

    def serialize(
        self,
        board: chess.Board,
    ) -> np.ndarray:
        pieces = self._serialize_pieces(board)
        enpassant = self._serialize_enpassant(board)
        castling = self._serialize_castling(board)
        turn = self._serialize_turn(board)
        array = np.concatenate([pieces, enpassant, castling, turn], axis=2)
        return array
