from typing import Union
import chess
import numpy as np


class ChessPositionSerializer:
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

    def _serialize_board_pieces(self, board: chess.Board) -> np.ndarray:
        if not self.is_pieces:
            return np.empty(shape=(0, 8, 8), dtype=self.dtype)

        pieces = np.zeros(shape=(12, 8 * 8), dtype=self.dtype)
        for sq_index, piece_type in board.piece_map().items():
            index = self._piece_type_to_index(piece_type.symbol())
            pieces[index, sq_index] = 1

        return pieces.reshape((12, 8, 8))

    def _serialize_board_enpassant(self, board: chess.Board) -> np.ndarray:
        if not self.is_enpassant:
            return np.empty(shape=(0, 8, 8), dtype=self.dtype)

        enpassant = np.zeros(shape=(1, 8 * 8), dtype=self.dtype)
        if board.ep_square:
            sq_index = board.ep_square
            enpassant[0, sq_index] = 1

        return enpassant.reshape((1, 8, 8))

    def _serialize_board_castling(self, board: chess.Board) -> np.ndarray:
        if not self.is_castling:
            return np.empty(shape=(0, 8, 8), dtype=self.dtype)

        castling = np.zeros(shape=(4, 8, 8), dtype=self.dtype)
        castling[0, :, :] = (
            1 if board.has_kingside_castling_rights(color=chess.WHITE) else 0
        )
        castling[1, :, :] = (
            1 if board.has_queenside_castling_rights(color=chess.WHITE) else 0
        )
        castling[2, :, :] = (
            1 if board.has_kingside_castling_rights(color=chess.BLACK) else 0
        )
        castling[3, :, :] = (
            1 if board.has_queenside_castling_rights(color=chess.BLACK) else 0
        )

        return castling

    def _serialize_board_turn(self, board: chess.Board) -> np.ndarray:
        if not self.is_turn:
            return np.empty(shape=(0, 8, 8), dtype=self.dtype)

        turn = np.zeros(shape=(2, 8, 8), dtype=self.dtype)
        if board.turn == chess.WHITE:
            turn[0, :, :] = 1
        elif board.turn == chess.BLACK:
            turn[1, :, :] = 1

        return turn

    def _serialize_fen_pieces(self, fen: str) -> np.ndarray:
        if not self.is_pieces:
            return np.empty(shape=(0, 8, 8), dtype=self.dtype)

        pieces_str = fen.split(" ")[0]
        pieces = np.zeros(shape=(12, 8 * 8), dtype=self.dtype)
        rank_index = 8
        file_index = 1
        for char in pieces_str:
            if char.isdigit():
                digit = int(char)
                file_index += digit
            elif char.isalpha():
                index = self._piece_type_to_index(char)
                sq_index = chess.square(file_index-1, rank_index-1)
                pieces[index, sq_index] = 1
                file_index += 1
            elif char == "/":
                rank_index -= 1
                file_index = 1

        return pieces.reshape((12, 8, 8))

    def _serialize_fen_enpassant(self, fen: str) -> np.ndarray:
        if not self.is_enpassant:
            return np.empty(shape=(0, 8, 8), dtype=self.dtype)

        enpassant_str = fen.split(" ")[3]
        enpassant = np.zeros(shape=(1, 8 * 8), dtype=self.dtype)
        if enpassant_str != "-":
            sq_index = chess.parse_square(enpassant_str)
            enpassant[0, sq_index] = 1

        return enpassant.reshape((1, 8, 8))

    def _serialize_fen_castling(self, fen: str) -> np.ndarray:
        if not self.is_castling:
            return np.empty(shape=(0, 8, 8), dtype=self.dtype)

        castling_str = fen.split(" ")[2]
        castling = np.zeros(shape=(4, 8, 8), dtype=self.dtype)
        castling[0, :, :] = (
            1 if "K" in castling_str else 0
        )
        castling[1, :, :] = (
            1 if "Q" in castling_str else 0
        )
        castling[2, :, :] = (
            1 if "k" in castling_str else 0
        )
        castling[3, :, :] = (
            1 if "q" in castling_str else 0
        )

        return castling

    def _serialize_fen_turn(self, fen: str) -> np.ndarray:
        if not self.is_turn:
            return np.empty(shape=(0, 8, 8), dtype=self.dtype)

        turn_str = fen.split(" ")[1]
        turn = np.zeros(shape=(2, 8, 8), dtype=self.dtype)
        if turn_str == "w":
            turn[0, :, :] = 1
        elif turn_str == "b":
            turn[1, :, :] = 1

        return turn

    def _serialize_board(self, board: chess.Board) -> np.ndarray:
        pieces = self._serialize_board_pieces(board)
        enpassant = self._serialize_board_enpassant(board)
        castling = self._serialize_board_castling(board)
        turn = self._serialize_board_turn(board)
        array = np.concatenate([pieces, enpassant, castling, turn], axis=0)
        return array

    def _serialize_fen(self, fen: str) -> np.ndarray:
        pieces = self._serialize_fen_pieces(fen)
        enpassant = self._serialize_fen_enpassant(fen)
        castling = self._serialize_fen_castling(fen)
        turn = self._serialize_fen_turn(fen)
        array = np.concatenate([pieces, enpassant, castling, turn], axis=0)
        return array

    def serialize(
        self,
        position: Union[chess.Board, str],
    ) -> np.ndarray:
        if isinstance(position, chess.Board):
            array = self._serialize_board(position)
        elif isinstance(position, str):
            array = self._serialize_fen(position)
        else:
            raise TypeError("position must be either str or chess.Board")

        return array


def parse_eval(y: str) -> float:
    if y[:2] == "#-":
        return -100
    elif y[0] == "#":
        return 100
    else:
        return float(y)


if __name__ == "__main__":
    from dataset import ChessValueDataset
    import numpy as np


    cvd = ChessValueDataset.from_file("cvd.json")
    s = ChessPositionSerializer()
    X = np.stack([s.serialize(fen) for fen in cvd.fen_to_value.keys()])
    y = np.stack([parse_eval(y) for y in cvd.fen_to_value.values()])
    np.savez("dataset.npz", X=X, y=y)
