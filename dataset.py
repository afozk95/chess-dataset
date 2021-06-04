from typing import Dict, Generator, Iterable, Literal, Optional, Union
from io import StringIO
from pathlib import Path
import re
import chess
import chess.pgn


class ChessValueDataset:
    def __init__(self) -> None:
        self.fen_to_value: Dict[str, str] = {}

    def _read_pgn(self, path: Union[Path, str]) -> str:
        with open(path, mode="r") as f:
            raw = f.read()
        return raw

    def _generate_games(self, pgn: str) -> Generator[chess.pgn.Game, None, None]:
        pgn = StringIO(pgn)
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            yield game

    def _insert_game(self, game: chess.pgn.Game, eval_regex: re.Pattern) -> None:
        for node in game.mainline():
            fen = node.board().fen()
            if fen not in self.fen_to_value:
                eval_match = eval_regex.search(node.comment)
                if eval_match:
                    evaluation = eval_match[1]
                    self.fen_to_value[fen] = evaluation

    def _insert_games(self, games: Iterable[chess.pgn.Game], eval_regex_pattern: str, verbose: bool = False) -> None:
        eval_regex = re.compile(eval_regex_pattern)
        for i, game in enumerate(games, start=1):
            self._insert_game(game, eval_regex)
            if verbose:
                print(f"Games {i}")
                print(f"Number of positions in dataset {len(self.fen_to_value)}")

    def insert(
        self,
        pgn_path: Optional[Union[Path, str]] = None,
        pgn: Optional[str] = None,
        eval_regex_pattern: Optional[str] = None,
        pgn_format: Literal["lichess", "other"] = "other",
        verbose: bool = False,
    ) -> None:
        assert pgn_path or pgn, "either pgn_path or pgn should be passed"
        if pgn_format == "other":
            assert eval_regex_pattern, "eval_regex_pattern should be passed for pgn_format = other"

        if pgn_path:
            pgn = self._read_pgn(pgn_path)

        if pgn_format == "lichess":
            games_as_str = pgn.split("\n\n\n")[:-1]
            games_with_eval_as_str = [g for g in games_as_str if "%eval" in g]
            games_with_eval_pgn = "\n\n\n".join(games_with_eval_as_str)
            games = self._generate_games(games_with_eval_pgn)
            eval_regex_pattern = r"\[%eval (.*)\]"
        elif pgn_format == "other":
            games = self._generate_games(pgn)

        self._insert_games(games, eval_regex_pattern, verbose)


if __name__ == "__main__":
    path = "data/lichess_tournament_2021.05.22_may21lta_titled-arena-may-21.pgn"
    cvd = ChessValueDataset()
    cvd.insert(pgn_path=path, pgn_format="lichess", verbose=True)

    import ipdb
    ipdb.set_trace()
