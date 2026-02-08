"""Parser error types."""


class ParseError(Exception):
    """Raised when DOT source cannot be parsed."""

    def __init__(
        self, message: str, line: int | None = None, column: int | None = None
    ):
        self.line = line
        self.column = column
        super().__init__(message)
