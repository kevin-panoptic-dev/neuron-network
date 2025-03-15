class ImmutableProperty(Exception):
    def __init__(self, *, property_name: str):
        super().__init__(f"Property `{property}` is immutable.")


class MustExist(Exception):
    def __init__(self, *, property_name: str):
        super().__init__(f"Property `{property}` must exist.")


class ShapeError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


if __name__ == "__main__":
    raise Exception("This file is not meant to be run directly.")
