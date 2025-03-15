from exception import MustExist, ShapeError, ImmutableProperty
from vector import Vector


class Matrix:
    def __init__(self, matrix: list):
        self._value: list = matrix

    def __getitem__(self, index: int):
        return self.value[index]

    def __setitem__(self, index: int, value: list):
        print("It's not recommended to change the value of a matrix.")
        raise ImmutableProperty(property_name="value")

    def __iter__(self):
        return iter(self.value)

    def __len__(self):
        return len(self.value)

    def __str__(self):
        return str(self.value)

    def __mul__(self, other: Vector):
        if not isinstance(other, Vector):
            raise TypeError(f"Expected Vector, got {type(other).__name__}.")
        if self.shape[1] != other.shape[0]:
            raise ShapeError("Matrix columns must match vector length.")
        return Vector(
            [
                sum(row_i * v_i for row_i, v_i in zip(row, other.value))
                for row in self.value
            ]
        )

    __rmul__ = __mul__

    @property
    def shape(self) -> tuple[int, int]:
        if not self.value or not self.value[0]:
            return (0, 0)
        return (len(self.value), len(self.value[0]))

    @property
    def value(self):
        if not isinstance(self._value, list):
            raise RuntimeError("self.value is not defined yet.")
        return self._value

    @value.setter
    def value(self, value: list):
        if not isinstance(value, list):
            raise TypeError(f"Expected list, got {type(value).__name__}.")
        if not all(isinstance(row, list) for row in value):
            raise TypeError("All rows must be lists.")
        if not all(len(row) == len(value[0]) for row in value):
            raise ShapeError("All rows must have the same length.")
        self._value = value

    @value.deleter
    def value(self):
        raise MustExist(property_name="value")


if __name__ == "__main__":
    raise Exception("This file is not meant to be run directly.")
