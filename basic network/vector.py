from __future__ import annotations
from exception import MustExist, ShapeError, ImmutableProperty
from typing import Union


class Vector:
    def __init__(self, vector: list):
        self._value: list = vector

    def __getitem__(self, index: int):
        return self.value[index]

    def __setitem__(self, index: int, value: float):
        print("It's not recommended to change the value of a vector.")
        raise ImmutableProperty(property_name="value")

    def __iter__(self):
        return iter(self.value)

    def __len__(self):
        return len(self.value)

    def __str__(self):
        return str(self.value)

    def __add__(self, other: Union["Vector", float, int]):
        if isinstance(other, Vector):
            try:
                return Vector(
                    [x + w for x, w in zip(self.value, other.value, strict=True)]
                )
            except ValueError:
                raise ShapeError("Vectors must have the same length.") from ValueError
        elif isinstance(other, (float, int)):
            return Vector([x + other for x in self.value])
        else:
            raise TypeError(
                f"Expected Vector, float, or int, got {type(other).__name__}."
            )

    __radd__ = __add__

    def __mul__(self, other) -> float | Vector:
        from matrix import Matrix  # avoid circular import

        if isinstance(other, Vector):
            try:
                return sum(x * w for x, w in zip(self.value, other.value, strict=True))
            except ValueError:
                raise ShapeError("Vectors must have the same length.") from ValueError
        elif isinstance(other, Matrix):
            if self.shape[0] != other.shape[1]:
                raise ShapeError("Vector length must match matrix columns.")
            return Vector(
                [
                    sum(self.value[i] * row[i] for i in range(len(self.value)))
                    for row in other.value
                ]
            )
        else:
            raise TypeError(f"Expected Vector or Matrix, got {type(other).__name__}.")

    def __rmul__(self, other):
        from matrix import Matrix  # avoid circular import

        if isinstance(other, Matrix):
            return other * self
        return self * other

    @property
    def shape(self) -> tuple[int]:
        return (len(self.value),)

    @property
    def value(self):
        if not isinstance(self._value, list):
            raise RuntimeError(f"self.value is not defined yet.")
        return self._value

    @value.setter
    def value(self, value: list):
        if not isinstance(value, list):
            raise TypeError(f"Expected list, got {type(value).__name__}.")
        self._value = value

    @value.deleter
    def value(self):
        raise MustExist(property_name="value")


if __name__ == "__main__":
    raise Exception("This file is not meant to be run directly.")
