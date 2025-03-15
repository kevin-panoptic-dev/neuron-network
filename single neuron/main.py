from __future__ import annotations
from vector import Vector
from matrix import Matrix
import random

random.seed(0)


# you may curious why we still need a dot function at the first place
# actually, packages like jax and numpy implement the __mul__ method differently
# if you just do a * b for jnp arrays, it will do element wise multiplication instead of dot product
def dot(a: Matrix | Vector, b: Vector):
    return a * b


def main():
    print("This is a simple neuron network.")

    number_of_row = input("Enter the number of rows of the matrix: ")
    number_of_column = input("Enter the number of columns of the matrix: ")
    matrix = Matrix(
        [
            [random.random() for _ in range(int(number_of_column))]
            for _ in range(int(number_of_row))
        ]
    )
    vector = Vector([random.random() for _ in range(int(number_of_column))])
    print(f"The layer output is {dot(matrix, vector) + random.random()}")


if __name__ == "__main__":
    main()
