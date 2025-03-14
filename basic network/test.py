import pytest
import jax.numpy as jnp
import jax
from numpy.testing import assert_array_almost_equal
from vector import Vector
from matrix import Matrix
from exception import ShapeError, ImmutableProperty, MustExist


def test_vector_initialization():
    v = Vector([1, 2, 3])
    assert v.value == [1, 2, 3]
    assert v.shape == (3,)


def test_vector_immutability():
    v = Vector([1, 2, 3])
    with pytest.raises(ImmutableProperty):
        v[0] = 5


def test_vector_deletion():
    v = Vector([1, 2, 3])
    with pytest.raises(MustExist):
        del v.value


def test_vector_addition():
    v1 = Vector([1, 2, 3])
    v2 = Vector([4, 5, 6])
    scalar = 2

    # Vector + Vector
    result = v1 + v2
    assert result.value == [5, 7, 9]

    # Vector + scalar
    result = v1 + scalar
    assert result.value == [3, 4, 5]

    # scalar + Vector
    result = scalar + v1
    assert result.value == [3, 4, 5]

    # Different shapes
    with pytest.raises(ShapeError):
        v1 + Vector([1, 2])


def test_vector_multiplication():
    v1 = Vector([1, 2, 3])
    v2 = Vector([4, 5, 6])
    m = Matrix([[1, 2, 3], [4, 5, 6]])

    # Vector * Vector (dot product)
    result = v1 * v2
    assert result == 32  # 1*4 + 2*5 + 3*6

    # Vector * Matrix
    result = v1 * m
    assert isinstance(result, Vector)
    assert result.value == [14, 32]

    # Different shapes
    with pytest.raises(ShapeError):
        _ = v1 * Vector([1, 2])


def test_matrix_initialization():
    m = Matrix([[1, 2], [3, 4]])
    assert m.value == [[1, 2], [3, 4]]
    assert m.shape == (2, 2)


def test_matrix_immutability():
    m = Matrix([[1, 2], [3, 4]])
    with pytest.raises(ImmutableProperty):
        m[0] = [5, 6]


def test_matrix_deletion():
    m = Matrix([[1, 2], [3, 4]])
    with pytest.raises(MustExist):
        del m.value


def test_matrix_vector_multiplication():
    m = Matrix([[1, 2, 3], [4, 5, 6]])
    v = Vector([1, 2, 3])

    result = m * v
    assert isinstance(result, Vector)
    assert result.value == [14, 32]

    with pytest.raises(ShapeError):
        _ = m * Vector([1, 2])


def test_matrix_shape_validation():
    # Invalid matrix (different row lengths)
    with pytest.raises(ShapeError):
        Matrix([[1, 2], [3, 4, 5]])


def test_empty_matrix():
    m = Matrix([])
    assert m.shape == (0, 0)


def test_vector_string_representation():
    v = Vector([1, 2, 3])
    assert str(v) == "[1, 2, 3]"


def test_matrix_string_representation():
    m = Matrix([[1, 2], [3, 4]])
    assert str(m) == "[[1, 2], [3, 4]]"


def test_vector_iteration():
    v = Vector([1, 2, 3])
    assert list(v) == [1, 2, 3]
    assert len(v) == 3


def test_matrix_iteration():
    m = Matrix([[1, 2], [3, 4]])
    assert list(m) == [[1, 2], [3, 4]]
    assert len(m) == 2


def test_matrix_vector_multiplication_vs_jax():
    # Set random seed for reproducibility
    key = jax.random.PRNGKey(0)

    # Generate random matrix and vector using JAX
    matrix_shape = (3, 4)
    vector_shape = (4,)

    jax_matrix = jax.random.normal(key, matrix_shape)
    key, subkey = jax.random.split(key)
    jax_vector = jax.random.normal(subkey, vector_shape)

    # Convert JAX arrays to Python lists for our custom classes
    matrix_list = jax_matrix.tolist()
    vector_list = jax_vector.tolist()

    # Create our custom Matrix and Vector
    custom_matrix = Matrix(matrix_list)
    custom_vector = Vector(vector_list)

    # Compute results using both methods
    custom_result = custom_matrix * custom_vector
    jax_result = jnp.dot(jax_matrix, jax_vector)

    # Compare results
    assert_array_almost_equal(
        custom_result.value,
        jax_result,
        decimal=5,
        err_msg="Custom matrix-vector multiplication differs from JAX implementation",
    )


def test_vector_vector_dot_product_vs_jax():
    # Set random seed for reproducibility
    key = jax.random.PRNGKey(1)

    # Generate two random vectors using JAX
    vector_shape = (5,)

    jax_vector1 = jax.random.normal(key, vector_shape)
    key, subkey = jax.random.split(key)
    jax_vector2 = jax.random.normal(subkey, vector_shape)

    # Convert JAX arrays to Python lists for our custom classes
    vector1_list = jax_vector1.tolist()
    vector2_list = jax_vector2.tolist()

    # Create our custom Vectors
    custom_vector1 = Vector(vector1_list)
    custom_vector2 = Vector(vector2_list)

    # Compute results using both methods
    custom_result = custom_vector1 * custom_vector2
    jax_result = jnp.dot(jax_vector1, jax_vector2)

    # Compare results
    assert_array_almost_equal(
        custom_result,  # type: ignore
        jax_result,
        decimal=5,
        err_msg="Custom vector dot product differs from JAX implementation",
    )
