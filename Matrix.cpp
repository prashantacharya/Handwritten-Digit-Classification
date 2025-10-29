// Copyright (C) 2025 acharyp@miamioh.edu

#ifndef MATRIX_CPP
#define MATRIX_CPP

#include <cassert>
#include <vector>
#include "Matrix.h"

Matrix::Matrix(const size_t row, const size_t col, const Val initVal) :
    data(row * col, initVal), rows(row), cols(col) {
}

// Operator to write the matrix to a given output stream
std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
    // Print the number of rows and columns to ease reading
    os << matrix.height() << " " << matrix.width() << '\n';
    // Print each entry to the output stream.
    for (size_t row = 0; row < matrix.rows; row++) {
        for (size_t col = 0; col < matrix.cols; col++) {
            os << matrix.data[row * matrix.cols + col] << " ";
        }
        os << '\n';
    }

    return os;
}

// Operator to read the matrix to a given input stream.
std::istream& operator>>(std::istream& is, Matrix& matrix) {
    // Temporary variables to load matrix sizes
    int height, width;
    is >> height >> width;
    // Now initialize the destination matrix to ensure it is of the
    // correct dimension.
    matrix = Matrix(height, width);
    // Read each entry from the input stream.
    for (size_t row = 0; row < matrix.rows; row++) {
        for (size_t col = 0; col < matrix.cols; col++) {
            is >> matrix.data[row * matrix.cols + col];
        }
    }

    return is;
}

Matrix Matrix::dot(const Matrix& rhs) const {
    // Ensure the dimensions are compatible for matrix multiplication.
    assert(cols == rhs.rows);
    // Setup the result matrix
    const auto mWidth = rhs.cols, width = cols;
    Matrix result(rows, mWidth);
    // Do the actual matrix multiplication

    const auto transposed_rhs = rhs.transpose();

    for (size_t row = 0; (row < rows); row++) {
        for (size_t col = 0; (col < mWidth); col++) {
            result.data[row * mWidth + col] = 0;
            for (size_t i = 0; (i < width); i++) {
                result.data[row * mWidth + col] += data[row * width + i] * 
                    transposed_rhs.data[col * width + i];
            }
        }
    }
    // Return the computed result
    return result;
}

Matrix Matrix::transpose() const {
    // If the matrix is empty, then there is nothing much to do.
    if (rows == 0 || cols == 0) {
        return *this;
    }

    // Create a result matrix that will be the transpose, with width
    // and height flipped.
    Matrix result(width(), height());
    // Now copy the values creating the transpose
    for (int row = 0; (row < height()); row++) {
        for (int col = 0; (col < width()); col++) {
            result.data[col * rows + row] = data[row * cols + col];
        }
    }
    // Return the resulting transpose.
    return result;
}

#endif
