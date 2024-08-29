use num_traits::{
    NumAssign, Float
};

use std::slice::{
    Iter,
    IterMut
};

use std::ops::{
    Neg, Add, Sub, Mul, Div,
    Index, IndexMut
};

use crate::vec2::Vec2;

use std::fmt;

#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct Mat2<T> (
    pub Vec2<T>,
    pub Vec2<T>,
);

impl<T> Mat2<T>
where
    T: NumAssign + Copy,
{
    /// Creates a new 2x2 matrix from two column vectors.
    ///
    /// # Parameters
    /// - `col0`: The first column of the matrix.
    /// - `col1`: The second column of the matrix.
    ///
    /// # Returns
    /// Returns a new `Mat2` instance with the specified columns.
    ///
    /// # Examples
    /// ```
    /// let col0 = Vec2::new(1.0, 2.0);
    /// let col1 = Vec2::new(3.0, 4.0);
    /// let matrix = Mat2::new(&col0, &col1);
    /// ```
    #[inline]
    pub fn new(col0: &Vec2<T>, col1: &Vec2<T>) -> Self {
        Self(col0.clone(), col1.clone())
    }

    /// Creates a 2x2 zero matrix.
    ///
    /// This method returns a matrix with all elements set to zero.
    ///
    /// # Returns
    /// Returns a `Mat2` instance with all elements as zero.
    ///
    /// # Examples
    /// ```
    /// let matrix = Mat2::zero();
    /// ```
    #[inline]
    pub fn zero() -> Self {
        Self(
            Vec2::zero(),
            Vec2::zero(),
        )
    }

    /// Creates a 2x2 identity matrix.
    ///
    /// The identity matrix has ones on the diagonal and zeros elsewhere. It is the multiplicative identity 
    /// for matrix multiplication.
    ///
    /// # Returns
    /// Returns a `Mat2` instance with ones on the diagonal and zeros elsewhere.
    ///
    /// # Examples
    /// ```
    /// let matrix = Mat2::identity();
    /// ```
    #[inline]
    pub fn identity() -> Self {
        Self(
            Vec2::new(T::one(), T::zero()),
            Vec2::new(T::zero(), T::one()),
        )
    }

    /// Transposes the matrix.
    ///
    /// The transpose of a matrix is obtained by swapping its rows and columns.
    ///
    /// # Returns
    /// Returns a new `Mat2` instance that is the transpose of the original matrix.
    ///
    /// # Examples
    /// ```
    /// let matrix = Mat2::identity();
    /// let transposed = matrix.transpose();
    /// ```
    #[inline]
    pub fn transpose(&self) -> Self {
        Self(
            Vec2::new(self.0.x, self.1.x),
            Vec2::new(self.0.y, self.1.y),
        )
    }

    /// Computes the determinant of the matrix.
    ///
    /// The determinant is a scalar value that can be used to determine if the matrix is invertible. 
    /// For a 2x2 matrix, the determinant is computed as:
    ///
    /// `det = (a00 * a11 - a01 * a10)`
    ///
    /// # Returns
    /// Returns the determinant of the matrix.
    ///
    /// # Examples
    /// ```
    /// let matrix = Mat2::new(&Vec2::new(1.0, 2.0), &Vec2::new(3.0, 4.0));
    /// let det = matrix.determinant();
    /// assert_eq!(det, -2.0);
    /// ```
    pub fn determinant(&self) -> T {
        self.0.x * self.1.y - self.0.y * self.1.x
    }

    /// Computes the trace of the matrix.
    ///
    /// The trace of a matrix is the sum of its diagonal elements. For a 2x2 matrix, the trace is computed as:
    ///
    /// `trace = a00 + a11`
    ///
    /// # Returns
    /// Returns the trace of the matrix.
    ///
    /// # Examples
    /// ```
    /// let matrix = Mat2::new(&Vec2::new(1.0, 2.0), &Vec2::new(3.0, 4.0));
    /// let trace = matrix.trace();
    /// assert_eq!(trace, 5.0);
    /// ```
    pub fn trace(&self) -> T {
        self.0.x + self.1.y
    }

    /// Multiplies this matrix with another 2x2 matrix.
    ///
    /// Matrix multiplication is performed using the formula:
    ///
    /// ```
    /// C[0][0] = A[0][0] * B[0][0] + A[1][0] * B[0][1]
    /// C[0][1] = A[0][1] * B[0][0] + A[1][1] * B[0][1]
    /// C[1][0] = A[0][0] * B[1][0] + A[1][0] * B[1][1]
    /// C[1][1] = A[0][1] * B[1][0] + A[1][1] * B[1][1]
    /// ```
    ///
    /// # Parameters
    /// - `other`: The matrix to multiply with.
    ///
    /// # Returns
    /// Returns a new `Mat2` instance that is the result of the matrix multiplication.
    ///
    /// # Examples
    /// ```
    /// let a = Mat2::new(&Vec2::new(1.0, 2.0), &Vec2::new(3.0, 4.0));
    /// let b = Mat2::new(&Vec2::new(5.0, 6.0), &Vec2::new(7.0, 8.0));
    /// let c = a.mul(&b);
    /// assert_eq!(c.0, Vec2::new(19.0, 22.0));
    /// assert_eq!(c.1, Vec2::new(43.0, 50.0));
    /// ```
    pub fn mul(&self, other: &Self) -> Self {
        let col0 = Vec2::new(
            self.0.x * other.0.x + self.1.x * other.0.y,
            self.0.y * other.0.x + self.1.y * other.0.y,
        );
        let col1 = Vec2::new(
            self.0.x * other.1.x + self.1.x * other.1.y,
            self.0.y * other.1.x + self.1.y * other.1.y,
        );
        Self::new(&col0, &col1)
    }

    /// Scales the matrix by a given 2D vector.
    ///
    /// This method scales the matrix by multiplying its elements according to the given scale vector. 
    /// The scaling affects only the diagonal elements of the matrix.
    ///
    /// # Parameters
    /// - `scale`: The 2D vector to scale the matrix by.
    ///
    /// # Examples
    /// ```
    /// let mut matrix = Mat2::identity();
    /// matrix.scale(Vec2::new(2.0, 3.0));
    /// assert_eq!(matrix.0, Vec2::new(2.0, 0.0));
    /// assert_eq!(matrix.1, Vec2::new(0.0, 3.0));
    /// ```
    #[inline]
    pub fn scale(&mut self, scale: Vec2<T>) {
        self.0.x *= scale.x;
        self.1.y *= scale.y;
    }
}

impl<T> Mat2<T>
where
    T: NumAssign + Float,
{
    /// Applies a rotation transformation to the matrix by the specified angle.
    ///
    /// This method performs a rotation of the matrix elements by an angle, effectively rotating
    /// the coordinate system represented by the matrix. The rotation is applied using a counter-clockwise
    /// rotation matrix. The rotation is applied to the matrix as follows:
    ///
    /// ```
    /// R = [ c -s ]
    ///     [ s  c ]
    /// ```
    /// where `c = cos(angle)` and `s = sin(angle)`.
    ///
    /// # Parameters
    /// - `angle`: The angle of rotation in radians. Positive values rotate counter-clockwise,
    ///   and negative values rotate clockwise.
    ///
    /// # Examples
    /// ```
    /// let mut matrix = Mat2::identity();
    /// matrix.rotation(0.5); // Rotate by 0.5 radians
    /// assert_eq!(matrix[0][0], 0.8775825618903728); // cos(0.5)
    /// assert_eq!(matrix[0][1], -0.479425538604203); // -sin(0.5)
    /// assert_eq!(matrix[1][0], 0.479425538604203); // sin(0.5)
    /// assert_eq!(matrix[1][1], 0.8775825618903728); // cos(0.5)
    /// ```
    pub fn rotation(&mut self, angle: T) {
        let c = angle.cos();
        let s = angle.sin();

        let new_x0 = c * self[0][0] - s * self[1][0];
        let new_x1 = c * self[0][1] - s * self[1][1];

        let new_y0 = s * self[0][0] + c * self[1][0];
        let new_y1 = s * self[0][1] + c * self[1][1];

        self[0][0] = new_x0;
        self[0][1] = new_x1;

        self[1][0] = new_y0;
        self[1][1] = new_y1;
    }
}

impl<T> fmt::Display for Mat2<T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "[{}, {}]", self.0, self.1)
    }
}

impl<T> Index<usize> for Mat2<T> {
    type Output = Vec2<T>;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.0,
            1 => &self.1,
            _ => panic!("Index out of bounds for Mat2"),
        }
    }
}

impl<T> IndexMut<usize> for Mat2<T> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.0,
            1 => &mut self.1,
            _ => panic!("Index out of bounds for Mat2"),
        }
    }
}

impl<'a, T> IntoIterator for &'a Mat2<T>
where
    T: NumAssign + Copy
{
    type Item = &'a Vec2<T>;
    type IntoIter = Iter<'a, Vec2<T>>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        let slice: &[Vec2<T>; 2] = unsafe { std::mem::transmute(self) };
        slice.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut Mat2<T>
where
    T: NumAssign + Copy
{
    type Item = &'a mut Vec2<T>;
    type IntoIter = IterMut<'a, Vec2<T>>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        let slice: &mut [Vec2<T>; 2] = unsafe { std::mem::transmute(self) };
        slice.iter_mut()
    }
}

impl<T> Neg for Mat2<T>
where
    T: NumAssign + Copy + Neg<Output = T>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self (
            -self.0,
            -self.1
        )
    }
}

impl<T> Add<Mat2<T>> for Mat2<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self (
            self.0 + other.0,
            self.1 + other.1
        )
    }
}

impl<T> Sub<Mat2<T>> for Mat2<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self (
            self.0 - other.0,
            self.1 - other.1
        )
    }
}

impl<T> Mul<Mat2<T>> for Mat2<T>
where
    T: NumAssign + Copy,
{
    type Output = Mat2<T>;

    #[inline]
    fn mul(self, other: Mat2<T>) -> Mat2<T> {
        Self::mul(&self, &other)
    }
}

impl<T> Mul<T> for Mat2<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    fn mul(self, scalar: T) -> Self {
        Self (
            self.0 * scalar,
            self.1 * scalar
        )
    }
}

impl<T> Div<T> for Mat2<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    fn div(self, scalar: T) -> Self {
        Self (
            self.0 / scalar,
            self.1 / scalar
        )
    }
}
