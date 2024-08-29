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
use crate::vec3::Vec3;

use std::fmt;

#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct Mat3<T> (
    pub Vec3<T>,
    pub Vec3<T>,
    pub Vec3<T>
);

impl<T> Mat3<T>
where
    T: NumAssign + Copy,
{
    /// Creates a new 3x3 matrix from three columns.
    ///
    /// This constructor initializes a `Mat3` instance with the provided column vectors.
    /// Each column vector is represented as a `Vec3<T>`.
    ///
    /// # Parameters
    /// - `col0`: The first column of the matrix.
    /// - `col1`: The second column of the matrix.
    /// - `col2`: The third column of the matrix.
    ///
    /// # Returns
    /// Returns a `Mat3` instance with the specified columns.
    ///
    /// # Examples
    /// ```
    /// let col0 = Vec3::new(1.0, 0.0, 0.0);
    /// let col1 = Vec3::new(0.0, 1.0, 0.0);
    /// let col2 = Vec3::new(0.0, 0.0, 1.0);
    /// let matrix = Mat3::new(&col0, &col1, &col2);
    /// ```
    #[inline]
    pub fn new(col0: &Vec3<T>, col1: &Vec3<T>, col2: &Vec3<T>) -> Self {
        Self(col0.clone(), col1.clone(), col2.clone())
    }

    /// Creates a 3x3 zero matrix.
    ///
    /// This constructor initializes a `Mat3` instance where all elements are set to zero.
    ///
    /// # Returns
    /// Returns a `Mat3` instance where all columns are zero vectors.
    ///
    /// # Examples
    /// ```
    /// let matrix = Mat3::zero();
    /// ```
    #[inline]
    pub fn zero() -> Self {
        Self(
            Vec3::zero(),
            Vec3::zero(),
            Vec3::zero(),
        )
    }

    /// Creates an identity matrix.
    ///
    /// This constructor initializes a `Mat3` instance as the identity matrix, which has `1`s on the diagonal
    /// and `0`s elsewhere. This is useful for matrix transformations where the identity matrix represents no change.
    ///
    /// # Returns
    /// Returns a `Mat3` instance representing the identity matrix.
    ///
    /// # Examples
    /// ```
    /// let matrix = Mat3::identity();
    /// ```
    #[inline]
    pub fn identity() -> Self {
        Self(
            Vec3::new(T::one(), T::zero(), T::zero()),
            Vec3::new(T::zero(), T::one(), T::zero()),
            Vec3::new(T::zero(), T::zero(), T::one()),
        )
    }

    /// Computes the transpose of the matrix.
    ///
    /// This function returns a new `Mat3` instance where the rows and columns of the original matrix are swapped.
    ///
    /// # Returns
    /// Returns the transpose of the matrix.
    ///
    /// # Examples
    /// ```
    /// let matrix = Mat3::identity();
    /// let transposed = matrix.transpose();
    /// ```
    #[inline]
    pub fn transpose(&self) -> Self {
        Self(
            Vec3::new(self.0.x, self.1.x, self.2.x),
            Vec3::new(self.0.y, self.1.y, self.2.y),
            Vec3::new(self.0.z, self.1.z, self.2.z),
        )
    }

    /// Calculates the determinant of the matrix.
    ///
    /// This function computes the determinant of the 3x3 matrix, which can be used to determine if the matrix
    /// is invertible (a non-zero determinant indicates the matrix is invertible).
    ///
    /// # Returns
    /// Returns the determinant of the matrix.
    ///
    /// # Examples
    /// ```
    /// let matrix = Mat3::identity();
    /// let det = matrix.determinant();
    /// assert_eq!(det, 1.0); // Determinant of the identity matrix
    /// ```
    pub fn determinant(&self) -> T {
        let a00 = self.0.x;
        let a01 = self.0.y;
        let a02 = self.0.z;
        let a10 = self.1.x;
        let a11 = self.1.y;
        let a12 = self.1.z;
        let a20 = self.2.x;
        let a21 = self.2.y;
        let a22 = self.2.z;

        a00 * (a11 * a22 - a12 * a21) -
        a01 * (a10 * a22 - a12 * a20) +
        a02 * (a10 * a21 - a11 * a20)
    }

    /// Computes the trace of the matrix.
    ///
    /// The trace of a matrix is the sum of its diagonal elements. For a 3x3 matrix, it is the sum of `self.0.x`,
    /// `self.1.y`, and `self.2.z`.
    ///
    /// # Returns
    /// Returns the trace of the matrix.
    ///
    /// # Examples
    /// ```
    /// let matrix = Mat3::identity();
    /// let trace = matrix.trace();
    /// assert_eq!(trace, 3.0); // Trace of the identity matrix
    /// ```
    pub fn trace(&self) -> T {
        self.0.x + self.1.y + self.2.z
    }

    /// Multiplies the current matrix by another matrix.
    ///
    /// This function performs matrix multiplication with another `Mat3` instance. The resulting matrix is computed
    /// as the dot products of rows from the first matrix and columns from the second matrix.
    ///
    /// # Parameters
    /// - `other`: The matrix to multiply with.
    ///
    /// # Returns
    /// Returns the product of the two matrices.
    ///
    /// # Examples
    /// ```
    /// let a = Mat3::identity();
    /// let b = Mat3::identity();
    /// let product = a.mul(&b);
    /// assert_eq!(product, a); // Product of two identity matrices is an identity matrix
    /// ```
    pub fn mul(&self, other: &Self) -> Self {
        let col0 = Vec3::new(
            self.0.dot(&Vec3::new(other.0.x, other.1.x, other.2.x)),
            self.1.dot(&Vec3::new(other.0.x, other.1.x, other.2.x)),
            self.2.dot(&Vec3::new(other.0.x, other.1.x, other.2.x)),
        );
        let col1 = Vec3::new(
            self.0.dot(&Vec3::new(other.0.y, other.1.y, other.2.y)),
            self.1.dot(&Vec3::new(other.0.y, other.1.y, other.2.y)),
            self.2.dot(&Vec3::new(other.0.y, other.1.y, other.2.y)),
        );
        let col2 = Vec3::new(
            self.0.dot(&Vec3::new(other.0.z, other.1.z, other.2.z)),
            self.1.dot(&Vec3::new(other.0.z, other.1.z, other.2.z)),
            self.2.dot(&Vec3::new(other.0.z, other.1.z, other.2.z)),
        );
        Self::new(&col0, &col1, &col2)
    }

    /// Translates the matrix in 2D space.
    ///
    /// This function translates the matrix by adding the translation vector to the matrix's translation components.
    ///
    /// # Parameters
    /// - `translate`: The 2D translation vector.
    ///
    /// # Examples
    /// ```
    /// let mut matrix = Mat3::identity();
    /// let translate = Vec2::new(2.0, 3.0);
    /// matrix.translate_2d(&translate);
    /// ```
    #[inline]
    pub fn translate_2d(&mut self, translate: &Vec2<T>) {
        self.2.x += translate.x;
        self.2.y += translate.y;
    }

    /// Scales the matrix in 2D space.
    ///
    /// This function scales the matrix by multiplying the appropriate components of the matrix by the given scale vector.
    ///
    /// # Parameters
    /// - `scale`: The 2D scale vector.
    ///
    /// # Examples
    /// ```
    /// let mut matrix = Mat3::identity();
    /// let scale = Vec2::new(2.0, 3.0);
    /// matrix.scale_2d(&scale);
    /// ```
    #[inline]
    pub fn scale_2d(&mut self, scale: &Vec2<T>) {
        self.0.x *= scale.x;
        self.1.y *= scale.y;
    }

    /// Scales the matrix in 3D space.
    ///
    /// This function scales the matrix by multiplying each component of the matrix with the corresponding component
    /// of the scale vector.
    ///
    /// # Parameters
    /// - `scale`: The 3D scale vector.
    ///
    /// # Examples
    /// ```
    /// let mut matrix = Mat3::identity();
    /// let scale = Vec3::new(2.0, 3.0, 4.0);
    /// matrix.scale_3d(&scale);
    /// ```
    #[inline]
    pub fn scale_3d(&mut self, scale: &Vec3<T>) {
        self.0.x *= scale.x;
        self.1.y *= scale.y;
        self.2.z *= scale.z;
    }
}

impl<T> Mat3<T>
where
    T: NumAssign + Float
{
    /// Computes the inverse of the matrix.
    ///
    /// This method calculates the inverse of the 3x3 matrix, if it exists. The inverse of a matrix is used to 
    /// reverse the effects of the matrix transformation. If the matrix is singular (i.e., its determinant is zero),
    /// the function returns `None`.
    ///
    /// # Returns
    /// Returns `Some(Self)` containing the inverse of the matrix if the determinant is non-zero; otherwise, returns `None`.
    ///
    /// # Examples
    /// ```
    /// let matrix = Mat3::identity();
    /// let inverse = matrix.invert();
    /// assert!(inverse.is_some()); // Identity matrix is invertible
    /// ```
    pub fn invert(&self) -> Option<Self> {
        let a00 = self.0.x;
        let a01 = self.0.y;
        let a02 = self.0.z;
        let a10 = self.1.x;
        let a11 = self.1.y;
        let a12 = self.1.z;
        let a20 = self.2.x;
        let a21 = self.2.y;
        let a22 = self.2.z;

        let b01 = a22 * a11 - a12 * a21;
        let b11 = -a22 * a10 + a12 * a20;
        let b21 = a21 * a10 - a11 * a20;

        let det = a00 * b01 + a01 * b11 + a02 * b21;

        if det.is_zero() {
            return None;
        }

        let inv_det = T::one() / det;

        Some(Self(
            Vec3::new(b01 * inv_det, (-a22 * a01 + a02 * a21) * inv_det, (a12 * a01 - a02 * a11) * inv_det),
            Vec3::new(b11 * inv_det, (a22 * a00 - a02 * a20) * inv_det, (-a12 * a00 + a02 * a10) * inv_det),
            Vec3::new(b21 * inv_det, (-a21 * a00 + a01 * a20) * inv_det, (a11 * a00 - a01 * a10) * inv_det)
        ))
    }

    /// Rotates the matrix by a given angle in 2D space.
    ///
    /// This method applies a 2D rotation transformation to the matrix. The rotation is counterclockwise by the
    /// specified angle in radians. This transformation affects only the 2D components of the matrix.
    ///
    /// # Parameters
    /// - `angle`: The angle of rotation in radians.
    ///
    /// # Examples
    /// ```
    /// let mut matrix = Mat3::identity();
    /// matrix.rotate_2d(std::f64::consts::FRAC_PI_2); // Rotate by 90 degrees
    /// ```
    pub fn rotate_2d(&mut self, angle: T) {
        let c = angle.cos();
        let s = angle.sin();

        let x0 = self[0][0] * c + self[1][0] * s;
        let x1 = self[0][1] * c + self[1][1] * s;
        let x2 = self[0][2] * c + self[1][2] * s;

        let y0 = -self[0][0] * s + self[1][0] * c;
        let y1 = -self[0][1] * s + self[1][1] * c;
        let y2 = -self[0][2] * s + self[1][2] * c;

        self[0][0] = x0;
        self[0][1] = x1;
        self[0][2] = x2;

        self[1][0] = y0;
        self[1][1] = y1;
        self[1][2] = y2;
    }

    /// Rotates the matrix by a given angle around a specified 3D axis.
    ///
    /// This method applies a 3D rotation transformation to the matrix. The rotation is counterclockwise around
    /// the specified axis by the given angle in radians. The axis is normalized before applying the rotation.
    ///
    /// # Parameters
    /// - `axis`: The 3D axis around which to rotate.
    /// - `angle`: The angle of rotation in radians.
    ///
    /// # Examples
    /// ```
    /// let mut matrix = Mat3::identity();
    /// let axis = Vec3::new(0.0, 0.0, 1.0); // Rotation around the Z axis
    /// matrix.rotate_3d(axis, std::f64::consts::FRAC_PI_2); // Rotate by 90 degrees
    /// ```
    pub fn rotate_3d(&mut self, mut axis: Vec3<T>, angle: T) {
        let len_sq = axis.length_squared();
        if !(len_sq.is_one() || len_sq.is_zero()) {
            let inv_len = len_sq.sqrt();
            axis *= inv_len;
        }

        let s = angle.sin();
        let c = angle.cos();
        let t = T::one() - c;

        let rotation = Self::new(
            &Vec3::new(
                axis.x * axis.x * t + c,
                axis.y * axis.x * t + axis.z * s,
                axis.z * axis.x * t - axis.y * s,
            ),
            &Vec3::new(
                axis.x * axis.y * t - axis.z * s,
                axis.y * axis.y * t + c,
                axis.z * axis.y * t + axis.x * s,
            ),
            &Vec3::new(
                axis.x * axis.z * t + axis.y * s,
                axis.y * axis.z * t - axis.x * s,
                axis.z * axis.z * t + c,
            )
        );

        *self = self.mul(rotation);
    }

    /// Rotates the matrix by a given angle around the X-axis in 3D space.
    ///
    /// This method applies a 3D rotation transformation to the matrix around the X-axis. The rotation is
    /// counterclockwise by the specified angle in radians.
    ///
    /// # Parameters
    /// - `angle`: The angle of rotation in radians.
    ///
    /// # Examples
    /// ```
    /// let mut matrix = Mat3::identity();
    /// matrix.rotate_x_3d(std::f64::consts::FRAC_PI_2); // Rotate around X-axis by 90 degrees
    /// ```
    pub fn rotate_x_3d(&mut self, angle: T) {
        let c = angle.cos();
        let s = angle.sin();

        let y0 = self[1][0] * c + self[2][0] * s;
        let y1 = self[1][1] * c + self[2][1] * s;
        let y2 = self[1][2] * c + self[2][2] * s;

        let z0 = -self[1][0] * s + self[2][0] * c;
        let z1 = -self[1][1] * s + self[2][1] * c;
        let z2 = -self[1][2] * s + self[2][2] * c;

        self[1][0] = y0;
        self[1][1] = y1;
        self[1][2] = y2;

        self[2][0] = z0;
        self[2][1] = z1;
        self[2][2] = z2;
    }    

    /// Rotates the matrix by a given angle around the Y-axis in 3D space.
    ///
    /// This method applies a 3D rotation transformation to the matrix around the Y-axis. The rotation is
    /// counterclockwise by the specified angle in radians.
    ///
    /// # Parameters
    /// - `angle`: The angle of rotation in radians.
    ///
    /// # Examples
    /// ```
    /// let mut matrix = Mat3::identity();
    /// matrix.rotate_y_3d(std::f64::consts::FRAC_PI_2); // Rotate around Y-axis by 90 degrees
    /// ```
    pub fn rotate_y_3d(&mut self, angle: T) {
        let c = angle.cos();
        let s = angle.sin();

        let x0 = self[0][0] * c - self[2][0] * s;
        let x1 = self[0][1] * c - self[2][1] * s;
        let x2 = self[0][2] * c - self[2][2] * s;

        let z0 = self[0][0] * s + self[2][0] * c;
        let z1 = self[0][1] * s + self[2][1] * c;
        let z2 = self[0][2] * s + self[2][2] * c;

        self[0][0] = x0;
        self[0][1] = x1;
        self[0][2] = x2;

        self[2][0] = z0;
        self[2][1] = z1;
        self[2][2] = z2;
    }

    /// Rotates the matrix by a given angle around the Z-axis in 3D space.
    ///
    /// This method applies a 3D rotation transformation to the matrix around the Z-axis. The rotation is
    /// counterclockwise by the specified angle in radians.
    ///
    /// # Parameters
    /// - `angle`: The angle of rotation in radians.
    ///
    /// # Examples
    /// ```
    /// let mut matrix = Mat3::identity();
    /// matrix.rotate_z_3d(std::f64::consts::FRAC_PI_2); // Rotate around Z-axis by 90 degrees
    /// ```
    pub fn rotate_z_3d(&mut self, angle: T) {
        let c = angle.cos();
        let s = angle.sin();

        let x0 = self[0][0] * c + self[1][0] * s;
        let x1 = self[0][1] * c + self[1][1] * s;
        let x2 = self[0][2] * c + self[1][2] * s;

        let y0 = -self[0][0] * s + self[1][0] * c;
        let y1 = -self[0][1] * s + self[1][1] * c;
        let y2 = -self[0][2] * s + self[1][2] * c;

        self[0][0] = x0;
        self[0][1] = x1;
        self[0][2] = x2;

        self[1][0] = y0;
        self[1][1] = y1;
        self[1][2] = y2;
    }    
}

impl<T> fmt::Display for Mat3<T>
where
    T: fmt::Display,
{
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "[{}, {}, {}]", self.0, self.1, self.2)
    }
}

impl<T> Index<usize> for Mat3<T> {
    type Output = Vec3<T>;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.0,
            1 => &self.1,
            2 => &self.2,
            _ => panic!("Index out of bounds for Mat3"),
        }
    }
}

impl<T> IndexMut<usize> for Mat3<T> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.0,
            1 => &mut self.1,
            2 => &mut self.2,
            _ => panic!("Index out of bounds for Mat3"),
        }
    }
}

impl<'a, T> IntoIterator for &'a Mat3<T>
where
    T: NumAssign + Copy
{
    type Item = &'a Vec3<T>;
    type IntoIter = Iter<'a, Vec3<T>>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        let slice: &[Vec3<T>; 3] = unsafe { std::mem::transmute(self) };
        slice.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut Mat3<T>
where
    T: NumAssign + Copy
{
    type Item = &'a mut Vec3<T>;
    type IntoIter = IterMut<'a, Vec3<T>>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        let slice: &mut [Vec3<T>; 3] = unsafe { std::mem::transmute(self) };
        slice.iter_mut()
    }
}

impl<T> Neg for Mat3<T>
where
    T: NumAssign + Copy + Neg<Output = T>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self (
            -self.0,
            -self.1,
            -self.2
        )
    }
}

impl<T> Add<Mat3<T>> for Mat3<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self (
            self.0 + other.0,
            self.1 + other.1,
            self.2 + other.2
        )
    }
}

impl<T> Sub<Mat3<T>> for Mat3<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self (
            self.0 - other.0,
            self.1 - other.1,
            self.2 - other.2
        )
    }
}

impl<T> Mul<Mat3<T>> for Mat3<T>
where
    T: NumAssign + Copy,
{
    type Output = Mat3<T>;

    #[inline]
    fn mul(self, other: Mat3<T>) -> Mat3<T> {
        Self::mul(&self, &other)
    }
}

impl<T> Mul<T> for Mat3<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    fn mul(self, scalar: T) -> Self {
        Self (
            self.0 * scalar,
            self.1 * scalar,
            self.2 * scalar
        )
    }
}

impl<T> Div<T> for Mat3<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    fn div(self, scalar: T) -> Self {
        Self (
            self.0 / scalar,
            self.1 / scalar,
            self.2 / scalar
        )
    }
}
