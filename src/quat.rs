use num_traits::{
    NumAssign, Float
};

use std::fmt;

use std::ops::{
    Add, Sub, Mul, Div,
    AddAssign, SubAssign, MulAssign, DivAssign,
    Index, IndexMut
};

use crate::vec4::Vec4;
use crate::mat4::Mat4;

/// Represents a quaternion with generic numeric components.
///
/// `Quat` is a generic struct representing a quaternion with components `x`, `y`, `z`, and `w`.
/// It provides methods for quaternion operations such as multiplication, inversion, and normalization.
///
/// # Type Parameters
///
/// - `T`: The numeric type of the quaternion components. This type must implement traits such as `Float`, `NumAssign`, `Copy`, etc.
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct Quat<T> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub w: T,
}

impl<T> Quat<T>
where
    T: NumAssign + Float + Copy,
{
    /// Creates a new `Quat` instance with the given components.
    ///
    /// # Parameters
    /// - `x`, `y`, `z`, `w`: The components of the quaternion.
    ///
    /// # Returns
    /// Returns a new `Quat` with the specified components.
    ///
    /// # Examples
    /// ```
    /// let q = Quat::new(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(q, Quat::new(1.0, 2.0, 3.0, 4.0));
    /// ```
    #[inline]
    pub fn new(x: T, y: T, z: T, w: T) -> Self {
        Self { x, y, z, w }
    }

    /// Returns the zero quaternion (0, 0, 0, 0).
    ///
    /// # Returns
    /// Returns a new `Quat` where all components are zero.
    ///
    /// # Examples
    /// ```
    /// let q = Quat::zero();
    /// assert_eq!(q, Quat::new(0.0, 0.0, 0.0, 0.0));
    /// ```
    #[inline]
    pub fn zero() -> Self {
        Self::new(T::zero(), T::zero(), T::zero(), T::zero())
    }

    /// Returns the quaternion (1, 1, 1, 1).
    ///
    /// # Returns
    /// Returns a new `Quat` where all components are one.
    ///
    /// # Examples
    /// ```
    /// let q = Quat::one();
    /// assert_eq!(q, Quat::new(1.0, 1.0, 1.0, 1.0));
    /// ```
    #[inline]
    pub fn one() -> Self {
        Self::new(T::one(), T::one(), T::one(), T::one())
    }

    /// Creates a quaternion from Euler angles (roll, pitch, yaw).
    ///
    /// This method converts Euler angles into a quaternion representation. It assumes
    /// the order of rotations is Z-Y-X (yaw-pitch-roll).
    ///
    /// # Parameters
    /// - `roll`: Rotation around the X axis.
    /// - `pitch`: Rotation around the Y axis.
    /// - `yaw`: Rotation around the Z axis.
    ///
    /// # Returns
    /// Returns a new `Quat` representing the rotation.
    ///
    /// # Examples
    /// ```
    /// let q = Quat::from_euler(0.0, std::f64::consts::PI / 2.0, 0.0);
    /// ```
    #[inline]
    pub fn from_euler(roll: T, pitch: T, yaw: T) -> Self {
        let two = T::from(2.0).unwrap();

        let half_roll = roll / two;
        let half_pitch = pitch / two;
        let half_yaw = yaw / two;

        let (sin_roll, cos_roll) = half_roll.sin_cos();
        let (sin_pitch, cos_pitch) = half_pitch.sin_cos();
        let (sin_yaw, cos_yaw) = half_yaw.sin_cos();

        Self::new(
            cos_yaw * sin_pitch * cos_roll + sin_yaw * cos_pitch * sin_roll,
            sin_yaw * cos_pitch * cos_roll - cos_yaw * sin_pitch * sin_roll,
            cos_yaw * cos_pitch * sin_roll - sin_yaw * sin_pitch * cos_roll,
            cos_yaw * cos_pitch * cos_roll + sin_yaw * sin_pitch * sin_roll,
        )
    }

    /// Creates a quaternion from a `Vec4` by using its components as the quaternion's
    /// (x, y, z, w) respectively.
    ///
    /// # Parameters
    /// - `v`: A `Vec4` instance representing the quaternion components.
    ///
    /// # Returns
    /// Returns a new `Quat` with the components taken from `v`.
    ///
    /// # Examples
    /// ```
    /// let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
    /// let q = Quat::from_vec4(&v);
    /// assert_eq!(q, Quat::new(1.0, 2.0, 3.0, 4.0));
    /// ```
    #[inline]
    pub fn from_vec4(v: &Vec4<T>) -> Self {
        Self::new(v.x, v.y, v.z, v.w)
    }

    /// Converts this quaternion to a `Vec4`.
    ///
    /// # Returns
    /// Returns a `Vec4` with the quaternion's components (x, y, z, w).
    ///
    /// # Examples
    /// ```
    /// let q = Quat::new(1.0, 2.0, 3.0, 4.0);
    /// let v = q.to_vec4();
    /// assert_eq!(v, Vec4::new(1.0, 2.0, 3.0, 4.0));
    /// ```
    #[inline]
    pub fn to_vec4(&self) -> Vec4<T> {
        Vec4::new(self.x, self.y, self.z, self.w)
    }

    /// Converts this quaternion to a 4x4 transformation matrix.
    ///
    /// # Returns
    /// Returns a `Mat4` representing the rotation described by the quaternion.
    ///
    /// # Examples
    /// ```
    /// let q = Quat::new(0.0, 0.0, 0.0, 1.0);
    /// let mat = q.to_mat4();
    /// ```
    #[inline]
    pub fn to_mat4(&self) -> Mat4<T> {
        let two = T::from(2.0).unwrap();

        let xx = self.x * self.x;
        let xy = self.x * self.y;
        let xz = self.x * self.z;
        let xw = self.x * self.w;

        let yy = self.y * self.y;
        let yz = self.y * self.z;
        let yw = self.y * self.w;

        let zz = self.z * self.z;
        let zw = self.z * self.w;

        Mat4::new(
            &Vec4::new(T::one() - two * (yy + zz), two * (xy - zw), two * (xz + yw), T::zero()),
            &Vec4::new(two * (xy + zw), T::one() - two * (xx + zz), two * (yz - xw), T::zero()),
            &Vec4::new(two * (xz - yw), two * (yz + xw), T::one() - two * (xx + yy), T::zero()),
            &Vec4::new(T::zero(), T::zero(), T::zero(), T::one()),
        )
    }

    /// Computes the dot product of this quaternion and another quaternion.
    ///
    /// # Parameters
    /// - `other`: The other quaternion to compute the dot product with.
    ///
    /// # Returns
    /// Returns the dot product as a `T`.
    ///
    /// # Examples
    /// ```
    /// let q1 = Quat::new(1.0, 0.0, 0.0, 0.0);
    /// let q2 = Quat::new(0.0, 1.0, 0.0, 0.0);
    /// let dot_product = q1.dot(&q2);
    /// assert_eq!(dot_product, 0.0); // Dot product of orthogonal quaternions
    /// ```
    #[inline]
    pub fn dot(&self, other: &Self) -> T {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }

    /// Computes the squared length of the quaternion.
    ///
    /// # Returns
    /// Returns the squared length as a `T`.
    ///
    /// # Examples
    /// ```
    /// let q = Quat::new(1.0, 2.0, 3.0, 4.0);
    /// let length_sq = q.length_squared();
    /// assert_eq!(length_sq, 30.0); // Squared length of the quaternion
    /// ```
    #[inline]
    pub fn length_squared(&self) -> T {
        self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w
    }

    /// Computes the length (magnitude) of the quaternion.
    ///
    /// # Returns
    /// Returns the length as a `T`.
    ///
    /// # Examples
    /// ```
    /// let q = Quat::new(1.0, 2.0, 3.0, 4.0);
    /// let length = q.length();
    /// assert_eq!(length, 5.4772256); // Length of the quaternion
    /// ```
    #[inline]
    pub fn length(&self) -> T {
        (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt()
    }

    /// Normalizes the quaternion to make its length equal to 1.
    ///
    /// If the quaternion is zero-length, normalization is not possible,
    /// and `None` is returned to indicate this failure.
    ///
    /// # Returns
    /// - `Some(Self)`: The normalized quaternion if the length is non-zero.
    /// - `None`: If the quaternion is zero-length.
    ///
    /// # Examples
    /// ```
    /// let q = Quat::new(1.0, 2.0, 3.0, 4.0);
    /// let normalized = q.normalize();
    /// assert_eq!(normalized, Some(Quat::new(0.1825742, 0.36514837, 0.5477225, 0.7302967)));
    /// ```
    #[inline]
    pub fn normalize(&self) -> Option<Self> {
        let len = self.length();
        if len.is_zero() {
            None
        } else {
            Some(Self::new(self.x / len, self.y / len, self.z / len, self.w / len))
        }
    }

    /// Computes the conjugate of the quaternion.
    ///
    /// The conjugate of a quaternion is obtained by negating the x, y, and z components
    /// while keeping the w component unchanged.
    ///
    /// # Returns
    /// Returns the conjugate of this quaternion.
    ///
    /// # Examples
    /// ```
    /// let q = Quat::new(1.0, 2.0, 3.0, 4.0);
    /// let conjugate = q.conjugate();
    /// assert_eq!(conjugate, Quat::new(-1.0, -2.0, -3.0, 4.0));
    /// ```
    #[inline]
    pub fn conjugate(&self) -> Self {
        Self::new(-self.x, -self.y, -self.z, self.w)
    }

    /// Computes the inverse of the quaternion.
    ///
    /// The inverse of a quaternion is computed using its conjugate and length squared.
    /// If the length squared is zero, the inverse is not defined and `None` is returned.
    ///
    /// # Returns
    /// - `Some(Self)`: The inverse of the quaternion if the length squared is non-zero.
    /// - `None`: If the length squared is zero.
    ///
    /// # Examples
    /// ```
    /// let q = Quat::new(1.0, 2.0, 3.0, 4.0);
    /// let inverse = q.inverse();
    /// assert_eq!(inverse, Some(Quat::new(-0.03333333, -0.06666667, -0.1, 0.13333334)));
    /// ```
    #[inline]
    pub fn inverse(&self) -> Option<Self> {
        let length_squared = self.length_squared();
        if length_squared.is_zero() {
            None
        } else {
            Some(self.conjugate() / length_squared)
        }
    }

    /// Multiplies this quaternion by another quaternion.
    ///
    /// This method performs quaternion multiplication, which is used to combine rotations.
    ///
    /// # Parameters
    /// - `other`: The quaternion to multiply with.
    ///
    /// # Returns
    /// Returns the resulting quaternion after multiplication.
    ///
    /// # Examples
    /// ```
    /// let q1 = Quat::new(0.0, 0.0, 0.0, 1.0);
    /// let q2 = Quat::new(0.0, 1.0, 0.0, 0.0);
    /// let result = q1.multiply(&q2);
    /// assert_eq!(result, Quat::new(-1.0, 0.0, 0.0, 0.0)); // Result of quaternion multiplication
    /// ```
    #[inline]
    pub fn multiply(&self, other: &Self) -> Self {
        Self::new(
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
        )
    }

    /// Performs spherical linear interpolation (slerp) between two quaternions.
    ///
    /// This method computes a quaternion that represents a rotation interpolated between
    /// `start` and `end` quaternions by a factor `t`. `t` ranges from 0.0 to 1.0, where
    /// 0.0 results in `start` and 1.0 results in `end`.
    ///
    /// # Parameters
    /// - `start`: The starting quaternion.
    /// - `end`: The ending quaternion.
    /// - `t`: The interpolation factor, between 0.0 and 1.0.
    ///
    /// # Returns
    /// Returns the interpolated quaternion.
    ///
    /// # Examples
    /// ```
    /// let q1 = Quat::new(1.0, 0.0, 0.0, 0.0);
    /// let q2 = Quat::new(0.0, 1.0, 0.0, 0.0);
    /// let interpolated = Quat::slerp(&q1, &q2, 0.5);
    /// assert_eq!(interpolated, Quat::new(0.5, 0.5, 0.0, 0.5)); // Interpolated quaternion
    /// ```
    #[inline]
    pub fn slerp(start: &Self, end: &Self, t: T) -> Self {
        let dot = start.dot(end);
        let dot = dot.clamp(T::from(-1.0).unwrap(), T::from(1.0).unwrap());
        let theta_0 = (T::one() - dot * dot).sqrt().acos();
        let theta = theta_0 * t;

        let (sin_theta, cos_theta) = theta.sin_cos();
        let sin_theta_0 = theta_0.sin();

        let t0 = cos_theta - dot * sin_theta / sin_theta_0;
        let t1 = sin_theta / sin_theta_0;

        Self::new(
            t0 * start.x + t1 * end.x,
            t0 * start.y + t1 * end.y,
            t0 * start.z + t1 * end.z,
            t0 * start.w + t1 * end.w,
        )
    }
}

impl<T> fmt::Display for Quat<T>
where
    T: fmt::Display,
{
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {}, {})", self.x, self.y, self.z, self.w)
    }
}

impl<T> Index<usize> for Quat<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            3 => &self.w,
            _ => panic!("Index out of bounds for Quat"),
        }
    }
}

impl<T> IndexMut<usize> for Quat<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            3 => &mut self.w,
            _ => panic!("Index out of bounds for Quat"),
        }
    }
}

impl<T> Add for Quat<T>
where
    T: NumAssign + Float + Copy,
{
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Self::new(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
            self.w + other.w,
        )
    }
}

impl<T> Add<T> for Quat<T>
where
    T: NumAssign + Float + Copy,
{
    type Output = Self;

    #[inline]
    fn add(self, scalar: T) -> Self {
        Self::new(
            self.x + scalar,
            self.y + scalar,
            self.z + scalar,
            self.w + scalar,
        )
    }
}

impl<T> Sub for Quat<T>
where
    T: NumAssign + Float + Copy,
{
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self::new(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
            self.w - other.w,
        )
    }
}

impl<T> Sub<T> for Quat<T>
where
    T: NumAssign + Float + Copy,
{
    type Output = Self;

    #[inline]
    fn sub(self, scalar: T) -> Self {
        Self::new(
            self.x - scalar,
            self.y - scalar,
            self.z - scalar,
            self.w - scalar,
        )
    }
}

impl<T> Mul for Quat<T>
where
    T: NumAssign + Float + Copy,
{
    type Output = Self;

    #[inline]
    fn mul(self, other: Self) -> Self {
        self.multiply(&other)
    }
}

impl<T> Mul<T> for Quat<T>
where
    T: NumAssign + Float + Copy,
{
    type Output = Self;

    #[inline]
    fn mul(self, scalar: T) -> Self {
        Self::new(
            self.x * scalar,
            self.y * scalar,
            self.z * scalar,
            self.w * scalar,
        )
    }
}

impl<T> Div for Quat<T>
where
    T: NumAssign + Float + Copy,
{
    type Output = Self;

    #[inline]
    fn div(self, other: Self) -> Self {
        self.multiply(&other.inverse().expect("Cannot divide by zero quaternion"))
    }
}

impl<T> Div<T> for Quat<T>
where
    T: NumAssign + Float + Copy,
{
    type Output = Self;

    #[inline]
    fn div(self, scalar: T) -> Self {
        Self::new(
            self.x / scalar,
            self.y / scalar,
            self.z / scalar,
            self.w / scalar,
        )
    }
}

impl<T> AddAssign for Quat<T>
where
    T: NumAssign + Float + Copy,
{
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
        self.w += other.w;
    }
}

impl<T> AddAssign<T> for Quat<T>
where
    T: NumAssign + Float + Copy,
{
    #[inline]
    fn add_assign(&mut self, scalar: T) {
        self.x += scalar;
        self.y += scalar;
        self.z += scalar;
        self.w += scalar;
    }
}

impl<T> SubAssign for Quat<T>
where
    T: NumAssign + Float + Copy,
{
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
        self.w -= other.w;
    }
}

impl<T> SubAssign<T> for Quat<T>
where
    T: NumAssign + Float + Copy,
{
    #[inline]
    fn sub_assign(&mut self, scalar: T) {
        self.x -= scalar;
        self.y -= scalar;
        self.z -= scalar;
        self.w -= scalar;
    }
}

impl<T> MulAssign for Quat<T>
where
    T: NumAssign + Float + Copy,
{
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        *self = self.multiply(&other);
    }
}

impl<T> MulAssign<T> for Quat<T>
where
    T: NumAssign + Float + Copy,
{
    #[inline]
    fn mul_assign(&mut self, scalar: T) {
        self.x *= scalar;
        self.y *= scalar;
        self.z *= scalar;
        self.w *= scalar;
    }
}

impl<T> DivAssign for Quat<T>
where
    T: NumAssign + Float + Copy,
{
    #[inline]
    fn div_assign(&mut self, other: Self) {
        *self = self.multiply(&other.inverse().expect("Cannot divide by zero quaternion"));
    }
}

impl<T> DivAssign<T> for Quat<T>
where
    T: NumAssign + Float + Copy,
{
    #[inline]
    fn div_assign(&mut self, scalar: T) {
        self.x /= scalar;
        self.y /= scalar;
        self.z /= scalar;
        self.w /= scalar;
    }
}
