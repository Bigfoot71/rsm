use num_traits::{
    NumAssign, Float
};

use std::slice::{
    Iter,
    IterMut
};

use std::ops::{
    Neg, Add, Sub, Mul, Div,
    AddAssign, SubAssign, MulAssign, DivAssign,
    Index, IndexMut
};

use std::fmt;

use crate::vec2::Vec2;
use crate::vec3::Vec3;
use crate::mat4::Mat4;

/// Represents a 4D vector with generic numeric components.
///
/// `Vec4` is a generic structure that represents a 4-dimensional vector with components `x`, `y`, `z`, and `w`.
/// It provides methods for vector operations such as vector arithmetic, normalization, dot product, and linear interpolation.
///
/// # Type Parameters
///
/// - `T`: The numeric type of the vector components. This type must implement traits like `Zero`, `One`, `NumAssign`, `Copy`, and others depending on the method.
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct Vec4<T> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub w: T,
}

impl<T> Vec4<T>
where
    T: NumAssign + Copy,
{
    /// Creates a new `Vec4` with the specified components.
    ///
    /// # Parameters
    /// - `x`: The x component of the vector.
    /// - `y`: The y component of the vector.
    /// - `z`: The z component of the vector.
    /// - `w`: The w component of the vector.
    ///
    /// # Returns
    /// Returns a new `Vec4` instance with the specified components.
    ///
    /// # Examples
    /// ```
    /// let vec = Vec4::new(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(vec, Vec4::new(1.0, 2.0, 3.0, 4.0)); // Vector with components (1, 2, 3, 4)
    /// ```
    #[inline]
    pub fn new(x: T, y: T, z: T, w: T) -> Self {
        Self { x, y, z, w }
    }

    /// Creates a new `Vec4` with all components set to zero.
    ///
    /// # Returns
    /// Returns a new `Vec4` instance where all components are zero.
    ///
    /// # Examples
    /// ```
    /// let vec = Vec4::zero();
    /// assert_eq!(vec, Vec4::new(0.0, 0.0, 0.0, 0.0)); // Vector with all components zero
    /// ```
    #[inline]
    pub fn zero() -> Self {
        Self::new(T::zero(), T::zero(), T::zero(), T::zero())
    }

    /// Creates a new `Vec4` with all components set to one.
    ///
    /// # Returns
    /// Returns a new `Vec4` instance where all components are one.
    ///
    /// # Examples
    /// ```
    /// let vec = Vec4::one();
    /// assert_eq!(vec, Vec4::new(1.0, 1.0, 1.0, 1.0)); // Vector with all components one
    /// ```
    #[inline]
    pub fn one() -> Self {
        Self::new(T::one(), T::one(), T::one(), T::one())
    }

    /// Creates a new `Vec4` where all components are set to the same value.
    ///
    /// # Parameters
    /// - `v`: The value to set for all components.
    ///
    /// # Returns
    /// Returns a new `Vec4` instance where all components are set to `v`.
    ///
    /// # Examples
    /// ```
    /// let vec = Vec4::set(5.0);
    /// assert_eq!(vec, Vec4::new(5.0, 5.0, 5.0, 5.0)); // Vector with all components 5
    /// ```
    #[inline]
    pub fn set(v: T) -> Self {
        Self::new(v, v, v, v)
    }

    /// Creates a `Vec4` from a `Vec2` by setting the z and w components to zero.
    ///
    /// # Parameters
    /// - `v`: The `Vec2` to convert.
    ///
    /// # Returns
    /// Returns a new `Vec4` instance where the x and y components are taken from `v`,
    /// and the z and w components are set to zero.
    ///
    /// # Examples
    /// ```
    /// let vec2 = Vec2::new(1.0, 2.0);
    /// let vec4 = Vec4::from_vec2(&vec2);
    /// assert_eq!(vec4, Vec4::new(1.0, 2.0, 0.0, 0.0)); // Vector (1, 2, 0, 0)
    /// ```
    #[inline]
    pub fn from_vec2(v: &Vec2<T>) -> Self {
        Self::new(v.x, v.y, T::zero(), T::zero())
    }

    /// Creates a `Vec4` from a `Vec3` by setting the w component to zero.
    ///
    /// # Parameters
    /// - `v`: The `Vec3` to convert.
    ///
    /// # Returns
    /// Returns a new `Vec4` instance where the x, y, and z components are taken from `v`,
    /// and the w component is set to zero.
    ///
    /// # Examples
    /// ```
    /// let vec3 = Vec3::new(1.0, 2.0, 3.0);
    /// let vec4 = Vec4::from_vec3(&vec3);
    /// assert_eq!(vec4, Vec4::new(1.0, 2.0, 3.0, 0.0)); // Vector (1, 2, 3, 0)
    /// ```
    #[inline]
    pub fn from_vec3(v: &Vec3<T>) -> Self {
        Self::new(v.x, v.y, v.z, T::zero())
    }

    /// Computes the dot product of this vector and another vector.
    ///
    /// # Parameters
    /// - `other`: The vector to compute the dot product with.
    ///
    /// # Returns
    /// Returns the dot product of `self` and `other` as a value of type `T`.
    ///
    /// # Examples
    /// ```
    /// let vec1 = Vec4::new(1.0, 2.0, 3.0, 4.0);
    /// let vec2 = Vec4::new(5.0, 6.0, 7.0, 8.0);
    /// let dot_product = vec1.dot(&vec2);
    /// assert_eq!(dot_product, 70.0); // Dot product calculation
    /// ```
    #[inline]
    pub fn dot(&self, other: &Self) -> T {
        self.x * other.x +
        self.y * other.y +
        self.z * other.z +
        self.w * other.w
    }

    /// Computes the squared length (magnitude) of this vector.
    ///
    /// # Returns
    /// Returns the squared length of the vector as a value of type `T`.
    ///
    /// # Examples
    /// ```
    /// let vec = Vec4::new(1.0, 2.0, 3.0, 4.0);
    /// let length_squared = vec.length_squared();
    /// assert_eq!(length_squared, 30.0); // Squared length calculation
    /// ```
    #[inline]
    pub fn length_squared(&self) -> T {
        self.x * self.x +
        self.y * self.y +
        self.z * self.z +
        self.w * self.w
    }

    /// Applies a transformation matrix to this vector.
    ///
    /// The vector is transformed using a 4x4 matrix.
    ///
    /// # Parameters
    /// - `transform`: The transformation matrix to apply.
    ///
    /// # Returns
    /// Returns a new `Vec4` instance that is the result of transforming `self` using `transform`.
    ///
    /// # Examples
    /// ```
    /// let vec = Vec4::new(1.0, 2.0, 3.0, 1.0);
    /// let transform = Mat4::identity(); // Assuming an identity matrix
    /// let transformed_vec = vec.transform(&transform);
    /// assert_eq!(transformed_vec, Vec4::new(1.0, 2.0, 3.0, 1.0)); // No change for identity matrix
    /// ```
    #[inline]
    pub fn transform(&self, transform: &Mat4<T>) -> Self {
        let x = transform.0.x * self.x + transform.1.x * self.y + transform.2.x * self.z + transform.3.x * self.w;
        let y = transform.0.y * self.x + transform.1.y * self.y + transform.2.y * self.z + transform.3.y * self.w;
        let z = transform.0.z * self.x + transform.1.z * self.y + transform.2.z * self.z + transform.3.z * self.w;
        let w = transform.0.w * self.x + transform.1.w * self.y + transform.2.w * self.z + transform.3.w * self.w;
        Self::new(x, y, z, w)
    }
}

impl<T> Vec4<T>
where
    T: NumAssign + Copy + PartialOrd,
{
    /// Computes the component-wise minimum of this vector and another vector.
    ///
    /// For each component, this method returns the minimum of the corresponding
    /// components of `self` and `other`.
    ///
    /// # Parameters
    /// - `other`: The vector to compare against.
    ///
    /// # Returns
    /// Returns a new `Vec4` instance where each component is the minimum
    /// of the corresponding components of `self` and `other`.
    ///
    /// # Examples
    /// ```
    /// let vec1 = Vec4::new(1.0, 2.0, 3.0, 4.0);
    /// let vec2 = Vec4::new(2.0, 1.0, 4.0, 3.0);
    /// let min_vec = vec1.min(&vec2);
    /// assert_eq!(min_vec, Vec4::new(1.0, 1.0, 3.0, 3.0)); // Minimum component-wise
    /// ```
    #[inline]
    pub fn min(&self, other: &Self) -> Self {
        Self::new(
            if self.x < other.x { self.x } else { other.x },
            if self.y < other.y { self.y } else { other.y },
            if self.z < other.z { self.z } else { other.z },
            if self.w < other.w { self.w } else { other.w }
        )
    }

    /// Computes the component-wise maximum of this vector and another vector.
    ///
    /// For each component, this method returns the maximum of the corresponding
    /// components of `self` and `other`.
    ///
    /// # Parameters
    /// - `other`: The vector to compare against.
    ///
    /// # Returns
    /// Returns a new `Vec4` instance where each component is the maximum
    /// of the corresponding components of `self` and `other`.
    ///
    /// # Examples
    /// ```
    /// let vec1 = Vec4::new(1.0, 2.0, 3.0, 4.0);
    /// let vec2 = Vec4::new(2.0, 1.0, 4.0, 3.0);
    /// let max_vec = vec1.max(&vec2);
    /// assert_eq!(max_vec, Vec4::new(2.0, 2.0, 4.0, 4.0)); // Maximum component-wise
    /// ```
    #[inline]
    pub fn max(&self, other: &Self) -> Self {
        Self::new(
            if self.x > other.x { self.x } else { other.x },
            if self.y > other.y { self.y } else { other.y },
            if self.z > other.z { self.z } else { other.z },
            if self.w > other.w { self.w } else { other.w }
        )
    }

    /// Clamps each component of this vector between the corresponding components of
    /// the given `min` and `max` vectors.
    ///
    /// For each component, this method ensures that the component of `self` is
    /// at least the corresponding component of `min` and at most the corresponding
    /// component of `max`.
    ///
    /// # Parameters
    /// - `min`: The vector representing the minimum bounds.
    /// - `max`: The vector representing the maximum bounds.
    ///
    /// # Returns
    /// Returns a new `Vec4` instance where each component is clamped between the
    /// corresponding components of `min` and `max`.
    ///
    /// # Examples
    /// ```
    /// let vec = Vec4::new(5.0, -3.0, 10.0, 2.0);
    /// let min_vec = Vec4::new(0.0, 0.0, 5.0, 1.0);
    /// let max_vec = Vec4::new(4.0, 2.0, 8.0, 3.0);
    /// let clamped_vec = vec.clamp(&min_vec, &max_vec);
    /// assert_eq!(clamped_vec, Vec4::new(4.0, 0.0, 8.0, 2.0)); // Clamped component-wise
    /// ```
    #[inline]
    pub fn clamp(&self, min: &Self, max: &Self) -> Self {
        Self::new(
            if self.x < min.x { min.x } else if self.x > max.x { max.x } else { self.x },
            if self.y < min.y { min.y } else if self.y > max.y { max.y } else { self.y },
            if self.z < min.z { min.z } else if self.z > max.z { max.z } else { self.z },
            if self.w < min.w { min.w } else if self.w > max.w { max.w } else { self.w }
        )
    }
}

impl<T> Vec4<T>
where
    T: NumAssign + Float,
{
    /// Computes the length (magnitude) of this `Vec4`.
    ///
    /// This method calculates the Euclidean length of the vector using the formula:
    /// `sqrt(x^2 + y^2 + z^2 + w^2)`.
    ///
    /// # Returns
    /// Returns the length of the vector as a `T`.
    ///
    /// # Examples
    /// ```
    /// let vec = Vec4::new(3.0, 4.0, 0.0, 0.0);
    /// let length = vec.length();
    /// assert_eq!(length, 5.0); // Euclidean length of the vector (3, 4, 0, 0)
    /// ```
    #[inline]
    pub fn length(&self) -> T {
        (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt()
    }

    /// Normalizes the vector, scaling it to a unit vector.
    ///
    /// This method returns a new `Vec4` with the same direction as `self` but with a length of 1.
    /// If the vector length is zero, meaning it's a zero vector, normalization is not possible,
    /// and `None` is returned to indicate this failure.
    ///
    /// # Returns
    /// - `Some(Self)`: A unit vector in the same direction as `self` if normalization is possible.
    /// - `None`: If `self` is a zero vector (length is zero), normalization is not possible.
    ///
    /// # Examples
    /// ```
    /// let vec = Vec4::new(1.0, 2.0, 2.0, 0.0);
    /// let normalized = vec.normalize();
    /// assert_eq!(normalized, Some(Vec4::new(0.33333334, 0.6666667, 0.6666667, 0.0)));
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

    /// Performs linear interpolation between this vector and another vector.
    ///
    /// This method returns a vector that is a linear interpolation between `self` and `other`
    /// by a factor `t`. If `t` is 0, the result is `self`. If `t` is 1, the result is `other`.
    /// Values of `t` between 0 and 1 will yield a result that is a blend of the two vectors.
    ///
    /// # Parameters
    /// - `other`: The target vector to interpolate towards.
    /// - `t`: The interpolation factor, where `t` is between 0 and 1.
    ///
    /// # Returns
    /// Returns a new `Vec4` that is the result of the interpolation.
    ///
    /// # Examples
    /// ```
    /// let vec1 = Vec4::new(1.0, 2.0, 3.0, 4.0);
    /// let vec2 = Vec4::new(5.0, 6.0, 7.0, 8.0);
    /// let t = 0.5;
    /// let interpolated = vec1.lerp(&vec2, t);
    /// assert_eq!(interpolated, Vec4::new(3.0, 4.0, 5.0, 6.0)); // Midpoint between vec1 and vec2
    /// ```
    #[inline]
    pub fn lerp(&self, other: &Self, t: T) -> Self {
        Self::new(
            self.x + t * (other.x - self.x),
            self.y + t * (other.y - self.y),
            self.z + t * (other.z - self.z),
            self.w + t * (other.w - self.w),
        )
    }
}

impl<T> fmt::Display for Vec4<T>
where
    T: fmt::Display,
{
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {}, {})", self.x, self.y, self.z, self.w)
    }
}

impl<T> From<(T, T, T, T)> for Vec4<T>
where
    T: NumAssign + Copy,
{
    #[inline]
    fn from(tuple: (T, T, T, T)) -> Self {
        Vec4::new(tuple.0, tuple.1, tuple.2, tuple.3)
    }
}

impl<T> Into<(T, T, T, T)> for Vec4<T>
where
    T: NumAssign + Copy,
{
    #[inline]
    fn into(self) -> (T, T, T, T) {
        (self.x, self.y, self.z, self.w)
    }
}

impl<T> Index<usize> for Vec4<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            3 => &self.w,
            _ => panic!("Index out of bounds for Vec4"),
        }
    }
}

impl<T> IndexMut<usize> for Vec4<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            3 => &mut self.w,
            _ => panic!("Index out of bounds for Vec4"),
        }
    }
}

impl<'a, T> IntoIterator for &'a Vec4<T>
where
    T: NumAssign + Copy
{
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        let slice: &[T; 4] = unsafe { std::mem::transmute(self) };
        slice.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut Vec4<T>
where
    T: NumAssign + Copy
{
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        let slice: &mut [T; 4] = unsafe { std::mem::transmute(self) };
        slice.iter_mut()
    }
}

impl<T> Neg for Vec4<T>
where
    T: NumAssign + Copy + Neg<Output = T>,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y, -self.z, -self.w)
    }
}

impl<T> Add for Vec4<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Self::new(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
            self.w + other.w)
    }
}

impl<T> Add<T> for Vec4<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    #[inline]
    fn add(self, scalar: T) -> Self {
        Self::new(
            self.x + scalar,
            self.y + scalar,
            self.z + scalar,
            self.w + scalar)
    }
}

impl<T> Sub for Vec4<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self::new(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
            self.w - other.w)
    }
}

impl<T> Sub<T> for Vec4<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    #[inline]
    fn sub(self, scalar: T) -> Self {
        Self::new(
            self.x - scalar,
            self.y - scalar,
            self.z - scalar,
            self.w - scalar)
    }
}

impl<T> Mul for Vec4<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    #[inline]
    fn mul(self, other: Self) -> Self {
        Self::new(
            self.x * other.x,
            self.y * other.y,
            self.z * other.z,
            self.w * other.w)
    }
}

impl<T> Mul<T> for Vec4<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    #[inline]
    fn mul(self, scalar: T) -> Self {
        Self::new(
            self.x * scalar,
            self.y * scalar,
            self.z * scalar,
            self.w * scalar)
    }
}

impl<T> Div for Vec4<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    #[inline]
    fn div(self, other: Self) -> Self {
        Self::new(
            self.x / other.x,
            self.y / other.y,
            self.z / other.z,
            self.w / other.w)
    }
}

impl<T> Div<T> for Vec4<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    #[inline]
    fn div(self, scalar: T) -> Self {
        Self::new(
            self.x / scalar,
            self.y / scalar,
            self.z / scalar,
            self.w / scalar)
    }
}

impl<T> AddAssign for Vec4<T>
where
    T: NumAssign + Copy,
{
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
        self.w += other.w;
    }
}

impl<T> AddAssign<T> for Vec4<T>
where
    T: NumAssign + Copy,
{
    #[inline]
    fn add_assign(&mut self, scalar: T) {
        self.x += scalar;
        self.y += scalar;
        self.z += scalar;
        self.w += scalar;
    }
}

impl<T> SubAssign for Vec4<T>
where
    T: NumAssign + Copy,
{
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
        self.w -= other.w;
    }
}

impl<T> SubAssign<T> for Vec4<T>
where
    T: NumAssign + Copy,
{
    #[inline]
    fn sub_assign(&mut self, scalar: T) {
        self.x -= scalar;
        self.y -= scalar;
        self.z -= scalar;
        self.w -= scalar;
    }
}

impl<T> MulAssign for Vec4<T>
where
    T: NumAssign + Copy,
{
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        self.x *= other.x;
        self.y *= other.y;
        self.z *= other.z;
        self.w *= other.w;
    }
}

impl<T> MulAssign<T> for Vec4<T>
where
    T: NumAssign + Copy,
{
    #[inline]
    fn mul_assign(&mut self, scalar: T) {
        self.x *= scalar;
        self.y *= scalar;
        self.z *= scalar;
        self.w *= scalar;
    }
}

impl<T> DivAssign for Vec4<T>
where
    T: NumAssign + Copy,
{
    #[inline]
    fn div_assign(&mut self, other: Self) {
        self.x /= other.x;
        self.y /= other.y;
        self.z /= other.z;
        self.w /= other.w;
    }
}

impl<T> DivAssign<T> for Vec4<T>
where
    T: NumAssign + Copy,
{
    #[inline]
    fn div_assign(&mut self, scalar: T) {
        self.x /= scalar;
        self.y /= scalar;
        self.z /= scalar;
        self.w /= scalar;
    }
}
