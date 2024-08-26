use num_traits::{
    NumAssign, Float
};

use std::slice::{
    Iter,
    IterMut
};

use std::ops::{
    Neg, Add, Sub, Mul, Div,
    AddAssign, SubAssign, MulAssign, DivAssign
};

use std::fmt;

use crate::vec3::Vec3;
use crate::vec4::Vec4;
use crate::mat2::Mat2;
use crate::mat3::Mat3;

/// Represents a 2D vector with generic numeric components.
///
/// `Vec2` is a generic structure that represents a 2-dimensional vector with components `x` and `y`.
/// It provides a variety of methods for vector operations, including vector arithmetic,
/// normalization, dot product, distance calculation, and linear interpolation.
///
/// # Type Parameters
///
/// - `T`: The numeric type of the vector components. This type must implement certain traits
/// such as `Zero`, `One`, `NumAssign`, `Copy`, and others depending on the method.
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct Vec2<T> {
    pub x: T,
    pub y: T,
}

impl<T> Vec2<T>
where
    T: NumAssign + Copy,
{
    /// Creates a new vector with the given `x` and `y` components.
    ///
    /// # Parameters
    ///
    /// - `x`: The x-coordinate of the vector.
    /// - `y`: The y-coordinate of the vector.
    ///
    /// # Returns
    ///
    /// A `Vec2` instance with the specified `x` and `y` components.
    ///
    /// # Example
    ///
    /// ```
    /// let vec = Vec2::new(3.0, 4.0);
    /// assert_eq!(vec.x, 3.0);
    /// assert_eq!(vec.y, 4.0);
    /// ```
    #[inline]
    pub fn new(x: T, y: T) -> Self {
        Self { x, y }
    }

    /// Returns a vector with both components set to zero.
    ///
    /// This is commonly used to initialize or reset a vector to a zero state.
    ///
    /// # Returns
    ///
    /// A `Vec2` instance where both components are zero.
    ///
    /// # Example
    ///
    /// ```
    /// let vec = Vec2::zero();
    /// assert_eq!(vec.x, 0.0);
    /// assert_eq!(vec.y, 0.0);
    /// ```
    #[inline]
    pub fn zero() -> Self {
        Self::new(T::zero(), T::zero())
    }

    /// Returns a vector with both components set to one.
    ///
    /// This is useful for initializing or scaling vectors to a unit state.
    ///
    /// # Returns
    ///
    /// A `Vec2` instance where both components are one.
    ///
    /// # Example
    ///
    /// ```
    /// let vec = Vec2::one();
    /// assert_eq!(vec.x, 1.0);
    /// assert_eq!(vec.y, 1.0);
    /// ```
    #[inline]
    pub fn one() -> Self {
        Self::new(T::one(), T::one())
    }

    /// Creates a vector with both components set to the given value `v`.
    ///
    /// # Parameters
    ///
    /// - `v`: The value to be assigned to both the x and y components of the vector.
    ///
    /// # Returns
    ///
    /// A `Vec2` instance where both components are initialized to `v`.
    ///
    /// # Example
    ///
    /// ```
    /// let vec = Vec2::set(5.0);
    /// assert_eq!(vec.x, 5.0);
    /// assert_eq!(vec.y, 5.0);
    /// ```
    #[inline]
    pub fn set(v: T) -> Self {
        Self { x: v, y: v }
    }

    /// Creates a `Vec2` from a `Vec3` by using the `x` and `y` components of the `Vec3`.
    ///
    /// This is useful when you want to convert a 3-dimensional vector to a 2-dimensional vector,
    /// discarding the `z` component.
    ///
    /// # Parameters
    ///
    /// - `v`: A reference to a `Vec3` instance from which the `x` and `y` components will be used.
    ///
    /// # Returns
    ///
    /// A `Vec2` instance with the `x` and `y` components taken from the `Vec3` instance.
    ///
    /// # Example
    ///
    /// ```
    /// let vec3 = Vec3::new(1.0, 2.0, 3.0);
    /// let vec2 = Vec2::from_vec3(&vec3);
    /// assert_eq!(vec2.x, 1.0);
    /// assert_eq!(vec2.y, 2.0);
    /// ```
    /// 
    #[inline]
    pub fn from_vec3(v: &Vec3<T>) -> Self {
        Self::new(v.x, v.y)
    }

    /// Creates a `Vec2` from a `Vec4` by using the `x` and `y` components of the `Vec4`.
    ///
    /// This is useful when you want to convert a 4-dimensional vector to a 2-dimensional vector,
    /// discarding the `z` and `w` components.
    ///
    /// # Parameters
    ///
    /// - `v`: A reference to a `Vec4` instance from which the `x` and `y` components will be used.
    ///
    /// # Returns
    ///
    /// A `Vec2` instance with the `x` and `y` components taken from the `Vec4` instance.
    ///
    /// # Example
    ///
    /// ```
    /// let vec4 = Vec4::new(1.0, 2.0, 3.0, 4.0);
    /// let vec2 = Vec2::from_vec4(&vec4);
    /// assert_eq!(vec2.x, 1.0);
    /// assert_eq!(vec2.y, 2.0);
    /// ```
    #[inline]
    pub fn from_vec4(v: &Vec4<T>) -> Self {
        Self::new(v.x, v.y)
    }

    /// Returns an iterator over the elements of the vector.
    ///
    /// This method allows you to iterate over the components of the vector
    /// in a read-only manner, providing immutable references to the elements.
    ///
    /// # Returns
    ///
    /// An iterator of type `Iter<'_, T>` over the vector's components.
    ///
    /// # Example
    ///
    /// ```
    /// let vec = Vec3::new(1.0, 2.0, 3.0);
    /// let mut sum = 0.0;
    /// for &component in vec.iter() {
    ///     sum += component;
    /// }
    /// assert_eq!(sum, 6.0);
    /// ```
    #[inline]
    pub fn iter(&self) -> Iter<'_, T> {
        let slice: &[T; 3] = unsafe { std::mem::transmute(self) };
        slice.iter()
    }

    /// Returns a mutable iterator over the elements of the vector.
    ///
    /// This method allows you to iterate over the components of the vector
    /// and modify them in place.
    ///
    /// # Returns
    ///
    /// A mutable iterator of type `IterMut<'_, T>` over the vector's components.
    ///
    /// # Example
    ///
    /// ```
    /// let mut vec = Vec3::new(1.0, 2.0, 3.0);
    /// for component in vec.iter_mut() {
    ///     *component += 1.0;
    /// }
    /// assert_eq!(vec.x, 2.0);
    /// assert_eq!(vec.y, 3.0);
    /// assert_eq!(vec.z, 4.0);
    /// ```
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        let slice: &mut [T; 3] = unsafe { std::mem::transmute(self) };
        slice.iter_mut()
    }

    /// Computes the dot product of the vector with another vector.
    ///
    /// The dot product is calculated as the sum of the products of the corresponding components of the two vectors.
    /// It is a measure of how much one vector extends in the direction of another. The result is a scalar value.
    ///
    /// # Arguments
    ///
    /// - `other`: The other vector with which to compute the dot product.
    ///
    /// # Returns
    ///
    /// The dot product of the two vectors as a value of type `T`.
    ///
    /// # Constraints
    ///
    /// - `T` must implement the `Mul` and `Add` traits to support multiplication and addition operations.
    ///
    /// # Example
    ///
    /// ```
    /// let vec1 = Vec2::new(1.0, 2.0);
    /// let vec2 = Vec2::new(3.0, 4.0);
    /// assert_eq!(vec1.dot(&vec2), 11.0);
    /// ```
    #[inline]
    pub fn dot(&self, other: &Self) -> T {
        self.x * other.x + self.y * other.y
    }

    /// Computes the squared length (or magnitude) of the vector.
    ///
    /// The squared length is calculated as the sum of the squares of the components of the vector.
    /// This is often used in computations where you need the length of the vector but want to avoid
    /// the overhead of computing the square root.
    ///
    /// # Returns
    ///
    /// The squared length of the vector as a value of type `T`. This is the result of the expression
    /// `x * x + y * y`.
    ///
    /// # Constraints
    ///
    /// - `T` must implement the `Mul` and `Add` traits to support multiplication and addition operations.
    ///
    /// # Example
    ///
    /// ```
    /// let vec = Vec2::new(3.0, 4.0);
    /// assert_eq!(vec.length_squared(), 25.0);
    /// ```
    /// 
    #[inline]
    pub fn length_squared(&self) -> T {
        self.x * self.x + self.y * self.y
    }

    /// Computes the squared distance between this vector and another vector.
    ///
    /// This method calculates the squared distance between the two vectors without computing the square root,
    /// which can be more efficient, especially when comparing distances or performing multiple distance calculations.
    ///
    /// # Arguments
    ///
    /// - `other`: The other vector to which the squared distance is calculated.
    ///
    /// # Returns
    ///
    /// The squared distance between the two vectors as a value of type `T`.
    ///
    /// # Constraints
    ///
    /// - `T` must implement the `Sub` and `Mul` traits to support subtraction and multiplication operations.
    ///
    /// # Example
    ///
    /// ```
    /// let vec1 = Vec2::new(1.0, 2.0);
    /// let vec2 = Vec2::new(4.0, 6.0);
    /// assert_eq!(vec1.distance_squared(&vec2), 25.0);
    /// ```
    #[inline]
    pub fn distance_squared(&self, other: &Self) -> T {
        (self.x - other.x) * (self.x - other.x) +
        (self.y - other.y) * (self.y - other.y)
    }

    /// Transforms a 2D vector using a 2x2 matrix.
    ///
    /// This method applies a 2D affine transformation to a `Vec2<T>` using a `Mat2<T>` matrix.
    /// The matrix multiplication is performed as follows:
    ///
    /// ```text
    /// [ x' ] = [ m00  m01 ] [ x ] = [ m00 * x + m01 * y ]
    /// [ y' ]   [ m10  m11 ] [ y ]   [ m10 * x + m11 * y ]
    /// ```
    ///
    /// where `Vec2<T>` is represented as `[x, y]` and `Mat2<T>` is represented as:
    ///
    /// ```text
    /// [ m00  m01 ]
    /// [ m10  m11 ]
    /// ```
    ///
    /// Parameters:
    ///
    /// - `transform`: A reference to a `Mat2<T>` matrix representing the 2D transformation to be applied.
    ///
    /// Returns:
    /// - A new `Vec2<T>` that is the result of transforming the original vector by the matrix.
    ///
    /// Example:
    /// ```
    /// let vec = Vec2::new(1.0, 2.0);
    /// let mat = Mat2::new(&Vec2::new(1.0, 0.0), &Vec2::new(0.0, 1.0));
    /// let transformed_vec = vec.transform_mat2(&mat);
    /// assert_eq!(transformed_vec, Vec2::new(1.0, 2.0));
    /// ```
    ///
    /// This example shows a simple case where the matrix is the identity matrix, and hence
    /// the vector remains unchanged.
    #[inline]
    pub fn transform_mat2(&self, transform: &Mat2<T>) -> Self {
        let x = transform.0.x * self.x + transform.1.x * self.y;
        let y = transform.0.y * self.x + transform.1.y * self.y;
        Self::new(x, y)
    }

    /// Transforms a 2D vector using a 3x3 matrix.
    ///
    /// This method applies a 2D affine transformation to a `Vec2<T>` using a `Mat3<T>` matrix.
    /// The matrix multiplication is performed as follows:
    ///
    /// ```text
    /// [ x' ] = [ m00  m01  m02 ] [ x ] = [ m00 * x + m01 * y + m02 ]
    /// [ y' ]   [ m10  m11  m12 ] [ y ]   [ m10 * x + m11 * y + m12 ]
    /// [ w' ]   [ 0    0    1  ] [ 1 ]   [ 1 ] // Homogeneous coordinate
    /// ```
    ///
    /// where `Vec2<T>` is represented as `[x, y]` and `Mat3<T>` is represented as:
    ///
    /// ```text
    /// [ m00  m01  m02 ]
    /// [ m10  m11  m12 ]
    /// [ 0    0    1  ]
    /// ```
    ///
    /// Parameters:
    ///
    /// - `transform`: A reference to a `Mat3<T>` matrix representing the 2D affine transformation to be applied.
    ///
    /// Returns:
    ///
    /// - A new `Vec2<T>` that is the result of transforming the original vector by the matrix.
    ///
    /// Example:
    /// ```
    /// let vec = Vec2::new(1.0, 2.0);
    /// let mat = Mat3::new(
    ///     Vec3::new(1.0, 0.0, 3.0), // Translation x
    ///     Vec3::new(0.0, 1.0, 4.0), // Translation y
    ///     Vec3::new(0.0, 0.0, 1.0)  // Homogeneous coordinate
    /// );
    /// let transformed_vec = vec.transform_mat3(&mat);
    /// assert_eq!(transformed_vec, Vec2::new(4.0, 6.0));
    /// ```
    ///
    /// In this example, the `Mat3` matrix represents a translation, moving the vector `(1.0, 2.0)`
    /// to `(4.0, 6.0)`.
    #[inline]
    pub fn transform_mat3(&self, transform: &Mat3<T>) -> Self {
        let x = transform.0.x * self.x + transform.1.x * self.y + transform.2.x;
        let y = transform.0.y * self.x + transform.1.y * self.y + transform.2.y;
        Self::new(x, y)
    }
}

impl<T> Vec2<T>
where
    T: NumAssign + Copy + PartialOrd,
{
    /// Returns a new `Vec2` containing the component-wise minimum of `self` and `other`.
    ///
    /// For each component `x` and `y`, the method compares the corresponding components of 
    /// `self` and `other` and returns the smaller of the two.
    ///
    /// # Parameters
    ///
    /// - `other`: A reference to another `Vec2` instance to compare with `self`.
    ///
    /// # Returns
    ///
    /// A new `Vec2` where each component is the minimum value between `self` and `other`.
    ///
    /// # Examples
    ///
    /// ```
    /// let vec1 = Vec2::new(3, 7);
    /// let vec2 = Vec2::new(4, 5);
    /// let min_vec = vec1.min(&vec2);
    /// assert_eq!(min_vec, Vec2::new(3, 5));
    /// ```
    #[inline]
    pub fn min(&self, other: &Self) -> Self {
        Self::new(
            if self.x < other.x { self.x } else { other.x },
            if self.y < other.y { self.y } else { other.y }
        )
    }

    /// Returns a new `Vec2` containing the component-wise maximum of `self` and `other`.
    ///
    /// For each component `x` and `y`, the method compares the corresponding components of 
    /// `self` and `other` and returns the larger of the two.
    ///
    /// # Parameters
    ///
    /// - `other`: A reference to another `Vec2` instance to compare with `self`.
    ///
    /// # Returns
    ///
    /// A new `Vec2` where each component is the maximum value between `self` and `other`.
    ///
    /// # Examples
    ///
    /// ```
    /// let vec1 = Vec2::new(3, 7);
    /// let vec2 = Vec2::new(4, 5);
    /// let max_vec = vec1.max(&vec2);
    /// assert_eq!(max_vec, Vec2::new(4, 7));
    /// ```
    #[inline]
    pub fn max(&self, other: &Self) -> Self {
        Self::new(
            if self.x > other.x { self.x } else { other.x },
            if self.y > other.y { self.y } else { other.y }
        )
    }

    /// Clamps the components of `self` to lie within the inclusive range defined by `min` and `max`.
    ///
    /// For each component `x` and `y`, the method compares the corresponding component of `self`
    /// to the provided `min` and `max` values and ensures it lies within this range. If a component
    /// of `self` is less than the corresponding component of `min`, it is set to the `min` value. 
    /// If it is greater than the corresponding component of `max`, it is set to the `max` value.
    ///
    /// # Parameters
    ///
    /// - `min`: A reference to a `Vec2` representing the minimum allowed values for each component.
    /// - `max`: A reference to a `Vec2` representing the maximum allowed values for each component.
    ///
    /// # Returns
    ///
    /// A new `Vec2` where each component is clamped to the range `[min, max]`.
    ///
    /// # Panics
    ///
    /// This method will panic if any component of `min` is greater than the corresponding component
    /// of `max`, as it is not possible to clamp a value within an invalid range.
    ///
    /// # Examples
    ///
    /// ```
    /// let vec = Vec2::new(5, 10);
    /// let min_vec = Vec2::new(3, 7);
    /// let max_vec = Vec2::new(6, 8);
    /// let clamped_vec = vec.clamp(&min_vec, &max_vec);
    /// assert_eq!(clamped_vec, Vec2::new(5, 8));
    /// ```
    #[inline]
    pub fn clamp(&self, min: &Self, max: &Self) -> Self {
        Self::new(
            if self.x < min.x { min.x } else if self.x > max.x { max.x } else { self.x },
            if self.y < min.y { min.y } else if self.y > max.y { max.y } else { self.y }
        )
    }
}

impl<T> Vec2<T>
where
    T: NumAssign + Float,
{
    /// Returns the length (magnitude) of the vector.
    ///
    /// The length of the vector is calculated as the Euclidean norm, which is the square root of
    /// the sum of the squares of its components.
    ///
    /// # Returns
    ///
    /// The length (magnitude) of the vector as a value of type `T`.
    ///
    /// # Constraints
    ///
    /// - `T` must implement the `Float` trait, which provides methods for floating-point arithmetic.
    ///
    /// # Example
    ///
    /// ```
    /// let vec = Vec2::new(3.0, 4.0);
    /// assert_eq!(vec.length(), 5.0);
    /// ```
    #[inline]
    pub fn length(&self) -> T {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    /// Returns a normalized (unit length) version of the vector.
    ///
    /// The normalized vector is a unit vector that points in the same direction as the original vector.
    /// Normalization is achieved by dividing each component of the vector by its length.
    /// If the vector has zero length (is a zero vector), `None` is returned to indicate that normalization is not possible.
    ///
    /// # Returns
    ///
    /// - `Some(Self)`: A new vector with unit length pointing in the same direction as the original vector, if normalization is possible.
    /// - `None`: If the vector has zero length, indicating that it cannot be normalized.
    ///
    /// # Constraints
    ///
    /// - `T` must implement the `Float` trait, which provides methods for floating-point arithmetic.
    ///
    /// # Example
    ///
    /// ```
    /// let vec = Vec2::new(3.0, 4.0);
    /// let normalized = vec.normalize().unwrap();
    /// assert_eq!(normalized.length(), 1.0);
    /// ```
    #[inline]
    pub fn normalize(&self) -> Option<Self> {
        let len = self.length();
        if len.is_zero() {
            None
        } else {
            Some(Self::new(self.x / len, self.y / len))
        }
    }

    /// Computes the distance between this vector and another vector.
    ///
    /// The distance is calculated as the Euclidean distance between the two vectors, which is the length of the vector
    /// representing the difference between them.
    ///
    /// # Arguments
    ///
    /// - `other`: The other vector to which the distance is calculated.
    ///
    /// # Returns
    ///
    /// The distance between the two vectors as a value of type `T`.
    ///
    /// # Constraints
    ///
    /// - `T` must implement the `Float` trait, which provides methods for floating-point arithmetic.
    ///
    /// # Example
    ///
    /// ```
    /// let vec1 = Vec2::new(1.0, 2.0);
    /// let vec2 = Vec2::new(4.0, 6.0);
    /// assert_eq!(vec1.distance(&vec2), 5.0);
    /// ```
    #[inline]
    pub fn distance(&self, other: &Self) -> T {
        (
            (self.x - other.x) * (self.x - other.x) +
            (self.y - other.y) * (self.y - other.y)
        )
        .sqrt()
    }

    /// Computes the direction from this vector to another vector.
    ///
    /// This method calculates a normalized vector that points from `self` to `other`. If `self` and `other` are the same vector,
    /// resulting in a zero-length vector, `None` is returned.
    ///
    /// # Arguments
    ///
    /// - `other`: The target vector to which the direction is calculated.
    ///
    /// # Returns
    ///
    /// An `Option<Self>` where:
    /// - `Some(Self)` contains the normalized direction vector pointing from `self` to `other` if the vectors are not identical.
    /// - `None` if `self` and `other` are identical (i.e., the direction vector has zero length).
    ///
    /// # Constraints
    ///
    /// - `T` must implement the `Float` trait, which provides methods for floating-point arithmetic.
    ///
    /// # Example
    ///
    /// ```
    /// let vec1 = Vec2::new(1.0, 2.0);
    /// let vec2 = Vec2::new(4.0, 6.0);
    /// if let Some(direction) = vec1.direction(&vec2) {
    ///     assert_eq!(direction, Vec2::new(0.6, 0.8));
    /// } else {
    ///     panic!("The vectors are identical, so no direction can be computed.");
    /// }
    /// ```
    #[inline]
    pub fn direction(&self, other: &Self) -> Option<Self> {
        let direction = Self::new(
            other.x - self.x,
            other.y - self.y
        );
        direction.normalize()
    }

    /// Computes the angle (in radians) between this vector and another vector.
    ///
    /// This method calculates the angle between `self` and `other` vectors using the arctangent of the cross product
    /// and dot product of the vectors. The result is in radians, and the angle is measured counterclockwise.
    ///
    /// # Arguments
    ///
    /// - `other`: The other vector to which the angle is computed.
    ///
    /// # Returns
    ///
    /// The angle between the vectors as a value of type `T`.
    ///
    /// # Constraints
    ///
    /// - `T` must implement the `Float` trait, which provides methods for floating-point arithmetic.
    ///
    /// # Example
    ///
    /// ```
    /// let vec1 = Vec2::new(1.0, 0.0);
    /// let vec2 = Vec2::new(0.0, 1.0);
    /// assert_eq!(vec1.angle(&vec2), std::f32::consts::PI / 2.0);
    /// ```
    #[inline]
    pub fn angle(&self, other: &Self) -> T {
        let dot = self.x * other.x + self.y * other.y;
        let det = self.x * other.y - self.y * other.x;
        det.atan2(dot)
    }

    /// Computes the angle (in radians) of the line defined by two vectors.
    ///
    /// This method calculates the angle of the line segment defined by `self` and `end` relative to the positive x-axis.
    /// The vectors should be normalized for accurate results. The angle is measured from the positive x-axis to the line,
    /// and the result is in radians. The direction is clockwise from the positive x-axis.
    ///
    /// # Arguments
    ///
    /// - `end`: The end point of the line segment from `self` to `end`.
    ///
    /// # Returns
    ///
    /// The angle of the line segment as a value of type `T`.
    ///
    /// # Constraints
    ///
    /// - `T` must implement the `Float` trait, which provides methods for floating-point arithmetic.
    ///
    /// # Example
    ///
    /// ```
    /// let start = Vec2::new(1.0, 1.0);
    /// let end = Vec2::new(4.0, 3.0);
    /// assert_eq!(start.line_angle(&end), (-0.6435011).abs()); // Example result
    /// ```
    #[inline]
    pub fn line_angle(&self, end: &Self) -> T {
        // Note: The angle is measured clockwise from the positive x-axis.
        // If vectors are normalized, this is simply -atan2 of the difference.
        - (end.y - self.y).atan2(end.x - self.x)
    }

    /// Linearly interpolates between this vector and another vector.
    ///
    /// This method performs linear interpolation between `self` and `other` based on the parameter `t`. 
    /// When `t` is `0.0`, the result is `self`. When `t` is `1.0`, the result is `other`. For values of `t` 
    /// between `0.0` and `1.0`, the result is a point between `self` and `other` on the line segment connecting them.
    ///
    /// # Arguments
    ///
    /// - `other`: The vector to interpolate towards.
    /// - `t`: The interpolation factor, where `t` ranges from `0.0` to `1.0`.
    ///
    /// # Returns
    ///
    /// A new vector representing the point that is linearly interpolated between `self` and `other`.
    ///
    /// # Constraints
    ///
    /// - `T` must implement the `Float` trait, which provides methods for floating-point arithmetic.
    ///
    /// # Example
    ///
    /// ```
    /// let start = Vec2::new(0.0, 0.0);
    /// let end = Vec2::new(10.0, 10.0);
    /// let result = start.lerp(&end, 0.5);
    /// assert_eq!(result, Vec2::new(5.0, 5.0));
    /// ```
    #[inline]
    pub fn lerp(&self, other: &Self, t: T) -> Self {
        Self::new(
            self.x + t * (other.x - self.x),
            self.y + t * (other.y - self.y))
    }

    /// Computes the reflection of the vector around a given normal vector.
    ///
    /// The reflection is calculated using the formula:
    /// `reflected = self - 2 * (self · normal) * normal`
    /// where `self · normal` is the dot product between `self` and `normal`.
    ///
    /// # Parameters
    ///
    /// - `normal`: The normal vector around which to reflect. This vector should be normalized.
    ///
    /// # Returns
    ///
    /// A new vector representing the reflection of `self` around `normal`.
    ///
    /// # Constraints
    ///
    /// - `T` must implement the `Float` trait, which provides methods for floating-point arithmetic.
    ///
    /// # Example
    ///
    /// ```
    /// let incident = Vec2::new(1.0, -1.0);
    /// let normal = Vec2::new(0.0, 1.0).normalize().unwrap(); // Normalized normal vector
    /// let reflected = incident.reflect(&normal);
    /// assert_eq!(reflected, Vec2::new(1.0, 1.0)); // Reflection of (1.0, -1.0) around (0.0, 1.0) is (1.0, 1.0)
    /// ```
    #[inline]
    pub fn reflect(&self, normal: &Self) -> Self {
        let dot = self.x * normal.x + self.y * normal.y;
        let two = T::from(2.0).unwrap();

        Self::new(
            self.x - (two * normal.x) * dot,
            self.y - (two * normal.y) * dot
        )
    }

    /// Computes the direction of a refracted ray.
    ///
    /// This function calculates the direction of a refracted ray given the direction of the incoming ray,
    /// the normal vector of the surface, and the ratio of the refractive indices of the two media.
    ///
    /// # Parameters
    ///
    /// - `normal`: The normalized normal vector of the interface between two optical media.
    /// - `r`: The ratio of the refractive index of the medium from where the ray comes
    ///         to the refractive index of the medium on the other side of the surface.
    ///
    /// # Returns
    ///
    /// An `Option<Self>`. Returns `Some(Self)` with the direction of the refracted ray if refraction is possible,
    /// or `None` if refraction is not possible (e.g., due to total internal reflection).
    ///
    /// # Constraints
    ///
    /// - `T` must implement the `Float` trait, which provides methods for floating-point arithmetic.
    ///
    /// # Notes
    ///
    /// - The incoming ray and the normal vector should be normalized.
    /// - The result will be `None` if total internal reflection occurs (i.e., `d < 0`).
    #[inline]
    pub fn refract(&self, normal: &Self, r: T) -> Option<Self> {
        // Calculate the dot product between the incoming ray and the normal
        let dot = self.dot(normal);

        // Compute the squared ratio of the refractive indices
        let r2 = r * r;

        // Compute the discriminant to check for total internal reflection
        let one = T::one();
        let zero = T::zero();
        let d = one - r2 * (one - dot * dot);

        // If d is negative, total internal reflection occurs, so refraction is not possible
        if d < zero {
            None
        } else {
            // Calculate the square root of the discriminant
            let sqrt_d = d.sqrt();
            
            // Calculate the direction of the refracted ray
            let r_dot = r * dot;
            let v_x = r * self.x - (r_dot + sqrt_d) * normal.x;
            let v_y = r * self.y - (r_dot + sqrt_d) * normal.y;

            // Return the refracted ray direction
            Some(Self::new(v_x, v_y))
        }
    }
}

impl<T> fmt::Display for Vec2<T>
where
    T: fmt::Display,
{
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

impl<T> From<(T, T)> for Vec2<T>
where
    T: NumAssign + Copy,
{
    #[inline]
    fn from(tuple: (T, T)) -> Self {
        Vec2::new(tuple.0, tuple.1)
    }
}

impl<T> Into<(T, T)> for Vec2<T>
where
    T: NumAssign + Copy,
{
    #[inline]
    fn into(self) -> (T, T) {
        (self.x, self.y)
    }
}

impl<'a, T> IntoIterator for &'a mut Vec2<T>
where
    T: NumAssign + Copy
{
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<'a, T> IntoIterator for &'a Vec2<T>
where
    T: NumAssign + Copy
{
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T> Neg for Vec2<T>
where
    T: NumAssign + Copy + Neg<Output = T>,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y)
    }
}

impl<T> Add for Vec2<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y)
    }
}

impl<T> Add<T> for Vec2<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    #[inline]
    fn add(self, scalar: T) -> Self {
        Self::new(self.x + scalar, self.y + scalar)
    }
}

impl<T> Sub for Vec2<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y)
    }
}

impl<T> Sub<T> for Vec2<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    #[inline]
    fn sub(self, scalar: T) -> Self {
        Self::new(self.x - scalar, self.y - scalar)
    }
}

impl<T> Mul for Vec2<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    #[inline]
    fn mul(self, other: Self) -> Self {
        Self::new(self.x * other.x, self.y * other.y)
    }
}

impl<T> Mul<T> for Vec2<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    #[inline]
    fn mul(self, scalar: T) -> Self {
        Self::new(self.x * scalar, self.y * scalar)
    }
}

impl<T> Div for Vec2<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    #[inline]
    fn div(self, other: Self) -> Self {
        Self::new(self.x / other.x, self.y / other.y)
    }
}

impl<T> Div<T> for Vec2<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    #[inline]
    fn div(self, scalar: T) -> Self {
        Self::new(self.x / scalar, self.y / scalar)
    }
}

impl<T> AddAssign for Vec2<T>
where
    T: NumAssign + Copy,
{
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
    }
}

impl<T> AddAssign<T> for Vec2<T>
where
    T: NumAssign + Copy,
{
    #[inline]
    fn add_assign(&mut self, scalar: T) {
        self.x += scalar;
        self.y += scalar;
    }
}

impl<T> SubAssign for Vec2<T>
where
    T: NumAssign + Copy,
{
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        self.x -= other.x;
        self.y -= other.y;
    }
}

impl<T> SubAssign<T> for Vec2<T>
where
    T: NumAssign + Copy,
{
    #[inline]
    fn sub_assign(&mut self, scalar: T) {
        self.x -= scalar;
        self.y -= scalar;
    }
}

impl<T> MulAssign for Vec2<T>
where
    T: NumAssign + Copy,
{
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        self.x *= other.x;
        self.y *= other.y;
    }
}

impl<T> MulAssign<T> for Vec2<T>
where
    T: NumAssign + Copy,
{
    #[inline]
    fn mul_assign(&mut self, scalar: T) {
        self.x *= scalar;
        self.y *= scalar;
    }
}

impl<T> DivAssign for Vec2<T>
where
    T: NumAssign + Copy,
{
    #[inline]
    fn div_assign(&mut self, other: Self) {
        self.x /= other.x;
        self.y /= other.y;
    }
}

impl<T> DivAssign<T> for Vec2<T>
where
    T: NumAssign + Copy,
{
    #[inline]
    fn div_assign(&mut self, scalar: T) {
        self.x /= scalar;
        self.y /= scalar;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let vec = Vec2::new(3.0, 4.0);
        assert_eq!(vec.x, 3.0);
        assert_eq!(vec.y, 4.0);
    }

    #[test]
    fn test_zero() {
        let vec = Vec2::<f64>::zero();
        assert_eq!(vec.x, 0.0);
        assert_eq!(vec.y, 0.0);
    }

    #[test]
    fn test_one() {
        let vec = Vec2::<f64>::one();
        assert_eq!(vec.x, 1.0);
        assert_eq!(vec.y, 1.0);
    }

    #[test]
    fn test_negation() {
        let vec = Vec2::new(3.0, -4.0);
        let neg_vec = -vec;
        assert_eq!(neg_vec.x, -3.0);
        assert_eq!(neg_vec.y, 4.0);
    }

    #[test]
    fn test_add() {
        let vec1 = Vec2::new(1.0, 2.0);
        let vec2 = Vec2::new(3.0, 4.0);
        let result = vec1 + vec2;
        assert_eq!(result, Vec2::new(4.0, 6.0));
    }

    #[test]
    fn test_subtract() {
        let vec1 = Vec2::new(5.0, 7.0);
        let vec2 = Vec2::new(2.0, 3.0);
        let result = vec1 - vec2;
        assert_eq!(result, Vec2::new(3.0, 4.0));
    }

    #[test]
    fn test_multiply() {
        let vec = Vec2::new(2.0, 3.0);
        let scalar = 2.0;
        let result = vec * scalar;
        assert_eq!(result, Vec2::new(4.0, 6.0));
    }

    #[test]
    fn test_divide() {
        let vec = Vec2::new(8.0, 6.0);
        let scalar = 2.0;
        let result = vec / scalar;
        assert_eq!(result, Vec2::new(4.0, 3.0));
    }

    #[test]
    fn test_length() {
        let vec = Vec2::new(3.0, 4.0);
        let length = vec.length();
        assert_eq!(length, 5.0);
    }

    #[test]
    fn test_normalize() {
        let vec = Vec2::new(3.0, 4.0);
        let normalized_vec = vec.normalize().unwrap();
        assert_eq!(normalized_vec.x, 3.0 / 5.0);
        assert_eq!(normalized_vec.y, 4.0 / 5.0);
    }

    #[test]
    fn test_dot_product() {
        let vec1 = Vec2::new(1.0, 2.0);
        let vec2 = Vec2::new(3.0, 4.0);
        let dot_product = vec1.dot(&vec2);
        assert_eq!(dot_product, 11.0);
    }

    #[test]
    fn test_distance() {
        let vec1 = Vec2::new(1.0, 1.0);
        let vec2 = Vec2::new(4.0, 5.0);
        let distance = vec1.distance(&vec2);
        assert_eq!(distance, 5.0);
    }

    #[test]
    fn test_distance_squared() {
        let vec1 = Vec2::new(1.0, 1.0);
        let vec2 = Vec2::new(4.0, 5.0);
        let distance_squared = vec1.distance_squared(&vec2);
        assert_eq!(distance_squared, 25.0);
    }

    #[test]
    fn test_lerp() {
        let vec1 = Vec2::new(0.0, 0.0);
        let vec2 = Vec2::new(10.0, 10.0);
        let result = vec1.lerp(&vec2, 0.5);
        assert_eq!(result, Vec2::new(5.0, 5.0));
    }
}
