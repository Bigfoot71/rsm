use num_traits::{
    NumAssign, Signed, Float
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
use crate::vec4::Vec4;
use crate::mat3::Mat3;
use crate::mat4::Mat4;

/// Represents a 3D vector with generic numeric components.
///
/// `Vec3` is a generic structure that represents a 3-dimensional vector with components `x`, `y`, and `z`.
/// It provides methods for vector operations such as vector arithmetic, normalization, dot product,
/// cross product, distance calculation, and linear interpolation.
///
/// # Type Parameters
///
/// - `T`: The numeric type of the vector components. This type must implement traits like `Zero`, `One`, `NumAssign`, `Copy`, and others depending on the method.
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct Vec3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T> Vec3<T>
where
    T: NumAssign + Copy,
{
    /// Constructs a new `Vec3` with the given `x`, `y`, and `z` components.
    ///
    /// # Parameters
    /// - `x`: The x component of the vector.
    /// - `y`: The y component of the vector.
    /// - `z`: The z component of the vector.
    ///
    /// # Returns
    /// Returns a new `Vec3` instance with the specified `x`, `y`, and `z` values.
    ///
    /// # Examples
    /// ```
    /// let vec = Vec3::new(1.0, 2.0, 3.0);
    /// assert_eq!(vec.x, 1.0);
    /// assert_eq!(vec.y, 2.0);
    /// assert_eq!(vec.z, 3.0);
    /// ```
    #[inline]
    pub fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }

    /// Constructs a `Vec3` with all components set to zero.
    ///
    /// # Returns
    /// Returns a new `Vec3` instance with all components set to zero.
    ///
    /// # Examples
    /// ```
    /// let vec = Vec3::zero();
    /// assert_eq!(vec.x, T::zero());
    /// assert_eq!(vec.y, T::zero());
    /// assert_eq!(vec.z, T::zero());
    /// ```
    #[inline]
    pub fn zero() -> Self {
        Self::new(T::zero(), T::zero(), T::zero())
    }

    /// Constructs a `Vec3` with all components set to one.
    ///
    /// # Returns
    /// Returns a new `Vec3` instance with all components set to one.
    ///
    /// # Examples
    /// ```
    /// let vec = Vec3::one();
    /// assert_eq!(vec.x, T::one());
    /// assert_eq!(vec.y, T::one());
    /// assert_eq!(vec.z, T::one());
    /// ```
    #[inline]
    pub fn one() -> Self {
        Self::new(T::one(), T::one(), T::one())
    }

    /// Constructs a `Vec3` where all components are set to the same value `v`.
    ///
    /// # Parameters
    /// - `v`: The value to set for all components of the vector.
    ///
    /// # Returns
    /// Returns a new `Vec3` instance with all components set to `v`.
    ///
    /// # Examples
    /// ```
    /// let vec = Vec3::set(2.0);
    /// assert_eq!(vec.x, 2.0);
    /// assert_eq!(vec.y, 2.0);
    /// assert_eq!(vec.z, 2.0);
    /// ```
    #[inline]
    pub fn set(v: T) -> Self {
        Self::new(v, v, v)
    }

    /// Constructs a `Vec3` from a `Vec2` by setting the z component to zero.
    ///
    /// This function is useful for converting a 2D vector to a 3D vector when the z component is not needed
    /// or is implicitly zero.
    ///
    /// # Parameters
    /// - `v`: The `Vec2` instance to convert.
    ///
    /// # Returns
    /// Returns a new `Vec3` instance with the x and y components copied from `v` and the z component set to zero.
    ///
    /// # Examples
    /// ```
    /// let vec2 = Vec2::new(1.0, 2.0);
    /// let vec3 = Vec3::from_vec2(&vec2);
    /// assert_eq!(vec3.x, 1.0);
    /// assert_eq!(vec3.y, 2.0);
    /// assert_eq!(vec3.z, T::zero()); // z component is zero
    /// ```
    #[inline]
    pub fn from_vec2(v: &Vec2<T>) -> Self {
        Self::new(v.x, v.y, T::zero())
    }

    /// Constructs a `Vec3` from a `Vec4` by discarding the w component.
    ///
    /// This function is useful for converting a 4D vector to a 3D vector when the w component is not needed
    /// or is implicitly zero.
    ///
    /// # Parameters
    /// - `v`: The `Vec4` instance to convert.
    ///
    /// # Returns
    /// Returns a new `Vec3` instance with the x, y, and z components copied from `v`.
    ///
    /// # Examples
    /// ```
    /// let vec4 = Vec4::new(1.0, 2.0, 3.0, 4.0);
    /// let vec3 = Vec3::from_vec4(&vec4);
    /// assert_eq!(vec3.x, 1.0);
    /// assert_eq!(vec3.y, 2.0);
    /// assert_eq!(vec3.z, 3.0);
    /// ```
    #[inline]
    pub fn from_vec4(v: &Vec4<T>) -> Self {
        Self::new(v.x, v.y, v.z)
    }

    /// Computes the dot product of this vector with another vector.
    ///
    /// The dot product of two vectors is a scalar value that represents their degree of alignment. 
    /// It is calculated as `self.x * other.x + self.y * other.y + self.z * other.z`.
    ///
    /// # Parameters
    /// - `other`: The vector to compute the dot product with.
    ///
    /// # Returns
    /// Returns the dot product of this vector and `other`.
    ///
    /// # Examples
    /// ```
    /// let vec1 = Vec3::new(1.0, 2.0, 3.0);
    /// let vec2 = Vec3::new(4.0, 5.0, 6.0);
    /// let dot_product = vec1.dot(&vec2);
    /// assert_eq!(dot_product, 32.0); // 1*4 + 2*5 + 3*6 = 32
    /// ```
    #[inline]
    pub fn dot(&self, other: &Self) -> T {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Computes the cross product of this vector with another vector.
    ///
    /// The cross product of two vectors results in a vector that is perpendicular to both of the original vectors.
    /// It is calculated using the formula:
    /// ```
    /// x = self.y * other.z - self.z * other.y
    /// y = self.z * other.x - self.x * other.z
    /// z = self.x * other.y - self.y * other.x
    /// ```
    ///
    /// # Parameters
    /// - `other`: The vector to compute the cross product with.
    ///
    /// # Returns
    /// Returns a new `Vec3` instance that is the result of the cross product.
    ///
    /// # Examples
    /// ```
    /// let vec1 = Vec3::new(1.0, 0.0, 0.0);
    /// let vec2 = Vec3::new(0.0, 1.0, 0.0);
    /// let cross_product = vec1.cross(&vec2);
    /// assert_eq!(cross_product, Vec3::new(0.0, 0.0, 1.0)); // Perpendicular vector
    /// ```
    #[inline]
    pub fn cross(&self, other: &Self) -> Self {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    /// Computes the squared length (magnitude) of the vector.
    ///
    /// The squared length is the dot product of the vector with itself, and it avoids the computation
    /// of the square root which is more computationally expensive. It is calculated as:
    /// ```
    /// length_squared = self.x * self.x + self.y * self.y + self.z * self.z
    /// ```
    ///
    /// # Returns
    /// Returns the squared length of the vector.
    ///
    /// # Examples
    /// ```
    /// let vec = Vec3::new(3.0, 4.0, 0.0);
    /// let length_squared = vec.length_squared();
    /// assert_eq!(length_squared, 25.0); // 3^2 + 4^2 = 25
    /// ```
    #[inline]
    pub fn length_squared(&self) -> T {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Computes the squared distance between this vector and another vector.
    ///
    /// The squared distance is the squared length of the vector difference. It avoids the computation
    /// of the square root, which is more computationally expensive. It is calculated as:
    /// ```
    /// distance_squared = (self.x - other.x) * (self.x - other.x) +
    ///                    (self.y - other.y) * (self.y - other.y) +
    ///                    (self.z - other.z) * (self.z - other.z)
    /// ```
    ///
    /// # Parameters
    /// - `other`: The vector to compute the distance to.
    ///
    /// # Returns
    /// Returns the squared distance between this vector and `other`.
    ///
    /// # Examples
    /// ```
    /// let vec1 = Vec3::new(1.0, 2.0, 3.0);
    /// let vec2 = Vec3::new(4.0, 5.0, 6.0);
    /// let distance_squared = vec1.distance_squared(&vec2);
    /// assert_eq!(distance_squared, 27.0); // (1-4)^2 + (2-5)^2 + (3-6)^2 = 27
    /// ```
    #[inline]
    pub fn distance_squared(&self, other: &Self) -> T {
        (self.x - other.x) * (self.x - other.x) +
        (self.y - other.y) * (self.y - other.y) +
        (self.z - other.z) * (self.z - other.z)
    }

    /// Transforms the vector using a 3x3 matrix.
    ///
    /// This function applies a 3x3 matrix transformation to the vector. It is commonly used for 
    /// linear transformations such as rotation, scaling, and shearing. The transformation is applied
    /// using the formula:
    /// ```
    /// x' = transform.0.x * self.x + transform.1.x * self.y + transform.2.x * self.z
    /// y' = transform.0.y * self.x + transform.1.y * self.y + transform.2.y * self.z
    /// z' = transform.0.z * self.x + transform.1.z * self.y + transform.2.z * self.z
    /// ```
    ///
    /// # Parameters
    /// - `transform`: The 3x3 matrix to apply to the vector.
    ///
    /// # Returns
    /// Returns a new `Vec3` instance that is the result of the transformation.
    ///
    /// # Examples
    /// ```
    /// // Define a 3x3 matrix and a vector
    /// let transform = Mat3::new(
    ///     Vec3::new(1.0, 0.0, 0.0),
    ///     Vec3::new(0.0, 1.0, 0.0),
    ///     Vec3::new(0.0, 0.0, 1.0)
    /// );
    /// let vec = Vec3::new(1.0, 2.0, 3.0);
    /// let transformed_vec = vec.transform_mat3(&transform);
    /// assert_eq!(transformed_vec, vec); // Identity matrix, so result is the same as original vector
    /// ```
    #[inline]
    pub fn transform_mat3(&self, transform: &Mat3<T>) -> Self {
        let x = transform.0.x * self.x + transform.1.x * self.y + transform.2.x * self.z;
        let y = transform.0.y * self.x + transform.1.y * self.y + transform.2.y * self.z;
        let z = transform.0.z * self.x + transform.1.z * self.y + transform.2.z * self.z;
        Self::new(x, y, z)
    }

    /// Transforms the vector using a 4x4 matrix.
    ///
    /// This function applies a 4x4 matrix transformation to the vector, treating the vector as if
    /// it were in homogeneous coordinates. This is useful for more complex transformations that involve
    /// translations. The transformation is applied using the formula:
    /// ```
    /// x' = transform.0.x * self.x + transform.1.x * self.y + transform.2.x * self.z + transform.3.x
    /// y' = transform.0.y * self.x + transform.1.y * self.y + transform.2.y * self.z + transform.3.y
    /// z' = transform.0.z * self.x + transform.1.z * self.y + transform.2.z * self.z + transform.3.z
    /// ```
    ///
    /// # Parameters
    /// - `transform`: The 4x4 matrix to apply to the vector.
    ///
    /// # Returns
    /// Returns a new `Vec3` instance that is the result of the transformation.
    ///
    /// # Examples
    /// ```
    /// // Define a 4x4 matrix and a vector
    /// let transform = Mat4::new(
    ///     Vec4::new(1.0, 0.0, 0.0, 0.0),
    ///     Vec4::new(0.0, 1.0, 0.0, 0.0),
    ///     Vec4::new(0.0, 0.0, 1.0, 0.0),
    ///     Vec4::new(2.0, 3.0, 4.0, 1.0)
    /// );
    /// let vec = Vec3::new(1.0, 2.0, 3.0);
    /// let transformed_vec = vec.transform_mat4(&transform);
    /// assert_eq!(transformed_vec, Vec3::new(3.0, 5.0, 7.0)); // Transformed by matrix
    /// ```
    #[inline]
    pub fn transform_mat4(&self, transform: &Mat4<T>) -> Self {
        let x = transform.0.x * self.x + transform.1.x * self.y + transform.2.x * self.z + transform.3.x;
        let y = transform.0.y * self.x + transform.1.y * self.y + transform.2.y * self.z + transform.3.y;
        let z = transform.0.z * self.x + transform.1.z * self.y + transform.2.z * self.z + transform.3.z;
        Self::new(x, y, z)
    }
}

impl<T> Vec3<T>
where
    T: NumAssign + Copy + PartialOrd,
{
    /// Computes the component-wise minimum between this vector and another vector.
    ///
    /// For each component (x, y, z), the resulting vector contains the smaller of the corresponding
    /// components from this vector and `other`. This function is useful for finding the smallest
    /// values between two vectors.
    ///
    /// # Parameters
    /// - `other`: The vector to compare with.
    ///
    /// # Returns
    /// Returns a new `Vec3` instance where each component is the minimum of the corresponding components
    /// of `self` and `other`.
    ///
    /// # Examples
    /// ```
    /// let vec1 = Vec3::new(1.0, 2.0, 3.0);
    /// let vec2 = Vec3::new(4.0, 1.0, 2.0);
    /// let min_vec = vec1.min(&vec2);
    /// assert_eq!(min_vec, Vec3::new(1.0, 1.0, 2.0)); // Component-wise minimum
    /// ```
    #[inline]
    pub fn min(&self, other: &Self) -> Self {
        Self::new(
            if self.x < other.x { self.x } else { other.x },
            if self.y < other.y { self.y } else { other.y },
            if self.z < other.z { self.z } else { other.z }
        )
    }

    /// Computes the component-wise maximum between this vector and another vector.
    ///
    /// For each component (x, y, z), the resulting vector contains the larger of the corresponding
    /// components from this vector and `other`. This function is useful for finding the largest
    /// values between two vectors.
    ///
    /// # Parameters
    /// - `other`: The vector to compare with.
    ///
    /// # Returns
    /// Returns a new `Vec3` instance where each component is the maximum of the corresponding components
    /// of `self` and `other`.
    ///
    /// # Examples
    /// ```
    /// let vec1 = Vec3::new(1.0, 2.0, 3.0);
    /// let vec2 = Vec3::new(4.0, 1.0, 2.0);
    /// let max_vec = vec1.max(&vec2);
    /// assert_eq!(max_vec, Vec3::new(4.0, 2.0, 3.0)); // Component-wise maximum
    /// ```
    #[inline]
    pub fn max(&self, other: &Self) -> Self {
        Self::new(
            if self.x > other.x { self.x } else { other.x },
            if self.y > other.y { self.y } else { other.y },
            if self.z > other.z { self.z } else { other.z }
        )
    }

    /// Clamps the components of this vector between the corresponding components of two other vectors.
    ///
    /// For each component (x, y, z), the resulting vector will have each component clamped to be within
    /// the range specified by `min` and `max`. If a component is less than the corresponding `min` value,
    /// it is set to `min`. If it is greater than the corresponding `max` value, it is set to `max`.
    ///
    /// # Parameters
    /// - `min`: The vector specifying the minimum values for clamping.
    /// - `max`: The vector specifying the maximum values for clamping.
    ///
    /// # Returns
    /// Returns a new `Vec3` instance where each component is clamped to the range specified by the corresponding
    /// components of `min` and `max`.
    ///
    /// # Examples
    /// ```
    /// let vec = Vec3::new(5.0, -1.0, 10.0);
    /// let min_vec = Vec3::new(0.0, 0.0, 0.0);
    /// let max_vec = Vec3::new(3.0, 3.0, 8.0);
    /// let clamped_vec = vec.clamp(&min_vec, &max_vec);
    /// assert_eq!(clamped_vec, Vec3::new(3.0, 0.0, 8.0)); // Clamped values
    /// ```
    #[inline]
    pub fn clamp(&self, min: &Self, max: &Self) -> Self {
        Self::new(
            if self.x < min.x { min.x } else if self.x > max.x { max.x } else { self.x },
            if self.y < min.y { min.y } else if self.y > max.y { max.y } else { self.y },
            if self.z < min.z { min.z } else if self.z > max.z { max.z } else { self.z }
        )
    }
}

impl<T> Vec3<T>
where
    T: NumAssign + Signed + Copy + PartialOrd,
{
    /// Computes a vector that is perpendicular to this vector.
    ///
    /// This method finds a vector that is orthogonal to the current vector by identifying the 
    /// axis with the smallest absolute component and constructing a perpendicular vector relative
    /// to that axis. The resulting perpendicular vector is calculated using the cross product
    /// with one of the cardinal axes (x, y, or z) based on which axis has the smallest absolute value.
    ///
    /// The algorithm works as follows:
    /// 1. Find the axis with the smallest absolute value in the current vector.
    /// 2. Create a unit vector along this axis (cardinal axis).
    /// 3. Compute the cross product between the original vector and the cardinal axis vector.
    ///
    /// This method assumes that the input vector is not zero and will work correctly in 3D space
    /// to provide a perpendicular vector.
    ///
    /// # Returns
    /// Returns a new `Vec3` instance that is perpendicular to `self`.
    ///
    /// # Examples
    /// ```
    /// let vec = Vec3::new(1.0, 2.0, 3.0);
    /// let perpendicular_vec = vec.perpendicular();
    /// assert_eq!(perpendicular_vec, Vec3::new(-3.0, 1.0, 0.0).normalize()); // A vector orthogonal to (1, 2, 3)
    /// ```
    ///
    /// # Panics
    /// Panics if `self` is the zero vector, as no perpendicular vector can be determined.
    #[inline]
    pub fn perpendicular(&self) -> Self {
        let mut min = self.x.abs();
        let mut cardinal_axis = Self::new(
            T::one(), T::zero(), T::zero());

        if self.y.abs() < min {
            min = self.y.abs();
            cardinal_axis = Self::new(
                T::zero(), T::one(), T::zero());
        }

        if self.z.abs() < min {
            cardinal_axis = Self::new(
                T::zero(), T::zero(), T::one());
        }

        self.cross(&cardinal_axis)
    }
}

impl<T> Vec3<T>
where
    T: NumAssign + Float,
{
    /// Computes the length (magnitude) of this vector.
    ///
    /// The length is computed as the square root of the sum of the squares of its components.
    ///
    /// # Returns
    /// Returns the length of the vector as a value of type `T`.
    ///
    /// # Examples
    /// ```
    /// let vec = Vec3::new(3.0, 4.0, 0.0);
    /// let len = vec.length();
    /// assert_eq!(len, 5.0); // Length of vector (3, 4, 0) is 5
    /// ```
    #[inline]
    pub fn length(&self) -> T {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Normalizes the vector.
    ///
    /// Normalizing scales the vector to have a length of 1. If the vector is zero-length,
    /// normalization is not possible, and `None` is returned.
    ///
    /// # Returns
    /// Returns an `Option<Self>` where `Some(Self)` is the normalized vector and `None` is
    /// returned if the vector's length is zero.
    ///
    /// # Examples
    /// ```
    /// let vec = Vec3::new(3.0, 4.0, 0.0);
    /// if let Some(norm_vec) = vec.normalize() {
    ///     assert_eq!(norm_vec, Vec3::new(0.6, 0.8, 0.0)); // Normalized vector
    /// }
    /// ```
    #[inline]
    pub fn normalize(&self) -> Option<Self> {
        let len = self.length();
        if len.is_zero() {
            None
        } else {
            Some(Self::new(self.x / len, self.y / len, self.z / len))
        }
    }

    /// Computes the distance between this vector and another vector.
    ///
    /// The distance is computed as the length of the vector difference between `self` and `other`.
    ///
    /// # Parameters
    /// - `other`: The vector to calculate the distance to.
    ///
    /// # Returns
    /// Returns the distance between the vectors as a value of type `T`.
    ///
    /// # Examples
    /// ```
    /// let vec1 = Vec3::new(1.0, 2.0, 3.0);
    /// let vec2 = Vec3::new(4.0, 5.0, 6.0);
    /// let dist = vec1.distance(&vec2);
    /// assert_eq!(dist, 5.196152422706632); // Distance between (1, 2, 3) and (4, 5, 6)
    /// ```
    #[inline]
    pub fn distance(&self, other: &Self) -> T {
        (
            (self.x - other.x) * (self.x - other.x) +
            (self.y - other.y) * (self.y - other.y) +
            (self.z - other.z) * (self.z - other.z)
        )
        .sqrt()
    }

    /// Computes the direction from this vector to another vector.
    ///
    /// The direction is calculated as the normalized vector difference between `other` and `self`.
    /// Returns `None` if the resulting vector has zero length.
    ///
    /// # Parameters
    /// - `other`: The target vector to find the direction towards.
    ///
    /// # Returns
    /// Returns an `Option<Self>` where `Some(Self)` is the direction vector if non-zero, and
    /// `None` if the direction vector is zero.
    ///
    /// # Examples
    /// ```
    /// let vec1 = Vec3::new(1.0, 2.0, 3.0);
    /// let vec2 = Vec3::new(4.0, 5.0, 6.0);
    /// if let Some(dir) = vec1.direction(&vec2) {
    ///     assert_eq!(dir, Vec3::new(0.57735027, 0.57735027, 0.57735027)); // Direction vector
    /// }
    /// ```
    #[inline]
    pub fn direction(&self, other: &Self) -> Option<Self> {
        Self::new(
            other.x - self.x,
            other.y - self.y,
            other.z - self.z
        ).normalize()
    }

    /// Computes the angle between this vector and another vector.
    ///
    /// The angle is calculated using the cross product and dot product of the two vectors.
    /// It returns the angle in radians between the vectors.
    ///
    /// # Parameters
    /// - `other`: The vector to find the angle with respect to.
    ///
    /// # Returns
    /// Returns the angle between the vectors in radians.
    ///
    /// # Examples
    /// ```
    /// let vec1 = Vec3::new(1.0, 0.0, 0.0);
    /// let vec2 = Vec3::new(0.0, 1.0, 0.0);
    /// let angle = vec1.angle(&vec2);
    /// assert_eq!(angle, 1.5707963267948966); // Angle between (1, 0, 0) and (0, 1, 0) is π/2
    /// ```
    #[inline]
    pub fn angle(&self, other: &Self) -> T {
        let cross = Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        );

        let len_cross = cross.length();
        let dot = self.dot(other);

        len_cross.atan2(dot)
    }

    /// Projects this vector onto another vector.
    ///
    /// The projection of `self` onto `other` is computed as the scalar projection multiplied by the
    /// `other` vector.
    ///
    /// # Parameters
    /// - `other`: The vector to project onto.
    ///
    /// # Returns
    /// Returns the projection of `self` onto `other` as a new `Vec3` instance.
    ///
    /// # Examples
    /// ```
    /// let vec1 = Vec3::new(1.0, 2.0, 3.0);
    /// let vec2 = Vec3::new(0.0, 1.0, 0.0);
    /// let proj = vec1.project(&vec2);
    /// assert_eq!(proj, Vec3::new(0.0, 2.0, 0.0)); // Projection of (1, 2, 3) onto (0, 1, 0)
    /// ```
    #[inline]
    pub fn project(&self, other: &Self) -> Self {
        let s_dot_o = self.dot(other);
        let o_dot_o = other.dot(other);
        let mag = s_dot_o / o_dot_o;
        *other * mag
    }

    /// Rejects the component of this vector in the direction of another vector.
    ///
    /// The rejection of `self` from `other` is the component of `self` orthogonal to `other`.
    ///
    /// # Parameters
    /// - `other`: The vector to reject from.
    ///
    /// # Returns
    /// Returns the rejection of `self` from `other` as a new `Vec3` instance.
    ///
    /// # Examples
    /// ```
    /// let vec1 = Vec3::new(1.0, 2.0, 3.0);
    /// let vec2 = Vec3::new(0.0, 1.0, 0.0);
    /// let rej = vec1.reject(&vec2);
    /// assert_eq!(rej, Vec3::new(1.0, 1.0, 3.0)); // Rejection of (1, 2, 3) from (0, 1, 0)
    /// ```
    #[inline]
    pub fn reject(&self, other: &Self) -> Self {
        let s_dot_o = self.dot(other);
        let o_dot_o = other.dot(other);
        let mag = s_dot_o / o_dot_o;
        *self - (*other * mag)
    }

    /// Orthonormalizes `self` and `other` using the Gram-Schmidt process.
    ///
    /// This process produces two orthonormal vectors where `self` is the first and `other` is the second.
    ///
    /// # Parameters
    /// - `other`: The vector to orthonormalize with `self`.
    ///
    /// # Examples
    /// ```
    /// let mut vec1 = Vec3::new(1.0, 0.0, 0.0);
    /// let mut vec2 = Vec3::new(0.0, 1.0, 1.0);
    /// vec1.ortho_normalize(&mut vec2);
    /// assert_eq!(vec1, Vec3::new(1.0, 0.0, 0.0)); // Orthonormalized vector1
    /// assert_eq!(vec2, Vec3::new(0.0, 0.70710677, 0.70710677)); // Orthonormalized vector2
    /// ```
    #[inline]
    pub fn ortho_normalize(&mut self, other: &mut Self) {
        if let Some(n_s) = self.normalize() {
            *self = n_s;
        }
        let mut c_so: Vec3<T> = self.cross(other);
        if let Some(nc_so) = c_so.normalize() {
            c_so = nc_so;
        }
        *other = c_so.cross(self);
    }

    /// Rotates this vector around a given axis by a specified angle.
    ///
    /// The rotation is performed using Rodrigues' rotation formula. The `axis` vector must be
    /// normalized for the rotation to be correct.
    ///
    /// # Parameters
    /// - `axis`: The axis to rotate around. It will be normalized if not already.
    /// - `angle`: The angle of rotation in radians.
    ///
    /// # Returns
    /// Returns a new vector that is the result of rotating `self` around `axis` by `angle`.
    ///
    /// # Examples
    /// ```
    /// let vec = Vec3::new(1.0, 0.0, 0.0);
    /// let axis = Vec3::new(0.0, 1.0, 0.0);
    /// let rotated_vec = vec.rotate_by_axis(axis, std::f64::consts::PI / 2.0);
    /// assert_eq!(rotated_vec, Vec3::new(0.0, 0.0, -1.0)); // Rotated vector
    /// ```
    #[inline]
    pub fn rotate_by_axis(&self, mut axis: Self, angle: T) -> Self {
        if let Some(n_axis) = axis.normalize() {
            axis = n_axis;
        }

        let two = T::from(2.0).unwrap();
        let w = axis * (angle / two).sin();

        let ws1 = w.cross(self);
        let ws2 = w.cross(&ws1);

        *self + (ws1 * two * angle.cos()) + (ws2 * two)
    }

    /// Reflects this vector across a given normal vector.
    ///
    /// The reflection is computed using the formula `self - 2 * (self . normal) * normal`.
    ///
    /// # Parameters
    /// - `normal`: The normal vector to reflect across.
    ///
    /// # Returns
    /// Returns the reflected vector as a new `Vec3` instance.
    ///
    /// # Examples
    /// ```
    /// let vec = Vec3::new(1.0, 2.0, 3.0);
    /// let normal = Vec3::new(0.0, 1.0, 0.0);
    /// let reflected_vec = vec.reflect(&normal);
    /// assert_eq!(reflected_vec, Vec3::new(1.0, -2.0, 3.0)); // Reflected vector
    /// ```
    #[inline]
    pub fn reflect(&self, normal: &Self) -> Self {
        let two = T::from(2.0).unwrap();
        *self - (*normal * two) * self.dot(normal)
    }

    /// Linearly interpolates between this vector and another vector.
    ///
    /// The interpolation factor `t` determines the blend between `self` and `other`. When `t` is
    /// 0.0, the result is `self`, and when `t` is 1.0, the result is `other`.
    ///
    /// # Parameters
    /// - `other`: The target vector to interpolate towards.
    /// - `t`: The interpolation factor, a value between 0 and 1.
    ///
    /// # Returns
    /// Returns the interpolated vector as a new `Vec3` instance.
    ///
    /// # Examples
    /// ```
    /// let vec1 = Vec3::new(0.0, 0.0, 0.0);
    /// let vec2 = Vec3::new(10.0, 10.0, 10.0);
    /// let interpolated_vec = vec1.lerp(&vec2, 0.5);
    /// assert_eq!(interpolated_vec, Vec3::new(5.0, 5.0, 5.0)); // Interpolated vector
    /// ```
    #[inline]
    pub fn lerp(&self, other: &Self, t: T) -> Self {
        Self::new(
            self.x + t * (other.x - self.x),
            self.y + t * (other.y - self.y),
            self.z + t * (other.z - self.z)
        )
    }

    /// Moves this vector towards a target vector by a specified distance.
    ///
    /// The movement is limited by `max_distance`, ensuring the vector does not overshoot
    /// the target. If the target is within the distance, it is returned directly.
    ///
    /// # Parameters
    /// - `target`: The target vector to move towards.
    /// - `max_distance`: The maximum distance to move.
    ///
    /// # Returns
    /// Returns the new position after moving towards the target.
    ///
    /// # Examples
    /// ```
    /// let vec = Vec3::new(0.0, 0.0, 0.0);
    /// let target = Vec3::new(10.0, 10.0, 10.0);
    /// let new_vec = vec.move_towards(&target, 5.0);
    /// assert_eq!(new_vec, Vec3::new(3.5355339, 3.5355339, 3.5355339)); // Moved vector
    /// ```
    pub fn move_towards(&self, target: &Self, max_distance: T) -> Self {
        let dir = *target - *self;
        let dist_sq = dir.length_squared();
        if dist_sq.is_zero() || (max_distance * max_distance >= dist_sq) {
            return *target;
        }
        let dist = dist_sq.sqrt();
        *self + (dir / dist) * max_distance
    }

    /// Computes the reciprocal of each component of the vector.
    ///
    /// This method returns a new vector where each component is the reciprocal of the
    /// corresponding component in `self`.
    ///
    /// # Returns
    /// Returns a new `Vec3` instance where each component is the reciprocal of the original.
    ///
    /// # Examples
    /// ```
    /// let vec = Vec3::new(1.0, 2.0, 4.0);
    /// let recip_vec = vec.recip();
    /// assert_eq!(recip_vec, Vec3::new(1.0, 0.5, 0.25)); // Reciprocal vector
    /// ```
    #[inline]
    pub fn recip(&self) -> Self {
        Self::new(
            self.x.recip(),
            self.y.recip(),
            self.z.recip()
        )
    }
}

impl<T> fmt::Display for Vec3<T>
where
    T: fmt::Display,
{
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

impl<T> Vec3<T> {
    pub fn convert<U>(self) -> Vec3<U>
    where
        T: Into<U> + Copy,
    {
        Vec3 {
            x: self.x.into(),
            y: self.y.into(),
            z: self.z.into()
        }
    }
}

impl<T> From<(T, T, T)> for Vec3<T>
where
    T: NumAssign + Copy,
{
    #[inline]
    fn from(tuple: (T, T, T)) -> Self {
        Vec3::new(tuple.0, tuple.1, tuple.2)
    }
}

impl<T> Into<(T, T, T)> for Vec3<T>
where
    T: NumAssign + Copy,
{
    #[inline]
    fn into(self) -> (T, T, T) {
        (self.x, self.y, self.z)
    }
}

impl<T> Index<usize> for Vec3<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Index out of bounds for Vec3"),
        }
    }
}

impl<T> IndexMut<usize> for Vec3<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("Index out of bounds for Vec3"),
        }
    }
}

impl<'a, T> IntoIterator for &'a Vec3<T>
where
    T: NumAssign + Copy
{
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        let slice: &[T; 3] = unsafe { std::mem::transmute(self) };
        slice.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut Vec3<T>
where
    T: NumAssign + Copy
{
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        let slice: &mut [T; 3] = unsafe { std::mem::transmute(self) };
        slice.iter_mut()
    }
}

impl<T> Neg for Vec3<T>
where
    T: NumAssign + Copy + Neg<Output = T>,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y, -self.z)
    }
}

impl<T> Add for Vec3<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl<T> Add<T> for Vec3<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    #[inline]
    fn add(self, scalar: T) -> Self {
        Self::new(self.x + scalar, self.y + scalar, self.z + scalar)
    }
}

impl<T> Sub for Vec3<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl<T> Sub<T> for Vec3<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    #[inline]
    fn sub(self, scalar: T) -> Self {
        Self::new(self.x - scalar, self.y - scalar, self.z - scalar)
    }
}

impl<T> Mul for Vec3<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    #[inline]
    fn mul(self, other: Self) -> Self {
        Self::new(self.x * other.x, self.y * other.y, self.z * other.z)
    }
}

impl<T> Mul<T> for Vec3<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    #[inline]
    fn mul(self, scalar: T) -> Self {
        Self::new(self.x * scalar, self.y * scalar, self.z * scalar)
    }
}

impl<T> Div for Vec3<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    #[inline]
    fn div(self, other: Self) -> Self {
        Self::new(self.x / other.x, self.y / other.y, self.z / other.z)
    }
}

impl<T> Div<T> for Vec3<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    #[inline]
    fn div(self, scalar: T) -> Self {
        Self::new(self.x / scalar, self.y / scalar, self.z / scalar)
    }
}

impl<T> AddAssign for Vec3<T>
where
    T: NumAssign + Copy,
{
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

impl<T> AddAssign<T> for Vec3<T>
where
    T: NumAssign + Copy,
{
    #[inline]
    fn add_assign(&mut self, scalar: T) {
        self.x += scalar;
        self.y += scalar;
        self.z += scalar;
    }
}

impl<T> SubAssign for Vec3<T>
where
    T: NumAssign + Copy,
{
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
    }
}

impl<T> SubAssign<T> for Vec3<T>
where
    T: NumAssign + Copy,
{
    #[inline]
    fn sub_assign(&mut self, scalar: T) {
        self.x -= scalar;
        self.y -= scalar;
        self.z -= scalar;
    }
}

impl<T> MulAssign for Vec3<T>
where
    T: NumAssign + Copy,
{
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        self.x *= other.x;
        self.y *= other.y;
        self.z *= other.z;
    }
}

impl<T> MulAssign<T> for Vec3<T>
where
    T: NumAssign + Copy,
{
    #[inline]
    fn mul_assign(&mut self, scalar: T) {
        self.x *= scalar;
        self.y *= scalar;
        self.z *= scalar;
    }
}

impl<T> DivAssign for Vec3<T>
where
    T: NumAssign + Copy,
{
    #[inline]
    fn div_assign(&mut self, other: Self) {
        self.x /= other.x;
        self.y /= other.y;
        self.z /= other.z;
    }
}

impl<T> DivAssign<T> for Vec3<T>
where
    T: NumAssign + Copy,
{
    #[inline]
    fn div_assign(&mut self, scalar: T) {
        self.x /= scalar;
        self.y /= scalar;
        self.z /= scalar;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Asserts that the absolute difference between two values is within a specified epsilon.
    /// 
    /// # Example
    /// ```rust
    /// assert_abs_diff_eq!(1.0 + 1e-9, 2.0, epsilon = 1e-8); // This will panic because the difference is not within epsilon
    /// ```
    #[macro_export]
    macro_rules! assert_abs_diff_eq {
        ($left:expr, $right:expr $(, epsilon = $epsilon:expr)?) => {{
            let left_val = $left;
            let right_val = $right;
            let epsilon = $(
                $epsilon
            )?;
            
            let abs_diff = (left_val - right_val).abs();
            if abs_diff > epsilon {
                panic!(
                    "assertion failed: `(left ~ right)`\n\
                    left: `{}`\n\
                    right: `{}`\n\
                    absolute difference: `{}`\n\
                    epsilon: `{}`",
                    left_val,
                    right_val,
                    abs_diff,
                    epsilon
                );
            }
        }};
    }

    #[test]
    fn test_new() {
        let vec = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(vec.x, 1.0);
        assert_eq!(vec.y, 2.0);
        assert_eq!(vec.z, 3.0);
    }

    #[test]
    fn test_set() {
        let vec = Vec3::set(4.0);
        assert_eq!(vec.x, 4.0);
        assert_eq!(vec.y, 4.0);
        assert_eq!(vec.z, 4.0);
    }

    #[test]
    fn test_zero() {
        let vec = Vec3::<f64>::zero();
        assert_eq!(vec.x, 0.0);
        assert_eq!(vec.y, 0.0);
        assert_eq!(vec.z, 0.0);
    }

    #[test]
    fn test_one() {
        let vec = Vec3::<f64>::one();
        assert_eq!(vec.x, 1.0);
        assert_eq!(vec.y, 1.0);
        assert_eq!(vec.z, 1.0);
    }

    #[test]
    fn test_length() {
        let vec = Vec3::new(3.0, 4.0, 0.0);
        assert_abs_diff_eq!(vec.length(), 5.0, epsilon = 1e-8);
    }

    #[test]
    fn test_normalize() {
        let vec = Vec3::new(3.0, 4.0, 0.0);
        let norm_vec = vec.normalize().unwrap();
        assert_abs_diff_eq!(norm_vec.length(), 1.0, epsilon = 1e-8);
    }

    #[test]
    fn test_dot() {
        let vec1 = Vec3::new(1.0, 2.0, 3.0);
        let vec2 = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(vec1.dot(&vec2), 32.0);
    }

    #[test]
    fn test_cross() {
        let vec1 = Vec3::new(1.0, 2.0, 3.0);
        let vec2 = Vec3::new(4.0, 5.0, 6.0);
        let cross = vec1.cross(&vec2);
        assert_eq!(cross.x, -3.0);
        assert_eq!(cross.y, 6.0);
        assert_eq!(cross.z, -3.0);
    }

    #[test]
    fn test_distance() {
        let vec1 = Vec3::new(1.0, 2.0, 3.0);
        let vec2 = Vec3::new(4.0, 5.0, 6.0);
        assert_abs_diff_eq!(vec1.distance(&vec2), 5.196152422706632, epsilon = 1e-8);
    }

    #[test]
    fn test_distance_squared() {
        let vec1 = Vec3::new(1.0, 2.0, 3.0);
        let vec2 = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(vec1.distance_squared(&vec2), 27.0);
    }

    #[test]
    fn test_direction() {
        let vec1 = Vec3::new(1.0, 2.0, 3.0);
        let vec2 = Vec3::new(4.0, 5.0, 6.0);
        let direction = vec1.direction(&vec2).unwrap();
        assert_abs_diff_eq!(direction.length(), 1.0, epsilon = 1e-8);
    }

    #[test]
    fn test_angle() {
        let vec1 = Vec3::new(1.0, 0.0, 0.0);
        let vec2 = Vec3::new(0.0, 1.0, 0.0);
        assert_abs_diff_eq!(vec1.angle(&vec2), std::f32::consts::PI / 2.0, epsilon = 1e-8);
    }

    #[test]
    fn test_lerp() {
        let vec1 = Vec3::new(1.0, 2.0, 3.0);
        let vec2 = Vec3::new(4.0, 5.0, 6.0);
        let lerped = vec1.lerp(&vec2, 0.5);
        assert_eq!(lerped.x, 2.5);
        assert_eq!(lerped.y, 3.5);
        assert_eq!(lerped.z, 4.5);
    }

    #[test]
    fn test_neg() {
        let vec = Vec3::new(1.0, -2.0, 3.0);
        let neg_vec = -vec;
        assert_eq!(neg_vec.x, -1.0);
        assert_eq!(neg_vec.y, 2.0);
        assert_eq!(neg_vec.z, -3.0);
    }

    #[test]
    fn test_add() {
        let vec1 = Vec3::new(1.0, 2.0, 3.0);
        let vec2 = Vec3::new(4.0, 5.0, 6.0);
        let sum = vec1 + vec2;
        assert_eq!(sum.x, 5.0);
        assert_eq!(sum.y, 7.0);
        assert_eq!(sum.z, 9.0);
    }

    #[test]
    fn test_sub() {
        let vec1 = Vec3::new(1.0, 2.0, 3.0);
        let vec2 = Vec3::new(4.0, 5.0, 6.0);
        let diff = vec1 - vec2;
        assert_eq!(diff.x, -3.0);
        assert_eq!(diff.y, -3.0);
        assert_eq!(diff.z, -3.0);
    }

    #[test]
    fn test_mul() {
        let vec1 = Vec3::new(1.0, 2.0, 3.0);
        let vec2 = Vec3::new(4.0, 5.0, 6.0);
        let prod = vec1 * vec2;
        assert_eq!(prod.x, 4.0);
        assert_eq!(prod.y, 10.0);
        assert_eq!(prod.z, 18.0);
    }

    #[test]
    fn test_div() {
        let vec1 = Vec3::new(4.0, 6.0, 8.0);
        let vec2 = Vec3::new(2.0, 3.0, 4.0);
        let quot = vec1 / vec2;
        assert_eq!(quot.x, 2.0);
        assert_eq!(quot.y, 2.0);
        assert_eq!(quot.z, 2.0);
    }

    #[test]
    fn test_add_assign() {
        let mut vec1 = Vec3::new(1.0, 2.0, 3.0);
        let vec2 = Vec3::new(4.0, 5.0, 6.0);
        vec1 += vec2;
        assert_eq!(vec1.x, 5.0);
        assert_eq!(vec1.y, 7.0);
        assert_eq!(vec1.z, 9.0);
    }

    #[test]
    fn test_sub_assign() {
        let mut vec1 = Vec3::new(1.0, 2.0, 3.0);
        let vec2 = Vec3::new(4.0, 5.0, 6.0);
        vec1 -= vec2;
        assert_eq!(vec1.x, -3.0);
        assert_eq!(vec1.y, -3.0);
        assert_eq!(vec1.z, -3.0);
    }

    #[test]
    fn test_mul_assign() {
        let mut vec1 = Vec3::new(1.0, 2.0, 3.0);
        let vec2 = Vec3::new(4.0, 5.0, 6.0);
        vec1 *= vec2;
        assert_eq!(vec1.x, 4.0);
        assert_eq!(vec1.y, 10.0);
        assert_eq!(vec1.z, 18.0);
    }

    #[test]
    fn test_div_assign() {
        let mut vec1 = Vec3::new(4.0, 6.0, 8.0);
        let vec2 = Vec3::new(2.0, 3.0, 4.0);
        vec1 /= vec2;
        assert_eq!(vec1.x, 2.0);
        assert_eq!(vec1.y, 2.0);
        assert_eq!(vec1.z, 2.0);
    }
}
