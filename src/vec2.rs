use num_traits::{Zero, One, NumAssign, Float};
use std::ops::{Neg, Add, Sub, Mul, Div};
use std::fmt;

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
    T: Zero + One + NumAssign + Copy,
{
    /// Creates a new vector with the given `x` and `y` components.
    pub fn new(x: T, y: T) -> Self {
        Self { x, y }
    }

    /// Creates a vector with both components set to the given value `v`.
    pub fn set(v: T) -> Self {
        Self { x: v, y: v }
    }

    /// Returns a vector with both components set to zero.
    pub fn zero() -> Self {
        Self::new(T::zero(), T::zero())
    }

    /// Returns a vector with both components set to one.
    pub fn one() -> Self {
        Self::new(T::one(), T::one())
    }

    /// Returns the length (magnitude) of the vector.
    ///
    /// # Constraints
    /// - `T` must implement the `Float` trait.
    pub fn length(&self) -> T
    where
        T: Float,
    {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    /// Returns a normalized (unit length) version of the vector.
    ///
    /// If the vector has zero length, `None` is returned.
    ///
    /// # Constraints
    /// - `T` must implement the `Float` trait.
    pub fn normalize(&self) -> Option<Self>
    where
        T: Float,
    {
        let len = self.length();
        if len.is_zero() {
            None
        } else {
            Some(Self::new(self.x / len, self.y / len))
        }
    }

    /// Computes the dot product of the vector with another vector.
    ///
    /// # Constraints
    /// - `T` must implement `Mul` and `Add` traits.
    pub fn dot(&self, other: &Self) -> T
    where
        T: Mul<Output = T> + Add<Output = T>,
    {
        self.x * other.x + self.y * other.y
    }

    /// Computes the distance between this vector and another vector.
    ///
    /// # Constraints
    /// - `T` must implement `Float`, `Sub`, and `Mul` traits.
    pub fn distance(&self, other: &Self) -> T
    where
        T: Float + Sub<Output = T> + Mul<Output = T>,
    {
        (
            (self.x - other.x) * (self.x - other.x) +
            (self.y - other.y) * (self.y - other.y)
        )
        .sqrt()
    }

    /// Computes the squared distance between this vector and another vector.
    ///
    /// This avoids the cost of computing a square root.
    ///
    /// # Constraints
    /// - `T` must implement `Sub` and `Mul` traits.
    pub fn distance_squared(&self, other: &Self) -> T
    where
        T: Sub<Output = T> + Mul<Output = T>,
    {
        (self.x - other.x) * (self.x - other.x) +
        (self.y - other.y) * (self.y - other.y)
    }

    /// Computes the direction from this vector to another vector.
    ///
    /// Returns a normalized vector pointing from `self` to `other`.
    /// If the vectors are identical, `None` is returned.
    ///
    /// # Constraints
    /// - `T` must implement `Float`, `Sub`, and `Div` traits.
    pub fn direction(&self, other: &Self) -> Option<Self>
    where
        T: Float + Sub<Output = T> + Div<Output = T>,
    {
        Self::new(
            other.x - self.x,
            other.y - self.y).normalize()
    }

    /// Computes the angle (in radians) between this vector and another vector.
    ///
    /// The angle is computed using the arccosine of the dot product of the vectors
    /// divided by the product of their lengths.
    ///
    /// # Constraints
    /// - `T` must implement the `Float` trait.
    pub fn angle(&self, other: &Self) -> T
    where
        T: Float,
    {
        let dot = self.dot(other);
        let len1 = self.length();
        let len2 = other.length();
        (dot / (len1 * len2)).acos()
    }

    /// Linearly interpolates between this vector and another vector.
    ///
    /// The interpolation is controlled by the parameter `t`, where `t = 0.0`
    /// returns `self` and `t = 1.0` returns `other`.
    ///
    /// # Constraints
    /// - `T` must implement the `Float` trait.
    pub fn lerp(&self, other: &Self, t: T) -> Self
    where
        T: Float,
    {
        Self::new(
            self.x + t * (other.x - self.x),
            self.y + t * (other.y - self.y))
    }
}


impl<T> fmt::Display for Vec2<T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

impl<T> From<(T, T)> for Vec2<T>
where
    T: Zero + One + NumAssign + Copy,
{
    fn from(tuple: (T, T)) -> Self {
        Vec2::new(tuple.0, tuple.1)
    }
}

impl<T> Into<(T, T)> for Vec2<T>
where
    T: Zero + One + NumAssign + Copy,
{
    fn into(self) -> (T, T) {
        (self.x, self.y)
    }
}

impl<T> Neg for Vec2<T>
where
    T: NumAssign + Copy + Neg<Output = T>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y)
    }
}

impl<T> Add for Vec2<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y)
    }
}

impl<T> Add<T> for Vec2<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;
    fn add(self, scalar: T) -> Self {
        Self::new(self.x + scalar, self.y + scalar)
    }
}

impl<T> Sub for Vec2<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y)
    }
}

impl<T> Sub<T> for Vec2<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;
    fn sub(self, scalar: T) -> Self {
        Self::new(self.x - scalar, self.y - scalar)
    }
}

impl<T> Mul for Vec2<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Self::new(self.x * other.x, self.y * other.y)
    }
}

impl<T> Mul<T> for Vec2<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;
    fn mul(self, scalar: T) -> Self {
        Self::new(self.x * scalar, self.y * scalar)
    }
}

impl<T> Div for Vec2<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;
    fn div(self, other: Self) -> Self {
        Self::new(self.x / other.x, self.y / other.y)
    }
}

impl<T> Div<T> for Vec2<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;
    fn div(self, scalar: T) -> Self {
        Self::new(self.x / scalar, self.y / scalar)
    }
}

use std::ops::{AddAssign, SubAssign, MulAssign, DivAssign};

impl<T> AddAssign for Vec2<T>
where
    T: NumAssign + Copy,
{
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
    }
}

impl<T> AddAssign<T> for Vec2<T>
where
    T: NumAssign + Copy,
{
    fn add_assign(&mut self, scalar: T) {
        self.x += scalar;
        self.y += scalar;
    }
}

impl<T> SubAssign for Vec2<T>
where
    T: NumAssign + Copy,
{
    fn sub_assign(&mut self, other: Self) {
        self.x -= other.x;
        self.y -= other.y;
    }
}

impl<T> SubAssign<T> for Vec2<T>
where
    T: NumAssign + Copy,
{
    fn sub_assign(&mut self, scalar: T) {
        self.x -= scalar;
        self.y -= scalar;
    }
}

impl<T> MulAssign for Vec2<T>
where
    T: NumAssign + Copy,
{
    fn mul_assign(&mut self, other: Self) {
        self.x *= other.x;
        self.y *= other.y;
    }
}

impl<T> MulAssign<T> for Vec2<T>
where
    T: NumAssign + Copy,
{
    fn mul_assign(&mut self, scalar: T) {
        self.x *= scalar;
        self.y *= scalar;
    }
}

impl<T> DivAssign for Vec2<T>
where
    T: NumAssign + Copy,
{
    fn div_assign(&mut self, other: Self) {
        self.x /= other.x;
        self.y /= other.y;
    }
}

impl<T> DivAssign<T> for Vec2<T>
where
    T: NumAssign + Copy,
{
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
