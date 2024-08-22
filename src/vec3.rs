use num_traits::{
    Zero, One, NumAssign, Float
};

use std::ops::{
    Neg, Add, Sub, Mul, Div,
    AddAssign, SubAssign, MulAssign, DivAssign
};

use std::fmt;

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
    T: Zero + One + NumAssign + Copy,
{
    pub fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }

    pub fn set(v: T) -> Self {
        Self { x: v, y: v, z: v }
    }

    pub fn zero() -> Self {
        Self::new(T::zero(), T::zero(), T::zero())
    }

    pub fn one() -> Self {
        Self::new(T::one(), T::one(), T::one())
    }

    pub fn dot(&self, other: &Self) -> T
    where
        T: Mul<Output = T> + Add<Output = T>,
    {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(&self, other: &Self) -> Self
    where
        T: Mul<Output = T> + Sub<Output = T>,
    {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    pub fn distance_squared(&self, other: &Self) -> T
    where
        T: Mul<Output = T> + Sub<Output = T>,
    {
        (self.x - other.x) * (self.x - other.x) +
        (self.y - other.y) * (self.y - other.y) +
        (self.z - other.z) * (self.z - other.z)
    }
}

impl<T> Vec3<T>
where
    T: Float + Zero + One + NumAssign + Copy,
{
    pub fn length(&self) -> T {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalize(&self) -> Option<Self> {
        let len = self.length();
        if len.is_zero() {
            None
        } else {
            Some(Self::new(self.x / len, self.y / len, self.z / len))
        }
    }

    pub fn distance(&self, other: &Self) -> T {
        (
            (self.x - other.x) * (self.x - other.x) +
            (self.y - other.y) * (self.y - other.y) +
            (self.z - other.z) * (self.z - other.z)
        )
        .sqrt()
    }

    pub fn direction(&self, other: &Self) -> Option<Self> {
        Self::new(
            other.x - self.x,
            other.y - self.y,
            other.z - self.z
        ).normalize()
    }

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

    pub fn lerp(&self, other: &Self, t: T) -> Self {
        Self::new(
            self.x + t * (other.x - self.x),
            self.y + t * (other.y - self.y),
            self.z + t * (other.z - self.z)
        )
    }
}

impl<T> fmt::Display for Vec3<T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

impl<T> From<(T, T, T)> for Vec3<T>
where
    T: Zero + One + NumAssign + Copy,
{
    fn from(tuple: (T, T, T)) -> Self {
        Vec3::new(tuple.0, tuple.1, tuple.2)
    }
}

impl<T> Into<(T, T, T)> for Vec3<T>
where
    T: Zero + One + NumAssign + Copy,
{
    fn into(self) -> (T, T, T) {
        (self.x, self.y, self.z)
    }
}

impl<T> Neg for Vec3<T>
where
    T: NumAssign + Copy + Neg<Output = T>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y, -self.z)
    }
}

impl<T> Add for Vec3<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl<T> Add<T> for Vec3<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    fn add(self, scalar: T) -> Self {
        Self::new(self.x + scalar, self.y + scalar, self.z + scalar)
    }
}

impl<T> Sub for Vec3<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl<T> Sub<T> for Vec3<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    fn sub(self, scalar: T) -> Self {
        Self::new(self.x - scalar, self.y - scalar, self.z - scalar)
    }
}

impl<T> Mul for Vec3<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self::new(self.x * other.x, self.y * other.y, self.z * other.z)
    }
}

impl<T> Mul<T> for Vec3<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    fn mul(self, scalar: T) -> Self {
        Self::new(self.x * scalar, self.y * scalar, self.z * scalar)
    }
}

impl<T> Div for Vec3<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    fn div(self, other: Self) -> Self {
        Self::new(self.x / other.x, self.y / other.y, self.z / other.z)
    }
}

impl<T> Div<T> for Vec3<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    fn div(self, scalar: T) -> Self {
        Self::new(self.x / scalar, self.y / scalar, self.z / scalar)
    }
}

impl<T> AddAssign for Vec3<T>
where
    T: NumAssign + Copy,
{
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
