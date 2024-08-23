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

use crate::vec2::Vec2;
use crate::vec3::Vec3;

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
    #[inline]
    pub fn new(x: T, y: T, z: T, w: T) -> Self {
        Self { x, y, z, w }
    }

    #[inline]
    pub fn zero() -> Self {
        Self::new(T::zero(), T::zero(), T::zero(), T::zero())
    }

    #[inline]
    pub fn one() -> Self {
        Self::new(T::one(), T::one(), T::one(), T::one())
    }

    #[inline]
    pub fn set(v: T) -> Self {
        Self::new(v, v, v, v)
    }

    #[inline]
    pub fn from_vec2(v: &Vec2<T>) -> Self {
        Self::new(v.x, v.y, T::zero(), T::zero())
    }

    #[inline]
    pub fn from_vec3(v: &Vec3<T>) -> Self {
        Self::new(v.x, v.y, v.z, T::zero())
    }

    #[inline]
    pub fn iter(&self) -> Iter<'_, T> {
        let slice: &[T; 4] = unsafe { std::mem::transmute(self) };
        slice.iter()
    }

    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        let slice: &mut [T; 4] = unsafe { std::mem::transmute(self) };
        slice.iter_mut()
    }

    #[inline]
    pub fn dot(&self, other: &Self) -> T {
        self.x * other.x +
        self.y * other.y +
        self.z * other.z +
        self.w * other.w
    }

    #[inline]
    pub fn length_squared(&self) -> T {
        self.x * self.x +
        self.y * self.y +
        self.z * self.z +
        self.w * self.w
    }
}

impl<T> Vec4<T>
where
    T: NumAssign + Float,
{
    #[inline]
    pub fn length(&self) -> T {
        (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt()
    }

    #[inline]
    pub fn normalize(&self) -> Option<Self> {
        let len = self.length();
        if len.is_zero() {
            None
        } else {
            Some(Self::new(self.x / len, self.y / len, self.z / len, self.w / len))
        }
    }

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

impl<'a, T> IntoIterator for &'a mut Vec4<T>
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

impl<'a, T> IntoIterator for &'a Vec4<T>
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
