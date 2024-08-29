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

    #[inline]
    pub fn from_vec4(v: &Vec4<T>) -> Self {
        Self::new(v.x, v.y, v.z, v.w)
    }

    #[inline]
    pub fn to_vec4(&self) -> Vec4<T> {
        Vec4::new(self.x, self.y, self.z, self.w)
    }

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

    #[inline]
    pub fn dot(&self, other: &Self) -> T {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }

    #[inline]
    pub fn length_squared(&self) -> T {
        self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w
    }

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
    pub fn conjugate(&self) -> Self {
        Self::new(-self.x, -self.y, -self.z, self.w)
    }

    #[inline]
    pub fn inverse(&self) -> Option<Self> {
        let length_squared = self.length_squared();
        if length_squared.is_zero() {
            None
        } else {
            Some(self.conjugate() / length_squared)
        }
    }

    #[inline]
    pub fn multiply(&self, other: &Self) -> Self {
        Self::new(
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
        )
    }

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
