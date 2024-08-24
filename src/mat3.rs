use num_traits::{
    NumAssign, Float
};

use std::slice::{
    Iter,
    IterMut
};

use std::ops::{
    Neg, Add, Sub, Mul, Div
};

use crate::vec2::Vec2;
use crate::vec3::Vec3;

use std::fmt;

#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct Mat3<T> (
    Vec3<T>,
    Vec3<T>,
    Vec3<T>
);

impl<T> Mat3<T>
where
    T: NumAssign + Copy,
{
    #[inline]
    pub fn new(col0: &Vec3<T>, col1: &Vec3<T>, col2: &Vec3<T>) -> Self {
        Self (col0.clone(), col1.clone(), col2.clone())
    }

    #[inline]
    pub fn zero() -> Self {
        Self (
            Vec3::zero(),
            Vec3::zero(),
            Vec3::zero(),
        )
    }

    #[inline]
    pub fn identity() -> Self {
        Self (
            Vec3::new(T::one(), T::zero(), T::zero()),
            Vec3::new(T::zero(), T::one(), T::zero()),
            Vec3::new(T::zero(), T::zero(), T::one()),
        )
    }

    #[inline]
    pub fn translate_2d(translate: &Vec2<T>) -> Self {
        Self (
            Vec3::new(T::one(), T::zero(), T::zero()),
            Vec3::new(T::zero(), T::one(), T::zero()),
            Vec3::new(translate.x, translate.y, T::one()),
        )
    }

    #[inline]
    pub fn scale_2d(scale: &Vec2<T>) -> Self {
        Self (
            Vec3::new(scale.x, T::zero(), T::zero()),
            Vec3::new(T::zero(), scale.y, T::zero()),
            Vec3::new(T::zero(), T::zero(), T::one()),
        )
    }

    #[inline]
    pub fn scale_3d(scale: &Vec3<T>) -> Self {
        Self (
            Vec3::new(scale.x, T::zero(), T::zero()),
            Vec3::new(T::zero(), scale.y, T::zero()),
            Vec3::new(T::zero(), T::zero(), scale.z),
        )
    }

    #[inline]
    pub fn iter(&self) -> Iter<'_, Vec3<T>> {
        let slice: &[Vec3<T>; 3] = unsafe { std::mem::transmute(self) };
        slice.iter()
    }

    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, Vec3<T>> {
        let slice: &mut [Vec3<T>; 3] = unsafe { std::mem::transmute(self) };
        slice.iter_mut()
    }

    #[inline]
    pub fn transpose(&self) -> Self {
        Self (
            Vec3::new(self.0.x, self.1.x, self.2.x),
            Vec3::new(self.0.y, self.1.y, self.2.y),
            Vec3::new(self.0.z, self.1.z, self.2.z),
        )
    }

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

    pub fn trace(&self) -> T {
        self.0.x + self.1.y + self.2.z
    }

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
}

impl<T> Mat3<T>
where
    T: NumAssign + Float
{
    pub fn invert(&self) -> Self {
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

        if det == T::zero() {
            return Self::zero(); // Or handle non-invertible case differently
        }

        let inv_det = T::one() / det;

        Self(
            Vec3::new(b01 * inv_det, (-a22 * a01 + a02 * a21) * inv_det, (a12 * a01 - a02 * a11) * inv_det),
            Vec3::new(b11 * inv_det, (a22 * a00 - a02 * a20) * inv_det, (-a12 * a00 + a02 * a10) * inv_det),
            Vec3::new(b21 * inv_det, (-a21 * a00 + a01 * a20) * inv_det, (a11 * a00 - a01 * a10) * inv_det)
        )
    }

    #[inline]
    pub fn rotate_2d(angle: T) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        Self (
            Vec3::new(c, s, T::zero()),
            Vec3::new(-s, c, T::zero()),
            Vec3::new(T::zero(), T::zero(), T::one()),
        )
    }

    pub fn rotate_3d(mut axis: Vec3<T>, angle: T) -> Self {
        let len_sq = axis.length_squared();
        if !(len_sq.is_one() || len_sq.is_zero()) {
            let inv_len = len_sq.sqrt();
            axis *= inv_len;
        }
        let s = angle.sin();
        let c = angle.cos();
        let t = T::one() - c;

        Self(
            Vec3::<T>::new(
                axis.x * axis.x * t + c,
                axis.y * axis.x * t + axis.z * s,
                axis.z * axis.x * t - axis.y * s,
            ),
            Vec3::<T>::new(
                axis.x * axis.y * t - axis.z * s,
                axis.y * axis.y * t + c,
                axis.z * axis.y * t + axis.x * s,
            ),
            Vec3::<T>::new(
                axis.x * axis.z * t + axis.y * s,
                axis.y * axis.z * t - axis.x * s,
                axis.z * axis.z * t + c,
            ),
        )
    }

    pub fn rotate_x_3d(angle: T) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        Self (
            Vec3::new(T::one(), T::zero(), T::zero()),
            Vec3::new(T::zero(), c, s),
            Vec3::new(T::zero(), -s, c),
        )
    }

    pub fn rotate_y_3d(angle: T) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        Self (
            Vec3::new(c, T::zero(), -s),
            Vec3::new(T::zero(), T::one(), T::zero()),
            Vec3::new(s, T::zero(), c),
        )
    }

    pub fn rotate_z_3d(angle: T) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        Self (
            Vec3::new(c, s, T::zero()),
            Vec3::new(-s, c, T::zero()),
            Vec3::new(T::zero(), T::zero(), T::one()),
        )
    }
}

impl<T> fmt::Display for Mat3<T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "[{}, {}, {}]", self.0, self.1, self.2)
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
        self.iter_mut()
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
        self.iter()
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
