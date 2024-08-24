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

use std::fmt;

#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct Mat2<T> (
    Vec2<T>,
    Vec2<T>,
);

impl<T> Mat2<T>
where
    T: NumAssign + Copy,
{
    #[inline]
    pub fn new(col0: &Vec2<T>, col1: &Vec2<T>) -> Self {
        Self(col0.clone(), col1.clone())
    }

    #[inline]
    pub fn zero() -> Self {
        Self (
            Vec2::zero(),
            Vec2::zero(),
        )
    }

    #[inline]
    pub fn identity() -> Self {
        Self (
            Vec2::new(T::one(), T::zero()),
            Vec2::new(T::zero(), T::one()),
        )
    }

    #[inline]
    pub fn scale(sx: T, sy: T) -> Self {
        Self (
            Vec2::new(sx, T::zero()),
            Vec2::new(T::zero(), sy),
        )
    }

    #[inline]
    pub fn iter(&self) -> Iter<'_, Vec2<T>> {
        let slice: &[Vec2<T>; 2] = unsafe { std::mem::transmute(self) };
        slice.iter()
    }

    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, Vec2<T>> {
        let slice: &mut [Vec2<T>; 2] = unsafe { std::mem::transmute(self) };
        slice.iter_mut()
    }

    #[inline]
    pub fn transpose(&self) -> Self {
        Self (
            Vec2::new(self.0.x, self.1.x),
            Vec2::new(self.0.y, self.1.y),
        )
    }

    pub fn determinant(&self) -> T {
        self.0.x * self.1.y - self.0.y * self.1.x
    }

    pub fn trace(&self) -> T {
        self.0.x + self.1.y
    }

    pub fn mul(&self, other: &Self) -> Self {
        let col0 = Vec2::new(
            self.0.x * other.0.x + self.1.x * other.0.y,
            self.0.y * other.0.x + self.1.y * other.0.y,
        );
        let col1 = Vec2::new(
            self.0.x * other.1.x + self.1.x * other.1.y,
            self.0.y * other.1.x + self.1.y * other.1.y,
        );
        Self::new(&col0, &col1)
    }
}

impl<T> Mat2<T>
where
    T: NumAssign + Float,
{
    #[inline]
    pub fn rotation(angle: T) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        Self (
            Vec2::new(c, -s),
            Vec2::new(s, c),
        )
    }
}

impl<T> fmt::Display for Mat2<T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "[{}, {}]", self.0, self.1)
    }
}

impl<'a, T> IntoIterator for &'a mut Mat2<T>
where
    T: NumAssign + Copy
{
    type Item = &'a mut Vec2<T>;
    type IntoIter = IterMut<'a, Vec2<T>>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<'a, T> IntoIterator for &'a Mat2<T>
where
    T: NumAssign + Copy
{
    type Item = &'a Vec2<T>;
    type IntoIter = Iter<'a, Vec2<T>>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T> Neg for Mat2<T>
where
    T: NumAssign + Copy + Neg<Output = T>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self (
            -self.0,
            -self.1
        )
    }
}

impl<T> Add<Mat2<T>> for Mat2<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self (
            self.0 + other.0,
            self.1 + other.1
        )
    }
}

impl<T> Sub<Mat2<T>> for Mat2<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self (
            self.0 - other.0,
            self.1 - other.1
        )
    }
}

impl<T> Mul<Mat2<T>> for Mat2<T>
where
    T: NumAssign + Copy,
{
    type Output = Mat2<T>;

    #[inline]
    fn mul(self, other: Mat2<T>) -> Mat2<T> {
        Self::mul(&self, &other)
    }
}

impl<T> Mul<T> for Mat2<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    fn mul(self, scalar: T) -> Self {
        Self (
            self.0 * scalar,
            self.1 * scalar
        )
    }
}

impl<T> Div<T> for Mat2<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    fn div(self, scalar: T) -> Self {
        Self (
            self.0 / scalar,
            self.1 / scalar
        )
    }
}
