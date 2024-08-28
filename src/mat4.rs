use num_traits::{
    NumAssign, Float
};

use std::slice::{
    Iter,
    IterMut
};

use std::ops::{
    Neg, Add, Sub, Mul, Div,
    Index, IndexMut
};

use crate::vec4::Vec4;
use crate::vec3::Vec3;

use std::fmt;

#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct Mat4<T> (
    pub Vec4<T>,
    pub Vec4<T>,
    pub Vec4<T>,
    pub Vec4<T>
);

impl<T> Mat4<T>
where
    T: NumAssign + Copy,
{
    #[inline]
    pub fn new(col0: &Vec4<T>, col1: &Vec4<T>, col2: &Vec4<T>, col3: &Vec4<T>) -> Self {
        Self (col0.clone(), col1.clone(), col2.clone(), col3.clone())
    }

    #[inline]
    pub fn zero() -> Self {
        Self (
            Vec4::zero(),
            Vec4::zero(),
            Vec4::zero(),
            Vec4::zero(),
        )
    }

    #[inline]
    pub fn identity() -> Self {
        Self (
            Vec4::new(T::one(), T::zero(), T::zero(), T::zero()),
            Vec4::new(T::zero(), T::one(), T::zero(), T::zero()),
            Vec4::new(T::zero(), T::zero(), T::one(), T::zero()),
            Vec4::new(T::zero(), T::zero(), T::zero(), T::one()),
        )
    }

    #[inline]
    pub fn transpose(&self) -> Self {
        Self (
            Vec4::new(self.0.x, self.1.x, self.2.x, self.3.x),
            Vec4::new(self.0.y, self.1.y, self.2.y, self.3.y),
            Vec4::new(self.0.z, self.1.z, self.2.z, self.3.z),
            Vec4::new(self.0.w, self.1.w, self.2.w, self.3.w),
        )
    }

    pub fn determinant(&self) -> T {
        let a00 = self.0.x;
        let a01 = self.0.y;
        let a02 = self.0.z;
        let a03 = self.0.w;
        let a10 = self.1.x;
        let a11 = self.1.y;
        let a12 = self.1.z;
        let a13 = self.1.w;
        let a20 = self.2.x;
        let a21 = self.2.y;
        let a22 = self.2.z;
        let a23 = self.2.w;
        let a30 = self.3.x;
        let a31 = self.3.y;
        let a32 = self.3.z;
        let a33 = self.3.w;

        let term1 = a30 * a21 * a12 * a03;
        let term2 = a20 * a31 * a12 * a03;
        let term3 = a30 * a11 * a22 * a03;
        let term4 = a10 * a31 * a22 * a03;
        let term5 = a20 * a11 * a32 * a03;
        let term6 = a10 * a21 * a32 * a03;
        let term7 = a30 * a21 * a02 * a13;
        let term8 = a20 * a31 * a02 * a13;
        let term9 = a30 * a01 * a22 * a13;
        let term10 = a00 * a31 * a22 * a13;
        let term11 = a20 * a01 * a32 * a13;
        let term12 = a00 * a21 * a32 * a13;
        let term13 = a30 * a11 * a02 * a23;
        let term14 = a10 * a31 * a02 * a23;
        let term15 = a30 * a01 * a12 * a23;
        let term16 = a00 * a31 * a12 * a23;
        let term17 = a10 * a01 * a32 * a23;
        let term18 = a00 * a11 * a32 * a23;
        let term19 = a20 * a11 * a02 * a33;
        let term20 = a10 * a21 * a02 * a33;
        let term21 = a20 * a01 * a12 * a33;
        let term22 = a00 * a21 * a12 * a33;
        let term23 = a10 * a01 * a22 * a33;
        let term24 = a00 * a11 * a22 * a33;

        let determinant = term1 - term2 - term3 + term4 +
                          term5 - term6 - term7 + term8 +
                          term9 - term10 - term11 + term12 +
                          term13 - term14 - term15 + term16 +
                          term17 - term18 - term19 + term20 +
                          term21 - term22 - term23 + term24;

        determinant
    }

    pub fn trace(&self) -> T {
        self.0.x + self.1.x + self.2.x + self.3.x
    }

    pub fn mul(&self, other: &Self) -> Self {
        let col0 = Vec4::new(
            self.0.dot(&Vec4::new(other.0.x, other.1.x, other.2.x, other.3.x)),
            self.1.dot(&Vec4::new(other.0.x, other.1.x, other.2.x, other.3.x)),
            self.2.dot(&Vec4::new(other.0.x, other.1.x, other.2.x, other.3.x)),
            self.3.dot(&Vec4::new(other.0.x, other.1.x, other.2.x, other.3.x)),
        );
        let col1 = Vec4::new(
            self.0.dot(&Vec4::new(other.0.y, other.1.y, other.2.y, other.3.y)),
            self.1.dot(&Vec4::new(other.0.y, other.1.y, other.2.y, other.3.y)),
            self.2.dot(&Vec4::new(other.0.y, other.1.y, other.2.y, other.3.y)),
            self.3.dot(&Vec4::new(other.0.y, other.1.y, other.2.y, other.3.y)),
        );
        let col2 = Vec4::new(
            self.0.dot(&Vec4::new(other.0.z, other.1.z, other.2.z, other.3.z)),
            self.1.dot(&Vec4::new(other.0.z, other.1.z, other.2.z, other.3.z)),
            self.2.dot(&Vec4::new(other.0.z, other.1.z, other.2.z, other.3.z)),
            self.3.dot(&Vec4::new(other.0.z, other.1.z, other.2.z, other.3.z)),
        );
        let col3 = Vec4::new(
            self.0.dot(&Vec4::new(other.0.w, other.1.w, other.2.w, other.3.w)),
            self.1.dot(&Vec4::new(other.0.w, other.1.w, other.2.w, other.3.w)),
            self.2.dot(&Vec4::new(other.0.w, other.1.w, other.2.w, other.3.w)),
            self.3.dot(&Vec4::new(other.0.w, other.1.w, other.2.w, other.3.w)),
        );
        Self::new(&col0, &col1, &col2, &col3)
    }

    #[inline]
    pub fn translate(&mut self, translate: &Vec3<T>) {
        self.3.x += translate.x;
        self.3.y += translate.y;
        self.3.z += translate.z;
    }

    #[inline]
    pub fn scale(&mut self, scale: &Vec3<T>) {
        self.0.x *= scale.x;
        self.1.y *= scale.y;
        self.2.z *= scale.z;
    }    
}

impl<T> Mat4<T>
where
    T: NumAssign + Float
{
    pub fn invert(&self) -> Option<Self> {
        let a00 = self.0.x;
        let a01 = self.0.y;
        let a02 = self.0.z;
        let a03 = self.0.w;
        let a10 = self.1.x;
        let a11 = self.1.y;
        let a12 = self.1.z;
        let a13 = self.1.w;
        let a20 = self.2.x;
        let a21 = self.2.y;
        let a22 = self.2.z;
        let a23 = self.2.w;
        let a30 = self.3.x;
        let a31 = self.3.y;
        let a32 = self.3.z;
        let a33 = self.3.w;

        let b00 = a00 * a11 - a01 * a10;
        let b01 = a00 * a12 - a02 * a10;
        let b02 = a00 * a13 - a03 * a10;
        let b03 = a01 * a12 - a02 * a11;
        let b04 = a01 * a13 - a03 * a11;
        let b05 = a02 * a13 - a03 * a12;
        let b06 = a20 * a31 - a21 * a30;
        let b07 = a20 * a32 - a22 * a30;
        let b08 = a20 * a33 - a23 * a30;
        let b09 = a21 * a32 - a22 * a31;
        let b10 = a21 * a33 - a23 * a31;
        let b11 = a22 * a33 - a23 * a32;

        let det = b00 * b11 - b01 * b10 +
                     b02 * b09 + b03 * b08 -
                     b04 * b07 + b05 * b06;

        if det.is_zero() {
            return None;
        }

        let inv_det = T::one() / det;

        Some(Self(
            Vec4::new(
                (a11 * b11 - a12 * b10 + a13 * b09) * inv_det,
                (-a01 * b11 + a02 * b10 - a03 * b09) * inv_det,
                (a31 * b05 - a32 * b04 + a33 * b03) * inv_det,
                (-a21 * b05 + a22 * b04 - a23 * b03) * inv_det,
            ),
            Vec4::new(
                (-a10 * b11 + a12 * b08 - a13 * b07) * inv_det,
                (a00 * b11 - a02 * b08 + a03 * b07) * inv_det,
                (-a30 * b05 + a32 * b02 - a33 * b01) * inv_det,
                (a20 * b05 - a22 * b02 + a23 * b01) * inv_det,
            ),
            Vec4::new(
                (a10 * b10 - a11 * b08 + a13 * b06) * inv_det,
                (-a00 * b10 + a01 * b08 - a03 * b06) * inv_det,
                (a30 * b04 - a31 * b02 + a33 * b00) * inv_det,
                (-a20 * b04 + a21 * b02 - a23 * b00) * inv_det,
            ),
            Vec4::new(
                (-a10 * b09 + a11 * b07 - a12 * b06) * inv_det,
                (a00 * b09 - a01 * b07 + a02 * b06) * inv_det,
                (-a30 * b03 + a31 * b01 - a32 * b00) * inv_det,
                (a20 * b03 - a21 * b01 + a22 * b00) * inv_det,
            )
        ))
    }

    pub fn rotate(&mut self, mut axis: Vec3<T>, angle: T) {
        let len_sq = axis.length_squared();
        if !(len_sq.is_one() || len_sq.is_zero()) {
            let inv_len = len_sq.sqrt();
            axis *= inv_len;
        }
    
        let s = angle.sin();
        let c = angle.cos();
        let t = T::one() - c;

        let m00 = axis.x * axis.x * t + c;
        let m01 = axis.y * axis.x * t + axis.z * s;
        let m02 = axis.z * axis.x * t - axis.y * s;
    
        let m10 = axis.x * axis.y * t - axis.z * s;
        let m11 = axis.y * axis.y * t + c;
        let m12 = axis.z * axis.y * t + axis.x * s;
    
        let m20 = axis.x * axis.z * t + axis.y * s;
        let m21 = axis.y * axis.z * t - axis.x * s;
        let m22 = axis.z * axis.z * t + c;

        for i in 0..3 {
            let x = self[i][0];
            let y = self[i][1];
            let z = self[i][2];
            self[i][0] = x * m00 + y * m10 + z * m20;
            self[i][1] = x * m01 + y * m11 + z * m21;
            self[i][2] = x * m02 + y * m12 + z * m22;
        }
    }

    pub fn rotate_x(&mut self, angle: T) {
        let c = angle.cos();
        let s = angle.sin();
        for i in 0..3 {
            let y = self[i][1];
            let z = self[i][2];
            self[i][1] = y * c - z * s;
            self[i][2] = y * s + z * c;
        }
    }

    pub fn rotate_y(&mut self, angle: T) {
        let c = angle.cos();
        let s = angle.sin();
        for i in 0..3 {
            let x = self[i][0];
            let z = self[i][2];
            self[i][0] = x * c + z * s;
            self[i][2] = -x * s + z * c;
        }
    }

    pub fn rotate_z(&mut self, angle: T) {
        let c = angle.cos();
        let s = angle.sin();
        for i in 0..3 {
            let x = self[i][0];
            let y = self[i][1];
            self[i][0] = x * c - y * s;
            self[i][1] = x * s + y * c;
        }
    }

    pub fn rotate_xyz(&mut self, angles: Vec3<T>) {
        let cx = angles.x.cos();
        let sx = angles.x.sin();
        for i in 0..3 {
            let y = self[i][1];
            let z = self[i][2];
            self[i][1] = y * cx - z * sx;
            self[i][2] = y * sx + z * cx;
        }
        let cy = angles.y.cos();
        let sy = angles.y.sin();
        for i in 0..3 {
            let x = self[i][0];
            let z = self[i][2];
            self[i][0] = x * cy + z * sy;
            self[i][2] = -x * sy + z * cy;
        }
        let cz = angles.z.cos();
        let sz = angles.z.sin();
        for i in 0..3 {
            let x = self[i][0];
            let y = self[i][1];
            self[i][0] = x * cz - y * sz;
            self[i][1] = x * sz + y * cz;
        }
    }    

    pub fn rotate_zyx(&mut self, angles: Vec3<T>) {
        let cz = angles.z.cos();
        let sz = angles.z.sin();
        for i in 0..3 {
            let x = self[i][0];
            let y = self[i][1];
            self[i][0] = x * cz - y * sz;
            self[i][1] = x * sz + y * cz;
        }
        let cy = angles.y.cos();
        let sy = angles.y.sin();
        for i in 0..3 {
            let x = self[i][0];
            let z = self[i][2];
            self[i][0] = x * cy + z * sy;
            self[i][2] = -x * sy + z * cy;
        }
        let cx = angles.x.cos();
        let sx = angles.x.sin();
        for i in 0..3 {
            let y = self[i][1];
            let z = self[i][2];
            self[i][1] = y * cx - z * sx;
            self[i][2] = y * sx + z * cx;
        }
    }

    pub fn frustum(left: T, right: T, bottom: T, top: T, near_plane: T, far_plane: T) -> Self {
        let two = T::from(2.0).unwrap();

        let rl = right - left;
        let tb = top - bottom;
        let fn_ = far_plane - near_plane;

        Self (
            Vec4::new(
                (near_plane * two) / rl,
                T::zero(),
                T::zero(),
                T::zero(),
            ),
            Vec4::new(
                T::zero(),
                (near_plane * two) / tb,
                T::zero(),
                T::zero(),
            ),
            Vec4::new(
                (right + left) / rl,
                (top + bottom) / tb,
                -((far_plane + near_plane) / fn_),
                -T::one(),
            ),
            Vec4::new(
                T::zero(),
                T::zero(),
                -((far_plane * near_plane * two) / fn_),
                T::zero(),
            )
        )
    }

    pub fn perspective(fov_y: T, aspect: T, near_plane: T, far_plane: T) -> Self {
        let two = T::from(2.0).unwrap();

        let top = near_plane * (fov_y * T::from(0.5).unwrap()).tan();
        let bottom = -top;
        let right = top * aspect;
        let left = -right;

        let rl = right - left;
        let tb = top - bottom;
        let fn_ = far_plane - near_plane;

        Self (
            Vec4::new(
                (near_plane * two) / rl,
                T::zero(),
                T::zero(),
                T::zero(),
            ),
            Vec4::new(
                T::zero(),
                (near_plane * two) / tb,
                T::zero(),
                T::zero(),
            ),
            Vec4::new(
                (right + left) / rl,
                (top + bottom) / tb,
                -((far_plane + near_plane) / fn_),
                -T::one(),
            ),
            Vec4::new(
                T::zero(),
                T::zero(),
                -((far_plane * near_plane * two) / fn_),
                T::zero(),
            )
        )
    }

    pub fn orthographic(left: T, right: T, bottom: T, top: T, near_plane: T, far_plane: T) -> Self {
        let two = T::from(2.0).unwrap();

        let rl = right - left;
        let tb = top - bottom;
        let fn_ = far_plane - near_plane;

        Self (
            Vec4::new(
                two / rl,
                T::zero(),
                T::zero(),
                T::zero(),
            ),
            Vec4::new(
                T::zero(),
                two / tb,
                T::zero(),
                T::zero(),
            ),
            Vec4::new(
                T::zero(),
                T::zero(),
                -two / fn_,
                T::zero(),
            ),
            Vec4::new(
                -((left + right) / rl),
                -((top + bottom) / tb),
                -((far_plane + near_plane) / fn_),
                T::from(1.0).unwrap(),
            )
        )
    }

    pub fn look_at(eye: Vec3<T>, target: Vec3<T>, up: Vec3<T>) -> Self {
        let vz = Vec3 {
            x: eye.x - target.x,
            y: eye.y - target.y,
            z: eye.z - target.z,
        };

        let len = (vz.x * vz.x + vz.y * vz.y + vz.z * vz.z).sqrt();
        let vz = if len != T::zero() {
            Vec3 {
                x: vz.x / len,
                y: vz.y / len,
                z: vz.z / len,
            }
        } else {
            Vec3 {
                x: T::zero(),
                y: T::zero(),
                z: T::zero(),
            }
        };

        let vx = Vec3 {
            x: up.y * vz.z - up.z * vz.y,
            y: up.z * vz.x - up.x * vz.z,
            z: up.x * vz.y - up.y * vz.x,
        };

        let len = (vx.x * vx.x + vx.y * vx.y + vx.z * vx.z).sqrt();
        let vx = if len != T::zero() {
            Vec3 {
                x: vx.x / len,
                y: vx.y / len,
                z: vx.z / len,
            }
        } else {
            Vec3 {
                x: T::zero(),
                y: T::zero(),
                z: T::zero(),
            }
        };

        let vy = Vec3 {
            x: vz.y * vx.z - vz.z * vx.y,
            y: vz.z * vx.x - vz.x * vx.z,
            z: vz.x * vx.y - vz.y * vx.x,
        };

        Self (
            Vec4::new(
                vx.x,
                vy.x,
                vz.x,
                T::zero(),
            ),
            Vec4::new(
                vx.y,
                vy.y,
                vz.y,
                T::zero(),
            ),
            Vec4::new(
                vx.z,
                vy.z,
                vz.z,
                T::zero(),
            ),
            Vec4::new(
                -(vx.x * eye.x + vx.y * eye.y + vx.z * eye.z),
                -(vy.x * eye.x + vy.y * eye.y + vy.z * eye.z),
                -(vz.x * eye.x + vz.y * eye.y + vz.z * eye.z),
                T::one(),
            )
        )
    }
}

impl<T> fmt::Display for Mat4<T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "[{}, {}, {}, {}]", self.0, self.1, self.2, self.3)
    }
}

impl<T> Index<usize> for Mat4<T> {
    type Output = Vec4<T>;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.0,
            1 => &self.1,
            2 => &self.2,
            3 => &self.3,
            _ => panic!("Index out of bounds for Mat4"),
        }
    }
}

impl<T> IndexMut<usize> for Mat4<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.0,
            1 => &mut self.1,
            2 => &mut self.2,
            3 => &mut self.3,
            _ => panic!("Index out of bounds for Mat4"),
        }
    }
}

impl<'a, T> IntoIterator for &'a Mat4<T>
where
    T: NumAssign + Copy
{
    type Item = &'a Vec4<T>;
    type IntoIter = Iter<'a, Vec4<T>>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        let slice: &[Vec4<T>; 4] = unsafe { std::mem::transmute(self) };
        slice.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut Mat4<T>
where
    T: NumAssign + Copy
{
    type Item = &'a mut Vec4<T>;
    type IntoIter = IterMut<'a, Vec4<T>>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        let slice: &mut [Vec4<T>; 4] = unsafe { std::mem::transmute(self) };
        slice.iter_mut()
    }
}

impl<T> Neg for Mat4<T>
where
    T: NumAssign + Copy + Neg<Output = T>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self (
            -self.0,
            -self.1,
            -self.2,
            -self.3,
        )
    }
}

impl<T> Add<Mat4<T>> for Mat4<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self (
            self.0 + other.0,
            self.1 + other.1,
            self.2 + other.2,
            self.3 + other.3,
        )
    }
}

impl<T> Sub<Mat4<T>> for Mat4<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self (
            self.0 - other.0,
            self.1 - other.1,
            self.2 - other.2,
            self.3 - other.3,
        )
    }
}

impl<T> Mul<Mat4<T>> for Mat4<T>
where
    T: NumAssign + Copy,
{
    type Output = Mat4<T>;

    #[inline]
    fn mul(self, other: Mat4<T>) -> Mat4<T> {
        Self::mul(&self, &other)
    }
}

impl<T> Mul<T> for Mat4<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    fn mul(self, scalar: T) -> Self {
        Self (
            self.0 * scalar,
            self.1 * scalar,
            self.2 * scalar,
            self.3 * scalar,
        )
    }
}

impl<T> Div<T> for Mat4<T>
where
    T: NumAssign + Copy,
{
    type Output = Self;

    fn div(self, scalar: T) -> Self {
        Self (
            self.0 / scalar,
            self.1 / scalar,
            self.2 / scalar,
            self.3 / scalar,
        )
    }
}
