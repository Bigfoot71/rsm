use num_traits::{Float, FromPrimitive, NumAssign, NumCast, Signed};

/// Clamps the given value between the specified minimum and maximum bounds.
///
/// This function ensures that the `value` is within the range `[min, max]`. If the `value` is less than `min`,
/// it returns `min`. If it is greater than `max`, it returns `max`. Otherwise, it returns the `value` itself.
///
/// # Type Parameters
/// - `T`: A type that implements the `PartialOrd` trait, which is used for comparison operations.
///
/// # Parameters
/// - `value`: The value to be clamped.
/// - `min`: The lower bound of the range.
/// - `max`: The upper bound of the range.
///
/// # Returns
/// Returns the clamped value, which is within the bounds of `min` and `max`.
///
/// # Examples
/// ```
/// let x = 10;
/// let clamped_value = clamp(x, 5, 15);
/// assert_eq!(clamped_value, 10); // value is within range
///
/// let clamped_value = clamp(x, 12, 20);
/// assert_eq!(clamped_value, 12); // value is clamped to min
///
/// let clamped_value = clamp(x, 0, 8);
/// assert_eq!(clamped_value, 8); // value is clamped to max
/// ```
#[inline]
pub fn clamp<T: PartialOrd>(value: T, min: T, max: T) -> T {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

/// Saturates the given floating-point value to the range [0.0, 1.0].
///
/// This function is a specific case of `clamp`, where the range is restricted to `[0, 1]`. It returns `0`
/// if the input value `x` is less than `0`, returns `1` if it is greater than `1`, and returns `x` itself
/// if it is within this range.
///
/// # Type Parameters
/// - `T`: A floating-point type that implements the `Float` trait.
///
/// # Parameters
/// - `x`: The value to be saturated.
///
/// # Returns
/// Returns the saturated value, constrained within the range `[0.0, 1.0]`.
///
/// # Examples
/// ```
/// use num_traits::float::Float;
///
/// let x = 1.2;
/// let saturated_value = saturate(x);
/// assert_eq!(saturated_value, 1.0); // value is clamped to max
///
/// let x = -0.5;
/// let saturated_value = saturate(x);
/// assert_eq!(saturated_value, 0.0); // value is clamped to min
///
/// let x = 0.5;
/// let saturated_value = saturate(x);
/// assert_eq!(saturated_value, 0.5); // value is within range
/// ```
#[inline]
pub fn saturate<T: Float>(x: T) -> T {
    clamp(x, T::zero(), T::one())
}

/// Wraps the given floating-point value within the specified range [min, max].
///
/// This function wraps the `value` around the interval `[min, max]` such that if the `value` exceeds `max`
/// it wraps around to `min`, and vice versa. It uses modular arithmetic to ensure the value falls within the
/// specified range. The `fmod` function is used to compute the modulus.
///
/// # Type Parameters
/// - `T`: A floating-point type that implements the `Float` and `NumCast` traits.
///
/// # Parameters
/// - `value`: The value to be wrapped.
/// - `min`: The lower bound of the range.
/// - `max`: The upper bound of the range.
///
/// # Returns
/// Returns the value wrapped around the interval `[min, max]`.
///
/// # Examples
/// ```
/// use num_traits::cast::NumCast;
/// use num_traits::float::Float;
///
/// let x = 7.5;
/// let wrapped_value = wrap(x, 0.0, 5.0);
/// assert_eq!(wrapped_value, 2.5); // value wraps around to fit within the range
///
/// let x = -3.0;
/// let wrapped_value = wrap(x, 0.0, 5.0);
/// assert_eq!(wrapped_value, 2.0); // value wraps around to fit within the range
/// ```
#[inline]
pub fn wrap<T: Float + NumCast>(value: T, min: T, max: T) -> T {
    let range = max - min;
    min + fmod(value - min, range)
}

/// Wraps an angle to be within the range [-π, π].
///
/// This function takes an angle (in radians) and normalizes it to be within the interval [-π, π]. This is useful
/// for ensuring that angles are within a standard range, which is particularly helpful in various applications
/// such as rotation and trigonometric computations.
///
/// # Type Parameters
/// - `T`: A floating-point type that implements the `Float`, `FromPrimitive`, `AddAssign`, and `SubAssign` traits.
///
/// # Parameters
/// - `angle`: The angle (in radians) to be wrapped.
///
/// # Returns
/// Returns the angle normalized to the range [-π, π].
///
/// # Examples
/// ```
/// use num_traits::float::Float;
/// use num_traits::FromPrimitive;
///
/// let angle = 4.0; // An angle larger than π
/// let wrapped_angle = wrap_angle(angle);
/// assert_eq!(wrapped_angle, -2.2831853071795862); // Wrapped angle within [-π, π]
///
/// let angle = -4.0; // An angle smaller than -π
/// let wrapped_angle = wrap_angle(angle);
/// assert_eq!(wrapped_angle, 2.2831853071795862); // Wrapped angle within [-π, π]
/// ```
#[inline]
pub fn wrap_angle<T: Float + FromPrimitive + std::ops::AddAssign + std::ops::SubAssign>(angle: T) -> T {
    let pi = T::from_f64(std::f64::consts::PI).unwrap();
    let two_pi = pi + pi;
    let mut wrapped = angle % two_pi;
    if wrapped < -pi {
        wrapped += two_pi;
    } else if wrapped > pi {
        wrapped -= two_pi;
    }
    wrapped
}

/// Normalizes a value to a range between 0 and 1.
///
/// This function scales the `value` from its original range `[start, end]` to a normalized range `[0, 1]`. This is
/// useful for tasks such as interpolation or preparing values for algorithms that require normalized inputs.
///
/// # Type Parameters
/// - `T`: A floating-point type that implements the `Float` trait.
///
/// # Parameters
/// - `value`: The value to be normalized.
/// - `start`: The start of the original range.
/// - `end`: The end of the original range.
///
/// # Returns
/// Returns the normalized value within the range `[0, 1]`.
///
/// # Examples
/// ```
/// use num_traits::float::Float;
///
/// let value = 7.0;
/// let normalized_value = normalize(value, 5.0, 10.0);
/// assert_eq!(normalized_value, 0.4); // Normalized value in range [0, 1]
///
/// let value = 15.0;
/// let normalized_value = normalize(value, 10.0, 20.0);
/// assert_eq!(normalized_value, 0.5); // Normalized value in range [0, 1]
/// ```
#[inline]
pub fn normalize<T: Float>(value: T, start: T, end: T) -> T {
    (value - start) / (end - start)
}

/// Remaps a value from one range to another.
///
/// This function transforms a `value` from the input range `[input_start, input_end]` to a corresponding value in
/// the output range `[output_start, output_end]`. This is particularly useful for scaling values between different
/// ranges or units.
///
/// # Type Parameters
/// - `T`: A floating-point type that implements the `Float` trait.
///
/// # Parameters
/// - `value`: The value to be remapped.
/// - `input_start`: The start of the input range.
/// - `input_end`: The end of the input range.
/// - `output_start`: The start of the output range.
/// - `output_end`: The end of the output range.
///
/// # Returns
/// Returns the remapped value within the range `[output_start, output_end]`.
///
/// # Examples
/// ```
/// use num_traits::float::Float;
///
/// let value = 5.0;
/// let remapped_value = remap(value, 0.0, 10.0, 100.0, 200.0);
/// assert_eq!(remapped_value, 150.0); // Remapped value in the new range
///
/// let value = 0.0;
/// let remapped_value = remap(value, 0.0, 1.0, 10.0, 20.0);
/// assert_eq!(remapped_value, 10.0); // Remapped value at the start of the new range
/// ```
#[inline]
pub fn remap<T: Float>(value: T, input_start: T, input_end: T, output_start: T, output_end: T) -> T {
    (value - input_start) / (input_end - input_start) * (output_end - output_start) + output_start
}

/// Computes the floating-point remainder of dividing `a` by `b`.
///
/// This function calculates the remainder of the division of `a` by `b`, which is defined as
/// `a - b * floor(a / b)`. It effectively performs a modulus operation for floating-point numbers.
///
/// # Type Parameters
/// - `T`: A floating-point type that implements the `Float` trait.
///
/// # Parameters
/// - `a`: The dividend.
/// - `b`: The divisor.
///
/// # Returns
/// Returns the remainder of `a` divided by `b`. The result will have the same sign as `a`.
///
/// # Examples
/// ```
/// use num_traits::float::Float;
///
/// let a = 5.7;
/// let b = 2.3;
/// let remainder = fmod(a, b);
/// assert_eq!(remainder, 1.1); // 5.7 - 2.3 * floor(5.7 / 2.3) = 1.1
///
/// let a = -5.7;
/// let b = 2.3;
/// let remainder = fmod(a, b);
/// assert_eq!(remainder, 1.2); // -5.7 - 2.3 * floor(-5.7 / 2.3) = 1.2
/// ```
#[inline]
pub fn fmod<T: Float>(a: T, b: T) -> T {
    a - b * (a / b).floor()
}

/// Extracts the fractional part of a floating-point number.
///
/// This function calculates the fractional part of the number `x` by subtracting its floor value
/// from itself. The result is always in the range [0, 1), representing the non-integer part of `x`.
///
/// # Type Parameters
/// - `T`: A floating-point type that implements the `Float` trait.
///
/// # Parameters
/// - `x`: The value from which the fractional part is to be extracted.
///
/// # Returns
/// Returns the fractional part of `x`, which is `x - floor(x)`. The result is in the range [0, 1).
///
/// # Examples
/// ```
/// use num_traits::float::Float;
///
/// let x = 5.75;
/// let fractional_part = fract(x);
/// assert_eq!(fractional_part, 0.75); // 5.75 - floor(5.75) = 0.75
///
/// let x = -2.3;
/// let fractional_part = fract(x);
/// assert_eq!(fractional_part, 0.7); // -2.3 - floor(-2.3) = 0.7
/// ```
#[inline]
pub fn fract<T: Float>(x: T) -> T {
    x - x.floor()
}

/// Computes the step function of `x` with respect to an edge value.
///
/// This function returns `0` if `x` is less than the `edge`, and `1` otherwise. It is often used in
/// mathematical functions to create a binary step or threshold effect, where values are either `0` or `1`
/// based on whether they cross a certain threshold.
///
/// # Type Parameters
/// - `T`: A floating-point type that implements the `Float` trait.
///
/// # Parameters
/// - `edge`: The threshold value that serves as the boundary.
/// - `x`: The value to be tested against the threshold.
///
/// # Returns
/// Returns `0` if `x` is less than `edge`, otherwise returns `1`.
///
/// # Examples
/// ```
/// use num_traits::float::Float;
///
/// let edge = 3.0;
/// let x = 4.0;
/// let step_value = step(edge, x);
/// assert_eq!(step_value, 1.0); // x is greater than edge, so return 1
///
/// let edge = 3.0;
/// let x = 2.0;
/// let step_value = step(edge, x);
/// assert_eq!(step_value, 0.0); // x is less than edge, so return 0
/// ```
#[inline]
pub fn step<T: Float>(edge: T, x: T) -> T {
    if x < edge { T::zero() } else { T::one() }
}

/// Computes the sign of a floating-point number.
///
/// This function returns `1` if the number `x` is positive, `-1` if it is negative, and `0` if it is zero.
/// It effectively provides the signum (or sign) of `x`, which can be useful for determining the direction
/// or magnitude of a value.
///
/// # Type Parameters
/// - `T`: A type that implements both the `Signed` and `Float` traits.
///
/// # Parameters
/// - `x`: The number whose sign is to be determined.
///
/// # Returns
/// Returns `1` if `x` is positive, `-1` if `x` is negative, and `0` if `x` is zero.
///
/// # Examples
/// ```
/// use num_traits::float::Float;
/// use num_traits::Signed;
///
/// let positive = 3.5;
/// let negative = -2.1;
/// let zero = 0.0;
///
/// assert_eq!(signum(positive), 1.0); // Positive number
/// assert_eq!(signum(negative), -1.0); // Negative number
/// assert_eq!(signum(zero), 0.0);     // Zero
/// ```
#[inline]
pub fn signum<T: Signed + Float>(x: T) -> T {
    if x > T::zero() {
        T::one()
    } else if x < T::zero() {
        -T::one()
    } else {
        T::zero()
    }
}

/// Determines if two floating-point numbers are approximately equal within a given tolerance.
///
/// This function checks if the absolute difference between `a` and `b` is less than a specified `epsilon`,
/// which is useful for floating-point comparisons where exact equality is impractical due to precision issues.
///
/// # Type Parameters
/// - `T`: A floating-point type that implements the `Float` trait.
///
/// # Parameters
/// - `a`: The first value to compare.
/// - `b`: The second value to compare.
/// - `epsilon`: The tolerance within which the values are considered approximately equal.
///
/// # Returns
/// Returns `true` if the absolute difference between `a` and `b` is less than `epsilon`, otherwise returns `false`.
///
/// # Examples
/// ```
/// use num_traits::float::Float;
///
/// let a = 1.000001;
/// let b = 1.000002;
/// let epsilon = 0.00001;
/// assert!(approx(a, b, epsilon)); // Values are approximately equal within epsilon
///
/// let a = 1.0;
/// let b = 2.0;
/// let epsilon = 0.00001;
/// assert!(!approx(a, b, epsilon)); // Values are not approximately equal
/// ```
#[inline]
pub fn approx<T: Float>(a: T, b: T, epsilon: T) -> bool {
    (a - b).abs() < epsilon
}

/// Performs linear interpolation between two values.
///
/// This function computes a value that is a linear interpolation between `a` and `b` based on the parameter `t`.
/// The parameter `t` is expected to be in the range `[0, 1]`, where `0` results in `a`, `1` results in `b`,
/// and values in between give a weighted average of `a` and `b`.
///
/// # Type Parameters
/// - `T`: A floating-point type that implements the `Float` trait.
///
/// # Parameters
/// - `a`: The starting value of the interpolation.
/// - `b`: The ending value of the interpolation.
/// - `t`: The interpolation factor, typically in the range `[0, 1]`.
///
/// # Returns
/// Returns the interpolated value between `a` and `b` based on `t`.
///
/// # Examples
/// ```
/// use num_traits::float::Float;
///
/// let a = 0.0;
/// let b = 10.0;
/// let t = 0.5;
/// let interpolated = lerp(a, b, t);
/// assert_eq!(interpolated, 5.0); // Midpoint between 0 and 10
///
/// let a = -1.0;
/// let b = 1.0;
/// let t = 0.25;
/// let interpolated = lerp(a, b, t);
/// assert_eq!(interpolated, -0.5); // Interpolated value based on t
/// ```
#[inline]
pub fn lerp<T: Float>(a: T, b: T, t: T) -> T {
    a + t * (b - a)
}

/// Performs linear interpolation between two angles, taking into account angle wrapping.
///
/// This function interpolates between two angles `a` and `b` considering the circular nature of angles,
/// meaning it accounts for the shortest path around the circle. It calculates the difference between `b`
/// and `a` using angle wrapping and then performs linear interpolation using the parameter `t`.
///
/// # Type Parameters
/// - `T`: A floating-point type that implements the `Float`, `FromPrimitive`, and `NumAssign` traits.
///
/// # Parameters
/// - `a`: The starting angle (in radians).
/// - `b`: The ending angle (in radians).
/// - `t`: The interpolation factor, typically in the range `[0, 1]`.
///
/// # Returns
/// Returns the interpolated angle between `a` and `b` based on `t`, taking into account angle wrapping.
///
/// # Examples
/// ```
/// use num_traits::{Float, FromPrimitive, NumAssign};
///
/// let a = 1.0; // Starting angle
/// let b = 4.0; // Ending angle
/// let t = 0.5; // Interpolation factor
/// let interpolated_angle = lerp_angle(a, b, t);
/// assert_eq!(interpolated_angle, 2.5); // Correctly interpolated angle considering wrapping
///
/// let a = 3.0; // Starting angle
/// let b = -3.0; // Ending angle (wrapping around)
/// let t = 0.5; // Interpolation factor
/// let interpolated_angle = lerp_angle(a, b, t);
/// assert_eq!(interpolated_angle, 0.0); // Correctly interpolated angle considering wrapping
/// ```
#[inline]
pub fn lerp_angle<T: Float + FromPrimitive + NumAssign>(a: T, b: T, t: T) -> T {
    let diff = wrap_angle(b - a);
    a + diff * t
}

/// Computes the normalized position of a value within a given range.
///
/// This function calculates the position of `value` in the range `[a, b]` as a normalized value in the range `[0, 1]`.
/// It is useful for converting values from one scale to a normalized scale.
///
/// # Type Parameters
/// - `T`: A floating-point type that implements the `Float` trait.
///
/// # Parameters
/// - `a`: The start of the range.
/// - `b`: The end of the range.
/// - `value`: The value to normalize within the range `[a, b]`.
///
/// # Returns
/// Returns the normalized value of `value` in the range `[0, 1]` based on its position within `[a, b]`.
///
/// # Examples
/// ```
/// use num_traits::float::Float;
///
/// let a = 0.0;
/// let b = 10.0;
/// let value = 5.0;
/// let normalized = inverse_lerp(a, b, value);
/// assert_eq!(normalized, 0.5); // Midpoint within the range
///
/// let a = -10.0;
/// let b = 10.0;
/// let value = 0.0;
/// let normalized = inverse_lerp(a, b, value);
/// assert_eq!(normalized, 0.5); // Midpoint within the range
/// ```
#[inline]
pub fn inverse_lerp<T: Float>(a: T, b: T, value: T) -> T {
    (value - a) / (b - a)
}

/// Computes a smoothstep function, which is used to interpolate smoothly between 0 and 1.
///
/// The smoothstep function is commonly used in graphics and animation to produce smooth transitions between
/// two values. It creates a smooth curve that starts at 0, rises to 1, and then flattens out. This function
/// clamps the input `x` to the range `[edge0, edge1]` and then applies a smoothstep interpolation formula.
///
/// # Type Parameters
/// - `T`: A floating-point type that implements the `Float` trait.
///
/// # Parameters
/// - `edge0`: The lower edge of the interpolation range.
/// - `edge1`: The upper edge of the interpolation range.
/// - `x`: The input value to interpolate, clamped to `[edge0, edge1]`.
///
/// # Returns
/// Returns the smoothly interpolated value between 0 and 1 based on the input `x` and edges.
///
/// # Examples
/// ```
/// use num_traits::float::Float;
///
/// let edge0 = 0.0;
/// let edge1 = 10.0;
/// let x = 5.0;
/// let smooth_value = smoothstep(edge0, edge1, x);
/// assert_eq!(smooth_value, 0.5); // Smooth transition value within the range
///
/// let edge0 = 0.0;
/// let edge1 = 1.0;
/// let x = 0.2;
/// let smooth_value = smoothstep(edge0, edge1, x);
/// assert_eq!(smooth_value, 0.08); // Smooth transition value closer to edge0
/// ```
#[inline]
pub fn smoothstep<T: Float>(edge0: T, edge1: T, x: T) -> T {
    let t = clamp((x - edge0) / (edge1 - edge0), T::zero(), T::one());
    t * t * (T::from(3.0).unwrap() - T::from(2.0).unwrap() * t)
}

/// Computes exponential decay of a value over time.
///
/// This function calculates the value of an initial quantity undergoing exponential decay over time. 
/// The rate of decay is specified by `decay_rate`, and `time` represents the elapsed time. 
/// The exponential decay is calculated using the formula: `initial * exp(-decay_rate * time)`.
///
/// # Type Parameters
/// - `T`: A floating-point type that implements the `Float` trait.
///
/// # Parameters
/// - `initial`: The initial value before decay.
/// - `decay_rate`: The rate at which the value decays. A higher rate results in faster decay.
/// - `time`: The elapsed time over which the decay occurs.
///
/// # Returns
/// Returns the value after applying exponential decay, which decreases over time according to the decay rate.
///
/// # Examples
/// ```
/// use num_traits::float::Float;
///
/// let initial = 100.0;
/// let decay_rate = 0.1;
/// let time = 5.0;
/// let decayed_value = exp_decay(initial, decay_rate, time);
/// assert!((decayed_value - 60.653).abs() < 1e-3); // Decayed value based on the formula
///
/// let initial = 50.0;
/// let decay_rate = 0.5;
/// let time = 2.0;
/// let decayed_value = exp_decay(initial, decay_rate, time);
/// assert!((decayed_value - 18.07).abs() < 1e-2); // Decayed value based on the formula
/// ```
#[inline]
pub fn exp_decay<T: Float>(initial: T, decay_rate: T, time: T) -> T {
    initial * (-decay_rate * time).exp()
}

/// Moves a value towards a target value by a maximum delta.
///
/// This function calculates a new value that moves `current` towards `target` by at most `max_delta`. 
/// If the distance between `current` and `target` is less than or equal to `max_delta`, 
/// the function returns `target`. Otherwise, it returns `current` adjusted by `max_delta` in the direction of `target`.
///
/// # Type Parameters
/// - `T`: A floating-point type that implements the `Float` trait.
///
/// # Parameters
/// - `current`: The current value.
/// - `target`: The target value to move towards.
/// - `max_delta`: The maximum distance that `current` can be moved towards `target` in one step.
///
/// # Returns
/// Returns the new value after moving towards `target` by at most `max_delta`. If `current` is already within `max_delta` of `target`, it returns `target`.
///
/// # Examples
/// ```
/// use num_traits::float::Float;
///
/// let current = 5.0;
/// let target = 10.0;
/// let max_delta = 4.0;
/// let new_value = move_towards(current, target, max_delta);
/// assert_eq!(new_value, 9.0); // Moves towards target by max_delta
///
/// let current = 5.0;
/// let target = 10.0;
/// let max_delta = 6.0;
/// let new_value = move_towards(current, target, max_delta);
/// assert_eq!(new_value, 10.0); // Reaches target as it's within max_delta
/// ```
#[inline]
pub fn move_towards<T: Float>(current: T, target: T, max_delta: T) -> T {
    let delta = target - current;
    let distance = delta.abs();

    if distance <= max_delta {
        target
    } else {
        current + delta / distance * max_delta
    }
}

/// Converts an angle from degrees to radians.
///
/// This function converts an angle given in degrees to radians. The conversion is performed using the formula:
/// `radians = degrees * (π / 180)`. This is useful for converting angles to the radian measure required by
/// many mathematical functions and trigonometric calculations.
///
/// # Type Parameters
/// - `T`: A floating-point type that implements the `Float` and `FromPrimitive` traits.
///
/// # Parameters
/// - `degrees`: The angle in degrees to be converted to radians.
///
/// # Returns
/// Returns the angle in radians corresponding to the input angle in degrees.
///
/// # Examples
/// ```
/// use num_traits::{Float, FromPrimitive};
///
/// let degrees = 180.0;
/// let radians = deg_to_rad(degrees);
/// assert_eq!(radians, 3.141592653589793); // Conversion of 180 degrees to radians (π)
///
/// let degrees = 90.0;
/// let radians = deg_to_rad(degrees);
/// assert_eq!(radians, 1.5707963267948966); // Conversion of 90 degrees to radians (π/2)
/// ```
#[inline]
pub fn deg_to_rad<T: Float + FromPrimitive>(degrees: T) -> T {
    let pi = T::from_f64(std::f64::consts::PI).unwrap();
    degrees * (pi / T::from_f64(180.0).unwrap())
}

/// Converts an angle from radians to degrees.
///
/// This function converts an angle given in radians to degrees. The conversion is performed using the formula:
/// `degrees = radians * (180 / π)`. This is useful for converting angles from radian measure to degrees,
/// which is a common unit in many applications.
///
/// # Type Parameters
/// - `T`: A floating-point type that implements the `Float` and `FromPrimitive` traits.
///
/// # Parameters
/// - `radians`: The angle in radians to be converted to degrees.
///
/// # Returns
/// Returns the angle in degrees corresponding to the input angle in radians.
///
/// # Examples
/// ```
/// use num_traits::{Float, FromPrimitive};
///
/// let radians = 3.141592653589793; // π
/// let degrees = rad_to_deg(radians);
/// assert_eq!(degrees, 180.0); // Conversion of π radians to degrees (180°)
///
/// let radians = 1.5707963267948966; // π/2
/// let degrees = rad_to_deg(radians);
/// assert_eq!(degrees, 90.0); // Conversion of π/2 radians to degrees (90°)
/// ```
#[inline]
pub fn rad_to_deg<T: Float + FromPrimitive>(radians: T) -> T {
    let pi = T::from_f64(std::f64::consts::PI).unwrap();
    radians * (T::from_f64(180.0).unwrap() / pi)
}

/// Computes the factorial of a non-negative integer.
///
/// This function calculates the factorial of a given non-negative integer `n`. The factorial of a number `n`
/// is the product of all positive integers less than or equal to `n`. It is defined as: `n! = n * (n-1) * ... * 1`.
///
/// # Parameters
/// - `n`: The non-negative integer whose factorial is to be computed.
///
/// # Returns
/// Returns the factorial of `n` as a `u64` integer. If `n` is `0`, the function returns `1` (by definition, `0! = 1`).
///
/// # Examples
/// ```
/// let n = 5;
/// let result = factorial(n);
/// assert_eq!(result, 120); // 5! = 5 * 4 * 3 * 2 * 1 = 120
///
/// let n = 0;
/// let result = factorial(n);
/// assert_eq!(result, 1); // 0! = 1 (by definition)
/// ```
#[inline]
pub fn factorial(n: u64) -> u64 {
    (1..=n).product()
}
