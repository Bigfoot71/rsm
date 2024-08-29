# RSM - Rust Math Library

**RSM** is a mathematics library for Rust, designed to handle vector, matrix, and scalar operations. It's suitable for graphics programming, game development, physics simulations, and more.

## Features

- **Vector Operations**: Support for 2D, 3D, and 4D vectors.
- **Matrix Operations**: Methods for 2x2, 3x3, and 4x4 matrices.
- **Scalar Operations**: Basic and advanced arithmetic operations.

## Getting Started

To use RSM in your Rust project:

1. **Add to `Cargo.toml`**:

    ```toml
    [dependencies]
    rsm = "0.1.0"
    ```

2. **Import in Your Code**:

    ```rust
    use rsm::vec2::Vec2;
    use rsm::mat3::Mat3;
    ```

3. **Example Usage**:

    ```rust
    fn main() {
        let v1 = Vec2::new(1.0, 2.0);
        let v2 = Vec2::new(3.0, 4.0);
        let result = v1 + v2;
        println!("Result: {:?}", result);
    }
    ```

## Documentation

For detailed documentation, visit [docs.rs/rsm](https://docs.rs/rsm-lib/latest/rsm_lib/).

## License

RSM is licensed under the MIT License. See [LICENSE.md](LICENSE.md) for details.

## Need Help?

For questions or feedback, open an issue on [GitHub](https://github.com/Bigfoot71/rsm/issues).
