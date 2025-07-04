# Ferris Mark DX

A Windows DirectX 11 2D sprite rendering benchmark written in Rust, inspired by [gophermark](https://github.com/unitoftime/experiments/tree/master/gophermark).

## Features

- **2D Sprite Rendering**: Multiple bouncing sprites with physics simulation
- **Performance Benchmarking**: Sprites-per-second throughput measurement
- **DirectX 11 Hardware Acceleration**: Efficient GPU rendering pipeline
- **Physics System**: Bouncing sprites with edge collision detection
- **Pixel-Perfect Rendering**: 128x128 pixel sprites with precise positioning
- **Point Filtering**: Nearest neighbor sampling for crisp pixel art
- **Real-time Performance Metrics**: FPS and sprites/second reporting
- **Configurable Sprite Count**: Command-line argument for testing different loads
- **VSync Support**: Smooth 60 FPS rendering with tear-free display
- **Debug Layer Integration**: DirectX validation and performance analysis

## Building and Running

### Basic Usage
```bash
# Run with default 100 sprites
cargo run

# Run with specific sprite count
cargo run 500
cargo run 1000
cargo run 5000
```

### Build Variants
```bash
# Debug Build (with DirectX debug layer)
cargo run

# Release Build (optimized, no debug layer)
cargo run --release

# Release Build with Debug Layer
cargo run --release --features d3d11-debug
```

### Benchmarking
The application accepts a sprite count as the first command-line argument:
```bash
cargo run 100      # 100 sprites
cargo run 500      # 500 sprites  
cargo run 1000     # 1000 sprites
cargo run 5000     # 5000 sprites
```

Performance metrics are displayed every second:
```
Sprites/sec: 6000000 | FPS: 60.0 | Sprites: 100000 | Frame time: 16.67ms
```

- **Sprites/sec**: Primary benchmark metric - total sprite throughput
- **FPS**: Frames per second (capped at 60 with VSync)
- **Sprites**: Current number of active sprites
- **Frame time**: Time per frame in milliseconds

## DirectX 11 Debug Layer

The application supports DirectX 11 debug layer validation which provides:
- API usage validation
- Additional error checking and warnings
- Performance analysis information
- Memory leak detection

### Debug Layer Modes

1. **Automatic (recommended)**: Debug layer is enabled in debug builds, disabled in release builds
2. **Manual**: Use the `d3d11-debug` feature flag to explicitly enable the debug layer

### Requirements for Debug Layer

The DirectX 11 debug layer requires the "Graphics Tools" Windows optional feature to be installed:
- Windows 10/11: Settings > Apps > Optional features > Add a feature > Graphics Tools
- Or install via PowerShell: `Enable-WindowsOptionalFeature -Online -FeatureName "DirectX-Tools"`

### Where Debug Messages Appear

DirectX debug messages typically appear in:
- **Visual Studio Output Window**: When running under the debugger
- **Windows Event Log**: Application and Services Logs > Microsoft > Windows > Direct3D11
- **Debug Output Stream**: Captured by debugging tools like PIX or Visual Studio Graphics Diagnostics

Note: Debug messages usually don't appear in the console when running standalone applications. The application will show "Debug Layer: ENABLED" to confirm the debug layer is active and capturing validation messages.

## Dependencies

- `windows` crate for Win32 and DirectX 11 APIs
- `image` crate for PNG texture loading
- Rust edition 2024

## Controls

- **Close Window**: Click X button or press Alt+F4
- **Resize**: Drag window edges (DirectX swap chain automatically resizes)

## Sprite System

The application renders multiple bouncing sprites with real-time physics simulation:

### Movement System
- **Physics Update**: Position integration with velocity and delta time
- **Boundary Collision**: Sprites bounce off screen edges with velocity reversal
- **Smooth Animation**: 60 FPS with VSync for tear-free movement
- **Random Initialization**: Each sprite starts with random velocity direction

### Rendering Pipeline
- **Batch Rendering**: All sprites rendered in single draw call batch
- **Transform Matrices**: Per-sprite positioning via constant buffer updates
- **Pixel-Perfect Sprites**: 128x128 pixel sprites with exact positioning
- **Texture Sampling**: Point filtering for crisp pixel art rendering
- **Efficient Geometry**: Shared vertex/index buffers for all sprites

### Performance Characteristics
- **GPU Accelerated**: DirectX 11 hardware rendering pipeline
- **Constant Buffer Updates**: Dynamic per-sprite transformation matrices
- **Memory Efficient**: Shared geometry data, unique transform per sprite
- **Scalable**: Performance scales with sprite count and GPU capabilities

## Benchmark Comparison

This implementation can be compared with other graphics APIs:
- Original [gophermark](https://github.com/unitoftime/experiments/tree/master/gophermark) (Go + OpenGL)
- DirectX 11 vs OpenGL performance characteristics
- CPU vs GPU bottleneck analysis via sprite count scaling

The sprites/second metric provides a standardized measurement for comparing 2D rendering performance across different implementations and hardware configurations.