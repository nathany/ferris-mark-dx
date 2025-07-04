# Ferris Mark DX

A Windows DirectX 11 rendering experiment written in Rust.

## Features

- Windows 11 compatible window creation
- DirectX 11 hardware-accelerated rendering
- Proper window management (close with X button or Alt+F4)
- Window resizing support
- Configurable debug layer support

## Building and Running

### Debug Build (with DirectX debug layer)
```bash
cargo run
```

### Release Build (optimized, no debug layer)
```bash
cargo run --release
```

### Release Build with Debug Layer
```bash
cargo run --release --features d3d11-debug
```

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

## Dependencies

- `windows` crate for Win32 and DirectX 11 APIs
- Rust edition 2024

## Controls

- **Close Window**: Click X button or press Alt+F4
- **Resize**: Drag window edges (DirectX swap chain automatically resizes)

## Current Rendering

The application currently renders a solid cornflower blue background using DirectX 11. This serves as a foundation for more complex 3D graphics and effects.