use std::env;
use std::ptr;
use std::time::Instant;
extern crate rand;
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::{
    Win32::Foundation::*, Win32::Graphics::Direct3D::*, Win32::Graphics::Direct3D11::*,
    Win32::Graphics::Dxgi::Common::*, Win32::Graphics::Dxgi::*, Win32::Graphics::Gdi::*,
    Win32::System::LibraryLoader::GetModuleHandleW, Win32::UI::HiDpi::*,
    Win32::UI::Input::KeyboardAndMouse::*, Win32::UI::WindowsAndMessaging::*, core::*,
};

// Configuration constants
const MAX_SPRITES: usize = 1000000; // Maximum sprites supported by the system

// Vertex structure for our textured quad
// #[repr(C)] ensures memory layout matches what DirectX expects
#[repr(C)]
#[derive(Clone, Copy)]
struct Vertex {
    position: [f32; 3],  // 3D position (x, y, z) - z is always 0 for 2D sprites
    tex_coord: [f32; 2], // UV texture coordinates (u, v) in [0,1] range
}

// Sprite structure for movement system
// Handles physics simulation and screen-space positioning
#[derive(Clone, Copy)]
struct Sprite {
    position: [f32; 2], // Screen position in pixels (x, y)
    velocity: [f32; 2], // Movement speed in pixels per second
    dpi_scale: f32,     // High-DPI display scaling factor
    sprite_width: f32,  // Sprite width in pixels
    sprite_height: f32, // Sprite height in pixels
}

impl Sprite {
    fn new() -> Self {
        use std::f32::consts::PI;

        // Random angle and speed
        let angle = rand::random::<f32>() * 2.0 * PI;
        let speed = 200.0 + rand::random::<f32>() * 100.0; // 200-300 pixels/second

        Self {
            position: [0.0, 0.0], // Will be set when sprite is added to context
            velocity: [angle.cos() * speed, angle.sin() * speed],
            dpi_scale: 1.0,       // Will be set properly when sprite is created
            sprite_width: 128.0,  // Will be set properly when sprite is created
            sprite_height: 128.0, // Will be set properly when sprite is created
        }
    }

    fn update(&mut self, dt: f32, window_width: f32, window_height: f32) {
        // Update position
        self.position[0] += self.velocity[0] * dt;
        self.position[1] += self.velocity[1] * dt;

        // Use actual sprite dimensions (adjusted for DPI)
        let sprite_width = self.sprite_width / self.dpi_scale;
        let sprite_height = self.sprite_height / self.dpi_scale;

        // Since sprite is center-origin, calculate half dimensions
        let half_width = sprite_width / 2.0;
        let half_height = sprite_height / 2.0;

        // Bounce off edges accounting for center origin
        if self.position[0] - half_width <= 0.0 {
            self.position[0] = half_width;
            self.velocity[0] = -self.velocity[0];
        }
        if self.position[0] + half_width >= window_width {
            self.position[0] = window_width - half_width;
            self.velocity[0] = -self.velocity[0];
        }
        if self.position[1] - half_height <= 0.0 {
            self.position[1] = half_height;
            self.velocity[1] = -self.velocity[1];
        }
        if self.position[1] + half_height >= window_height {
            self.position[1] = window_height - half_height;
            self.velocity[1] = -self.velocity[1];
        }
    }
}

/// SpriteBatch for efficient sprite rendering
/// Collects multiple sprites into a single draw call to minimize GPU state changes
struct SpriteBatch {
    vertices: Vec<Vertex>, // Vertex data for all sprites in the batch
    indices: Vec<u32>,     // Index data for all sprites in the batch
    max_sprites: usize,    // Maximum number of sprites this batch can hold
}

impl SpriteBatch {
    /// Create a new SpriteBatch with the specified maximum sprite capacity
    fn new(max_sprites: usize) -> Self {
        Self {
            vertices: Vec::with_capacity(max_sprites * 4), // 4 vertices per sprite quad
            indices: Vec::with_capacity(max_sprites * 6),  // 6 indices per sprite (2 triangles)
            max_sprites,
        }
    }

    /// Clear all sprites from the batch
    fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
    }

    /// Add a sprite to the batch with the specified position and dimensions
    fn add(
        &mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        window_width: f32,
        window_height: f32,
    ) {
        if self.vertices.len() / 4 >= self.max_sprites {
            return; // Batch is full - cannot add more sprites
        }

        let current_sprite = (self.vertices.len() / 4) as u32;
        let base_index = current_sprite * 4;

        // Convert pixel coordinates to NDC coordinates [-1, 1]
        // NDC: X [-1,1] left to right, Y [-1,1] bottom to top
        let left_ndc = (x / window_width) * 2.0 - 1.0;
        let right_ndc = ((x + width) / window_width) * 2.0 - 1.0;
        let top_ndc = 1.0 - (y / window_height) * 2.0; // Flip Y axis
        let bottom_ndc = 1.0 - ((y + height) / window_height) * 2.0;

        // Add 4 vertices for the sprite quad (2 triangles)
        self.vertices.extend_from_slice(&[
            Vertex {
                position: [right_ndc, top_ndc, 0.0],
                tex_coord: [1.0, 0.0],
            }, // top right
            Vertex {
                position: [right_ndc, bottom_ndc, 0.0],
                tex_coord: [1.0, 1.0],
            }, // bottom right
            Vertex {
                position: [left_ndc, bottom_ndc, 0.0],
                tex_coord: [0.0, 1.0],
            }, // bottom left
            Vertex {
                position: [left_ndc, top_ndc, 0.0],
                tex_coord: [0.0, 0.0],
            }, // top left
        ]);

        // Add 6 indices for 2 triangles forming a quad
        self.indices.extend_from_slice(&[
            base_index,
            base_index + 1,
            base_index + 3, // first triangle
            base_index + 1,
            base_index + 2,
            base_index + 3, // second triangle
        ]);
    }

    /// Get the number of sprites currently in the batch
    fn sprite_count(&self) -> usize {
        self.vertices.len() / 4
    }

    /// Check if the batch is empty
    fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }
}

// Transform matrix constant buffer - data passed to GPU shaders
// #[repr(C)] ensures memory layout matches HLSL constant buffer
#[repr(C)]
#[derive(Clone, Copy)]
struct TransformBuffer {
    transform: [f32; 16], // 4x4 matrix for sprite positioning
}

struct D3D11Context {
    // Core DirectX objects
    device: ID3D11Device,                // Factory for creating resources
    device_context: ID3D11DeviceContext, // Interface for rendering commands
    swap_chain: IDXGISwapChain,          // Front/back buffer management
    render_target_view: Option<ID3D11RenderTargetView>, // Where to render pixels

    // Geometry resources
    vertex_buffer: Option<ID3D11Buffer>, // Quad vertices in GPU memory
    index_buffer: Option<ID3D11Buffer>,  // Triangle indices for quad

    // Shader pipeline
    vertex_shader: Option<ID3D11VertexShader>, // Transforms vertex positions
    pixel_shader: Option<ID3D11PixelShader>,   // Determines pixel colors
    input_layout: Option<ID3D11InputLayout>,   // Vertex data format description

    // Texture resources
    texture: Option<ID3D11Texture2D>, // Sprite image data
    texture_view: Option<ID3D11ShaderResourceView>, // Shader interface to texture
    sampler_state: Option<ID3D11SamplerState>, // Texture filtering settings

    // Rendering state
    constant_buffer: Option<ID3D11Buffer>, // Per-sprite transform data
    blend_state: Option<ID3D11BlendState>, // Alpha transparency settings

    // Window and sprite properties
    window_width: f32,
    window_height: f32,
    dpi_scale: f32,    // High-DPI scaling factor
    sprite_width: f32, // Sprite dimensions in pixels
    sprite_height: f32,
    sprites: Vec<Sprite>,      // All sprite instances
    sprite_batch: SpriteBatch, // Batching system for efficient rendering
    vsync: bool,               // VSync enabled/disabled setting

    // Performance tracking
    last_time: Instant,
    frame_count: u32,
    last_log_time: Instant,
}

impl D3D11Context {
    unsafe fn new(hwnd: HWND) -> Result<Self> {
        // Create device and device context
        let mut device = None;
        let mut device_context = None;
        let mut swap_chain = None;

        // Get DPI scaling factor
        let dpi = unsafe { GetDpiForWindow(hwnd) };
        let dpi_scale = dpi as f32 / 96.0; // 96 is the standard DPI

        // Get client rect for swap chain description
        let mut client_rect = RECT::default();
        unsafe { GetClientRect(hwnd, &mut client_rect)? };
        let width = (client_rect.right - client_rect.left) as u32;
        let height = (client_rect.bottom - client_rect.top) as u32;

        // Swap chain description - configures the front/back buffer setup
        let swap_chain_desc = DXGI_SWAP_CHAIN_DESC {
            BufferDesc: DXGI_MODE_DESC {
                Width: width,
                Height: height,
                RefreshRate: DXGI_RATIONAL {
                    Numerator: 60,  // 60 Hz refresh rate
                    Denominator: 1, // for smooth animation
                },
                Format: DXGI_FORMAT_R8G8B8A8_UNORM, // 32-bit RGBA color (8 bits per channel)
                ScanlineOrdering: DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED, // Let driver choose
                Scaling: DXGI_MODE_SCALING_UNSPECIFIED, // Let driver choose scaling method
            },
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,   // No multisampling - better performance for 2D sprites
                Quality: 0, // No antialiasing quality needed for pixel art
            },
            BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT, // Buffer will be used for rendering
            BufferCount: 1,                               // Single back buffer (double buffering)
            OutputWindow: hwnd,                           // Target window handle
            Windowed: TRUE,                               // Windowed mode (not fullscreen)
            SwapEffect: DXGI_SWAP_EFFECT_DISCARD, // Discard back buffer after present (fastest)
            Flags: 0,                             // No special flags needed
        };

        // Create device, device context, and swap chain
        // Enable debug layer in debug builds or when d3d11-debug feature is enabled
        #[cfg(any(debug_assertions, feature = "d3d11-debug"))]
        let create_device_flags = D3D11_CREATE_DEVICE_DEBUG;
        #[cfg(not(any(debug_assertions, feature = "d3d11-debug")))]
        let create_device_flags = D3D11_CREATE_DEVICE_FLAG(0);

        unsafe {
            D3D11CreateDeviceAndSwapChain(
                None,                      // Use default adapter (primary graphics card)
                D3D_DRIVER_TYPE_HARDWARE,  // Use hardware acceleration (GPU rendering)
                None,                      // No software rasterizer needed
                create_device_flags,       // Debug layer flags (conditional)
                None,                      // Use default feature levels (DX11 support)
                D3D11_SDK_VERSION,         // SDK version for compatibility
                Some(&swap_chain_desc),    // Swap chain configuration
                Some(&mut swap_chain),     // Output swap chain
                Some(&mut device),         // Output device
                Some(ptr::null_mut()),     // Don't need actual feature level out
                Some(&mut device_context), // Output device context
            )?;
        }

        let device = device.unwrap();
        let device_context = device_context.unwrap();
        let swap_chain = swap_chain.unwrap();

        // Create blend state for alpha transparency support
        // This enables proper rendering of PNG sprites with transparent areas
        let blend_desc = D3D11_BLEND_DESC {
            AlphaToCoverageEnable: FALSE,  // Not using MSAA alpha-to-coverage
            IndependentBlendEnable: FALSE, // Same blend mode for all render targets
            RenderTarget: [
                D3D11_RENDER_TARGET_BLEND_DESC {
                    BlendEnable: TRUE,                    // Enable alpha blending
                    SrcBlend: D3D11_BLEND_SRC_ALPHA,      // Source: sprite alpha
                    DestBlend: D3D11_BLEND_INV_SRC_ALPHA, // Dest: (1 - sprite alpha)
                    BlendOp: D3D11_BLEND_OP_ADD,          // Standard alpha blend formula
                    SrcBlendAlpha: D3D11_BLEND_ONE,       // Preserve source alpha
                    DestBlendAlpha: D3D11_BLEND_ZERO,     // Ignore destination alpha
                    BlendOpAlpha: D3D11_BLEND_OP_ADD,     // Standard alpha operation
                    RenderTargetWriteMask: D3D11_COLOR_WRITE_ENABLE_ALL.0 as u8, // Write all RGBA
                },
                D3D11_RENDER_TARGET_BLEND_DESC::default(),
                D3D11_RENDER_TARGET_BLEND_DESC::default(),
                D3D11_RENDER_TARGET_BLEND_DESC::default(),
                D3D11_RENDER_TARGET_BLEND_DESC::default(),
                D3D11_RENDER_TARGET_BLEND_DESC::default(),
                D3D11_RENDER_TARGET_BLEND_DESC::default(),
                D3D11_RENDER_TARGET_BLEND_DESC::default(),
            ],
        };

        let mut blend_state = None;
        unsafe {
            device.CreateBlendState(&blend_desc, Some(&mut blend_state))?;
        }

        // Create render target view
        let back_buffer: ID3D11Texture2D = unsafe { swap_chain.GetBuffer(0)? };
        let mut render_target_view = None;
        unsafe {
            device.CreateRenderTargetView(&back_buffer, None, Some(&mut render_target_view))?;
        }

        // Log debug layer status and verify it's working
        #[cfg(any(debug_assertions, feature = "d3d11-debug"))]
        {
            println!("DirectX 11 Debug Layer: ENABLED");

            // Try to query the debug interface to verify it's actually working
            let debug_interface: windows::core::Result<
                windows::Win32::Graphics::Direct3D11::ID3D11Debug,
            > = device.cast();

            match debug_interface {
                Ok(debug) => {
                    println!("Debug interface successfully obtained - debug layer is active!");
                    // Force a validation message by trying to get info queue
                    let info_queue: windows::core::Result<
                        windows::Win32::Graphics::Direct3D11::ID3D11InfoQueue,
                    > = debug.cast();
                    if let Ok(_queue) = info_queue {
                        println!("Info queue available - validation messages will be captured");
                    }
                }
                Err(_) => println!(
                    "Warning: Debug interface not available - debug layer may not be working"
                ),
            }
        }
        #[cfg(not(any(debug_assertions, feature = "d3d11-debug")))]
        println!("DirectX 11 Debug Layer: DISABLED");

        let now = Instant::now();
        let mut context = D3D11Context {
            device,
            device_context,
            swap_chain,
            render_target_view,
            vertex_buffer: None,
            index_buffer: None,
            vertex_shader: None,
            pixel_shader: None,
            input_layout: None,
            texture: None,
            texture_view: None,
            sampler_state: None,
            constant_buffer: None,
            blend_state,
            window_width: width as f32,
            window_height: height as f32,
            dpi_scale,
            sprite_width: 128.0,  // Will be updated when texture is loaded
            sprite_height: 128.0, // Will be updated when texture is loaded
            sprites: Vec::new(),
            sprite_batch: SpriteBatch::new(MAX_SPRITES), // Support batching for all sprites
            vsync: true, // Default VSync enabled, will be overridden by command line
            last_time: now,
            frame_count: 0,
            last_log_time: now,
        };

        // Create quad resources (includes texture loading)
        unsafe {
            context.create_quad_resources()?;
        }

        // Parse command line arguments for configuration
        let config = parse_args();
        context.init_sprites(config.sprite_count);
        println!("Initialized {} sprites", config.sprite_count);

        // Store VSync setting for use in render loop
        context.vsync = config.vsync;
        println!(
            "Window client area: {}x{}",
            context.window_width, context.window_height
        );
        println!(
            "Sprite size: {}x{}",
            context.sprite_width, context.sprite_height
        );

        Ok(context)
    }

    fn init_sprites(&mut self, count: usize) {
        self.sprites.clear();
        for _ in 0..count {
            let mut sprite = Sprite::new();
            sprite.dpi_scale = self.dpi_scale;
            sprite.sprite_width = self.sprite_width;
            sprite.sprite_height = self.sprite_height;

            // Random spawn position across the window (like commented Go code)
            let sprite_width = sprite.sprite_width / sprite.dpi_scale;
            let sprite_height = sprite.sprite_height / sprite.dpi_scale;
            let half_width = sprite_width / 2.0;
            let half_height = sprite_height / 2.0;

            sprite.position[0] =
                half_width + rand::random::<f32>() * (self.window_width - sprite_width);
            sprite.position[1] =
                half_height + rand::random::<f32>() * (self.window_height - sprite_height);

            self.sprites.push(sprite);
        }
    }

    unsafe fn create_quad_resources(&mut self) -> Result<()> {
        // Load texture first to get dimensions
        unsafe {
            self.load_texture()?;
        }

        // Create dynamic vertex buffer for batching (can hold up to max_sprites)
        let max_vertices = self.sprite_batch.max_sprites * 4; // 4 vertices per sprite
        let vertex_buffer_desc = D3D11_BUFFER_DESC {
            ByteWidth: (std::mem::size_of::<Vertex>() * max_vertices) as u32,
            Usage: D3D11_USAGE_DYNAMIC, // Allow CPU updates
            BindFlags: D3D11_BIND_VERTEX_BUFFER.0 as u32,
            CPUAccessFlags: D3D11_CPU_ACCESS_WRITE.0 as u32, // CPU can write to buffer
            MiscFlags: 0,
            StructureByteStride: 0,
        };

        unsafe {
            self.device.CreateBuffer(
                &vertex_buffer_desc,
                None, // No initial data
                Some(&mut self.vertex_buffer),
            )?;
        }

        // Create dynamic index buffer for batching (can hold up to max_sprites)
        let max_indices = self.sprite_batch.max_sprites * 6; // 6 indices per sprite
        let index_buffer_desc = D3D11_BUFFER_DESC {
            ByteWidth: (std::mem::size_of::<u32>() * max_indices) as u32,
            Usage: D3D11_USAGE_DYNAMIC, // Allow CPU updates
            BindFlags: D3D11_BIND_INDEX_BUFFER.0 as u32,
            CPUAccessFlags: D3D11_CPU_ACCESS_WRITE.0 as u32, // CPU can write to buffer
            MiscFlags: 0,
            StructureByteStride: 0,
        };

        unsafe {
            self.device.CreateBuffer(
                &index_buffer_desc,
                None, // No initial data
                Some(&mut self.index_buffer),
            )?;
        }

        // Vertex shader source with simple orthogonal projection
        let vs_source = VERTEX_SHADER;

        // Pixel shader source
        let ps_source = PIXEL_SHADER;

        // Compile and create vertex shader
        let vs_blob = unsafe { self.compile_shader(vs_source, "main", "vs_5_0")? };
        unsafe {
            self.device.CreateVertexShader(
                std::slice::from_raw_parts(
                    vs_blob.GetBufferPointer() as *const u8,
                    vs_blob.GetBufferSize(),
                ),
                None,
                Some(&mut self.vertex_shader),
            )?;
        }

        // Compile and create pixel shader
        let ps_blob = unsafe { self.compile_shader(ps_source, "main", "ps_5_0")? };
        unsafe {
            self.device.CreatePixelShader(
                std::slice::from_raw_parts(
                    ps_blob.GetBufferPointer() as *const u8,
                    ps_blob.GetBufferSize(),
                ),
                None,
                Some(&mut self.pixel_shader),
            )?;
        }

        // Create input layout
        let input_desc = [
            D3D11_INPUT_ELEMENT_DESC {
                SemanticName: PCSTR(c"POSITION".as_ptr() as *const u8),
                SemanticIndex: 0,
                Format: DXGI_FORMAT_R32G32B32_FLOAT,
                InputSlot: 0,
                AlignedByteOffset: 0,
                InputSlotClass: D3D11_INPUT_PER_VERTEX_DATA,
                InstanceDataStepRate: 0,
            },
            D3D11_INPUT_ELEMENT_DESC {
                SemanticName: PCSTR(c"TEXCOORD".as_ptr() as *const u8),
                SemanticIndex: 0,
                Format: DXGI_FORMAT_R32G32_FLOAT,
                InputSlot: 0,
                AlignedByteOffset: 12, // 3 floats * 4 bytes
                InputSlotClass: D3D11_INPUT_PER_VERTEX_DATA,
                InstanceDataStepRate: 0,
            },
        ];

        unsafe {
            self.device.CreateInputLayout(
                &input_desc,
                std::slice::from_raw_parts(
                    vs_blob.GetBufferPointer() as *const u8,
                    vs_blob.GetBufferSize(),
                ),
                Some(&mut self.input_layout),
            )?;
        }

        // Texture already loaded in create_quad_resources

        // Create sampler state with point filtering for pixel-perfect rendering
        // Point filtering ensures crisp pixel art without blurring between texels
        // Sampler controls how texture pixels are interpolated when drawn at different sizes
        let sampler_desc = D3D11_SAMPLER_DESC {
            Filter: D3D11_FILTER_MIN_MAG_MIP_POINT, // Point filtering = sharp pixels (no blending)
            AddressU: D3D11_TEXTURE_ADDRESS_CLAMP,  // Clamp U coords to [0,1] range
            AddressV: D3D11_TEXTURE_ADDRESS_CLAMP,  // Clamp V coords to [0,1] range
            AddressW: D3D11_TEXTURE_ADDRESS_CLAMP,  // Not used for 2D textures
            MipLODBias: 0.0,
            MaxAnisotropy: 1,
            ComparisonFunc: D3D11_COMPARISON_NEVER,
            BorderColor: [0.0, 0.0, 0.0, 0.0],
            MinLOD: 0.0,
            MaxLOD: f32::MAX,
        };

        let mut sampler_state = None;
        unsafe {
            self.device
                .CreateSamplerState(&sampler_desc, Some(&mut sampler_state))?;
        }
        self.sampler_state = sampler_state;

        // Create constant buffer for transform matrix
        // Dynamic buffer allows CPU to update transform data each frame
        let cb_desc = D3D11_BUFFER_DESC {
            ByteWidth: std::mem::size_of::<TransformBuffer>() as u32,
            Usage: D3D11_USAGE_DYNAMIC, // CPU can write, GPU can read
            BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
            CPUAccessFlags: D3D11_CPU_ACCESS_WRITE.0 as u32, // CPU write access
            MiscFlags: 0,
            StructureByteStride: 0,
        };

        let mut constant_buffer = None;
        unsafe {
            self.device
                .CreateBuffer(&cb_desc, None, Some(&mut constant_buffer))?;
        }
        self.constant_buffer = constant_buffer;

        Ok(())
    }

    unsafe fn load_texture(&mut self) -> Result<()> {
        // Load the PNG image using the image crate
        let img = image::open("ferris_pixel_99x70_transparent.png")
            .map_err(|_e| Error::from_hresult(windows::core::HRESULT(-1)))?;

        // Convert to RGBA8 format (8 bits per channel, 32 bits per pixel)
        let img = img.to_rgba8();
        let (width, height) = img.dimensions();
        let pixels = img.as_raw(); // Raw pixel data as bytes

        // Update sprite dimensions based on actual PNG size
        self.sprite_width = width as f32;
        self.sprite_height = height as f32;
        println!("Loaded sprite: {width}x{height} pixels");

        // Create texture description - tells DirectX about the image format
        // This configures GPU memory layout and usage for the sprite texture
        let texture_desc = D3D11_TEXTURE2D_DESC {
            Width: width,                       // Texture width in pixels
            Height: height,                     // Texture height in pixels
            MipLevels: 1,                       // No mipmaps (single detail level)
            ArraySize: 1,                       // Single texture, not texture array
            Format: DXGI_FORMAT_R8G8B8A8_UNORM, // 32-bit RGBA (8 bits per channel)
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,   // No multisampling - single sample per pixel
                Quality: 0, // No quality settings needed
            },
            Usage: D3D11_USAGE_DEFAULT, // Standard GPU-only memory (fastest)
            BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32, // Can be bound as texture in shaders
            CPUAccessFlags: 0,          // CPU cannot access after creation (immutable)
            MiscFlags: 0,
        };

        // Initial texture data - provides the pixel data to copy to GPU
        // Initial texture data from PNG file - upload pixel data to GPU
        let subresource_data = D3D11_SUBRESOURCE_DATA {
            pSysMem: pixels.as_ptr() as *const _, // Pointer to pixel data array
            SysMemPitch: width * 4,               // Row pitch: 4 bytes per pixel (RGBA)
            SysMemSlicePitch: 0,                  // Not used for 2D textures
        };

        // Create texture in GPU memory
        unsafe {
            self.device.CreateTexture2D(
                &texture_desc,
                Some(&subresource_data),
                Some(&mut self.texture),
            )?;
        }

        // Create shader resource view - interface for shaders to read texture
        // This creates a "view" that allows pixel shaders to sample from the texture
        if let Some(texture) = &self.texture {
            unsafe {
                self.device.CreateShaderResourceView(
                    texture,
                    None, // Use default view format and mip levels (entire texture)
                    Some(&mut self.texture_view),
                )?;
            }
        }

        Ok(())
    }

    /// Compile HLSL shader source code into GPU-executable bytecode
    /// This converts human-readable HLSL into optimized machine code for the GPU
    unsafe fn compile_shader(
        &self,
        source: &str,
        entry_point: &str,
        target: &str,
    ) -> Result<ID3DBlob> {
        let mut blob = None; // Will contain compiled shader bytecode
        let mut error_blob = None; // Will contain error messages if compilation fails

        // Convert Rust strings to C strings for DirectX API compatibility
        let source_cstr = std::ffi::CString::new(source).unwrap();
        let entry_cstr = std::ffi::CString::new(entry_point).unwrap();
        let target_cstr = std::ffi::CString::new(target).unwrap();

        // Compile HLSL shader source code to GPU bytecode using DirectX compiler
        // D3DCompile transforms HLSL into optimized GPU instructions
        let result = unsafe {
            D3DCompile(
                source_cstr.as_ptr() as *const _,         // HLSL source code string
                source.len(),                             // Source code length in bytes
                None, // Source file name (not needed for strings)
                None, // Preprocessor macro definitions (none)
                None, // Include file handler (none needed)
                PCSTR(entry_cstr.as_ptr() as *const u8), // Entry point function name ("main")
                PCSTR(target_cstr.as_ptr() as *const u8), // Shader model (vs_5_0, ps_5_0)
                0,    // Compile flags
                0,    // Effect flags
                &mut blob, // Compiled bytecode output
                Some(&mut error_blob), // Error messages output
            )
        };

        // Handle compilation errors
        if result.is_err() {
            if let Some(error_blob) = error_blob {
                let error_msg = unsafe {
                    let ptr = error_blob.GetBufferPointer() as *const u8;
                    let len = error_blob.GetBufferSize();
                    std::slice::from_raw_parts(ptr, len)
                };
                println!(
                    "Shader compilation error: {}",
                    String::from_utf8_lossy(error_msg)
                );
            }
            return Err(result.unwrap_err());
        }

        Ok(blob.unwrap())
    }

    fn update(&mut self) {
        let current_time = Instant::now();
        let dt = current_time.duration_since(self.last_time).as_secs_f32();
        self.last_time = current_time;

        // Update all sprites
        for sprite in &mut self.sprites {
            sprite.update(dt, self.window_width, self.window_height);
        }

        // Log sprites per second (throughput benchmark like Go example)
        self.frame_count += 1;
        if current_time.duration_since(self.last_log_time).as_secs() >= 1 {
            let fps = self.frame_count as f32
                / current_time
                    .duration_since(self.last_log_time)
                    .as_secs_f32();
            let sprites_per_second = fps * self.sprites.len() as f32;
            println!(
                "Sprites/sec: {:.0} | FPS: {:.1} | Sprites: {} | Frame time: {:.2}ms",
                sprites_per_second,
                fps,
                self.sprites.len(),
                1000.0 / fps
            );
            self.frame_count = 0;
            self.last_log_time = current_time;
        }
    }

    /// Update the sprite batch with current sprite data and upload to GPU
    unsafe fn update_sprite_batch(&mut self) -> Result<()> {
        // Clear the batch and rebuild it with current sprite positions
        self.sprite_batch.clear();

        // Get physical sprite dimensions (adjusted for DPI scaling)
        let physical_sprite_width = self.sprite_width / self.dpi_scale;
        let physical_sprite_height = self.sprite_height / self.dpi_scale;

        // Add each sprite to the batch
        for sprite in &self.sprites {
            // Convert sprite center position to top-left corner for rendering
            let x = sprite.position[0] - physical_sprite_width / 2.0;
            let y = sprite.position[1] - physical_sprite_height / 2.0;

            self.sprite_batch.add(
                x,
                y,
                physical_sprite_width,
                physical_sprite_height,
                self.window_width,
                self.window_height,
            );
        }

        // Upload vertex data to GPU if we have sprites to render
        if !self.sprite_batch.is_empty() {
            if let Some(vertex_buffer) = &self.vertex_buffer {
                let mut mapped_resource = D3D11_MAPPED_SUBRESOURCE::default();
                unsafe {
                    self.device_context.Map(
                        vertex_buffer,
                        0,
                        D3D11_MAP_WRITE_DISCARD,
                        0,
                        Some(&mut mapped_resource),
                    )?;

                    // Copy vertex data to GPU memory
                    let vertex_data = self.sprite_batch.vertices.as_ptr() as *const u8;
                    let vertex_size =
                        std::mem::size_of::<Vertex>() * self.sprite_batch.vertices.len();
                    std::ptr::copy_nonoverlapping(
                        vertex_data,
                        mapped_resource.pData as *mut u8,
                        vertex_size,
                    );

                    self.device_context.Unmap(vertex_buffer, 0);
                }
            }
        }

        // Upload index data to GPU if we have sprites to render
        if !self.sprite_batch.is_empty() {
            if let Some(index_buffer) = &self.index_buffer {
                let mut mapped_resource = D3D11_MAPPED_SUBRESOURCE::default();
                unsafe {
                    self.device_context.Map(
                        index_buffer,
                        0,
                        D3D11_MAP_WRITE_DISCARD,
                        0,
                        Some(&mut mapped_resource),
                    )?;

                    // Copy index data to GPU memory
                    let index_data = self.sprite_batch.indices.as_ptr() as *const u8;
                    let index_size = std::mem::size_of::<u32>() * self.sprite_batch.indices.len();
                    std::ptr::copy_nonoverlapping(
                        index_data,
                        mapped_resource.pData as *mut u8,
                        index_size,
                    );

                    self.device_context.Unmap(index_buffer, 0);
                }
            }
        }

        Ok(())
    }

    /// Render all sprites using the DirectX 11 graphics pipeline
    /// This sets up the complete rendering state and draws all batched sprites
    unsafe fn render(&self) {
        if let Some(rtv) = &self.render_target_view {
            // Clear the render target to a solid color (cornflower blue)
            // This erases the previous frame and provides a clean background
            let clear_color = [0.39, 0.58, 0.93, 1.0]; // RGBA values [0.0, 1.0]
            unsafe {
                self.device_context.ClearRenderTargetView(rtv, &clear_color);

                // === OUTPUT MERGER STAGE ===
                // Configure where final pixels are written
                self.device_context
                    .OMSetRenderTargets(Some(&[Some(rtv.clone())]), None);

                // Enable alpha blending for sprite transparency
                // Uses the blend state configured during initialization
                if let Some(blend_state) = &self.blend_state {
                    let blend_factor = [0.0, 0.0, 0.0, 0.0]; // Not used for standard alpha blend
                    self.device_context.OMSetBlendState(
                        blend_state,
                        Some(&blend_factor),
                        0xffffffff, // Write to all color channels (RGBA)
                    );
                }

                // === INPUT ASSEMBLER STAGE ===
                // Configure vertex data and how it forms primitives

                // Bind vertex buffer containing all sprite quad vertices
                if let Some(vertex_buffer) = &self.vertex_buffer {
                    let stride = std::mem::size_of::<Vertex>() as u32; // Bytes per vertex
                    let offset = 0; // Start at beginning of buffer
                    self.device_context.IASetVertexBuffers(
                        0, // Input slot 0 (can have multiple vertex streams)
                        1, // Number of buffers to bind
                        Some(&Some(vertex_buffer.clone())),
                        Some(&stride),
                        Some(&offset),
                    );
                }

                // Bind index buffer that defines triangle connectivity
                // Indices reference vertices in the vertex buffer to form triangles
                if let Some(index_buffer) = &self.index_buffer {
                    self.device_context
                        .IASetIndexBuffer(index_buffer, DXGI_FORMAT_R32_UINT, 0);
                }

                // Bind input layout that describes vertex structure to vertex shader
                // This maps vertex buffer data fields to shader input variables
                if let Some(input_layout) = &self.input_layout {
                    self.device_context.IASetInputLayout(input_layout);
                }

                // Set primitive topology - how vertices form geometric shapes
                // TRIANGLELIST: Every 3 indices form one triangle (most common for 2D)
                self.device_context
                    .IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

                // === VERTEX SHADER STAGE ===
                // Transform vertices from local space to screen space
                if let Some(vertex_shader) = &self.vertex_shader {
                    self.device_context.VSSetShader(vertex_shader, None);
                }

                // === PIXEL SHADER STAGE ===
                // Calculate final color for each pixel covered by triangles
                if let Some(pixel_shader) = &self.pixel_shader {
                    self.device_context.PSSetShader(pixel_shader, None);
                }

                // Bind sprite texture to pixel shader for sampling
                // Texture slot 0 corresponds to "texture0" sampler in pixel shader
                if let Some(texture_view) = &self.texture_view {
                    self.device_context
                        .PSSetShaderResources(0, Some(&[Some(texture_view.clone())]));
                }

                // Bind sampler state that controls texture filtering
                // Point filtering ensures crisp pixel art without blurring
                if let Some(sampler_state) = &self.sampler_state {
                    self.device_context
                        .PSSetSamplers(0, Some(&[Some(sampler_state.clone())]));
                }

                // === RASTERIZER STAGE ===
                // Configure viewport that maps NDC coordinates to screen pixels
                let viewport = D3D11_VIEWPORT {
                    TopLeftX: 0.0,              // Left edge of viewport
                    TopLeftY: 0.0,              // Top edge of viewport
                    Width: self.window_width,   // Viewport width in pixels
                    Height: self.window_height, // Viewport height in pixels
                    MinDepth: 0.0,              // Near depth plane (0.0 = closest)
                    MaxDepth: 1.0,              // Far depth plane (1.0 = farthest)
                };
                self.device_context.RSSetViewports(Some(&[viewport]));

                // === DRAW CALL ===
                // Render all sprites in a single efficient batch
                if !self.sprite_batch.is_empty() {
                    // Set identity transform matrix since vertices are pre-transformed to NDC
                    // SpriteBatch already converted pixel coordinates to NDC space
                    let identity_matrix = [
                        1.0, 0.0, 0.0, 0.0, // Row 1: No X transformation
                        0.0, 1.0, 0.0, 0.0, // Row 2: No Y transformation
                        0.0, 0.0, 1.0, 0.0, // Row 3: No Z transformation
                        0.0, 0.0, 0.0, 1.0, // Row 4: Homogeneous coordinate
                    ];
                    let transform_buffer = TransformBuffer {
                        transform: identity_matrix,
                    };

                    // Upload transform matrix to GPU constant buffer
                    // This makes the matrix available to the vertex shader
                    if let Some(constant_buffer) = &self.constant_buffer {
                        let mut mapped_resource = D3D11_MAPPED_SUBRESOURCE::default();
                        if self
                            .device_context
                            .Map(
                                constant_buffer,
                                0,
                                D3D11_MAP_WRITE_DISCARD, // Discard old data for performance
                                0,
                                Some(&mut mapped_resource),
                            )
                            .is_ok()
                        {
                            // Copy transform data to GPU memory
                            let dst = mapped_resource.pData as *mut TransformBuffer;
                            ptr::copy_nonoverlapping(&transform_buffer, dst, 1);
                            self.device_context.Unmap(constant_buffer, 0);
                        }

                        // Bind constant buffer to vertex shader slot 0
                        self.device_context
                            .VSSetConstantBuffers(0, Some(&[Some(constant_buffer.clone())]));
                    }

                    // Execute the draw call - render all sprites with a single GPU command
                    // This is the key optimization: 1000s of sprites = 1 draw call instead of 1000s
                    let index_count = (self.sprite_batch.sprite_count() * 6) as u32;
                    self.device_context.DrawIndexed(index_count, 0, 0);
                }

                // === FRAME PRESENTATION ===
                // Present the completed frame to the screen
                // VSync control allows performance testing with uncapped framerate
                let sync_interval = if self.vsync { 1 } else { 0 };
                let _ = self.swap_chain.Present(sync_interval, 0);
            }
        }
    }

    unsafe fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        if width == 0 || height == 0 {
            return Ok(());
        }

        // Update stored window dimensions
        self.window_width = width as f32;
        self.window_height = height as f32;

        // Release the old render target view
        self.render_target_view = None;

        // Resize the swap chain buffers
        unsafe {
            self.swap_chain
                .ResizeBuffers(0, width, height, DXGI_FORMAT_UNKNOWN, 0)?;
        }

        // Recreate the render target view
        let back_buffer: ID3D11Texture2D = unsafe { self.swap_chain.GetBuffer(0)? };
        let mut render_target_view = None;
        unsafe {
            self.device.CreateRenderTargetView(
                &back_buffer,
                None,
                Some(&mut render_target_view),
            )?;
        }
        self.render_target_view = render_target_view;

        // Recreate quad with new window dimensions for proper NDC calculations
        unsafe {
            self.create_quad_resources()?;
        }

        Ok(())
    }
}

static mut D3D_CONTEXT: Option<D3D11Context> = None;

/// Configuration parsed from command line arguments
#[derive(Debug)]
struct Config {
    sprite_count: usize,
    vsync: bool,
}

/// Parse command line arguments and return configuration
/// Usage: ferris-mark-dx [sprite_count] [--vsync-off]
fn parse_args() -> Config {
    let args: Vec<String> = env::args().collect();
    let mut config = Config {
        sprite_count: 100, // Default sprite count
        vsync: true,       // Default VSync enabled
    };

    // Parse sprite count from first argument
    if args.len() > 1 {
        match args[1].parse::<usize>() {
            Ok(count) => {
                if count > 0 && count <= MAX_SPRITES {
                    config.sprite_count = count;
                } else {
                    println!(
                        "Warning: Sprite count must be between 1 and {MAX_SPRITES}. Using default: 100"
                    );
                }
            }
            Err(_) => {
                println!(
                    "Warning: Invalid sprite count '{}'. Using default: 100",
                    args[1]
                );
            }
        }
    }

    // Check for VSync option
    if args.contains(&"--vsync-off".to_string()) {
        config.vsync = false;
        println!("VSync disabled");
    }

    config
}

fn main() -> Result<()> {
    unsafe {
        // Get the module handle
        let instance = GetModuleHandleW(None)?;

        // Define the window class name
        let window_class = w!("FerrisMarkD3D11Window");

        // Create the window class
        let wc = WNDCLASSW {
            lpfnWndProc: Some(wndproc),
            hInstance: instance.into(),
            lpszClassName: window_class,
            hbrBackground: HBRUSH(0), // No background brush - we'll handle rendering
            hCursor: LoadCursorW(None, IDC_ARROW)?,
            ..Default::default()
        };

        // Register the window class
        let atom = RegisterClassW(&wc);
        debug_assert!(atom != 0);

        // Set process DPI awareness
        let _ = SetProcessDpiAwareness(PROCESS_SYSTEM_DPI_AWARE);

        // Create the window
        let hwnd = CreateWindowExW(
            WINDOW_EX_STYLE::default(),
            window_class,
            w!("Ferris Mark - Direct3D 11"),
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            1920,
            1080,
            None,
            None,
            instance,
            None,
        );

        if hwnd.0 == 0 {
            return Err(Error::from_win32());
        }

        // Initialize Direct3D 11
        let d3d_context = D3D11Context::new(hwnd)?;
        D3D_CONTEXT = Some(d3d_context);

        // Show the window
        let _ = ShowWindow(hwnd, SW_SHOW);
        let _ = UpdateWindow(hwnd);

        // Message loop with rendering
        let mut message = MSG::default();
        loop {
            // Process all pending messages
            while PeekMessageW(&mut message, None, 0, 0, PM_REMOVE).into() {
                if message.message == WM_QUIT {
                    break;
                }
                let _ = TranslateMessage(&message);
                DispatchMessageW(&message);
            }

            if message.message == WM_QUIT {
                break;
            }

            // Update and render
            let context_ptr = std::ptr::addr_of_mut!(D3D_CONTEXT);
            if let Some(context) = (*context_ptr).as_mut() {
                context.update();
                let _ = context.update_sprite_batch();
                context.render();
            }
        }

        // Cleanup
        D3D_CONTEXT = None;

        Ok(())
    }
}

extern "system" fn wndproc(window: HWND, message: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    unsafe {
        match message {
            WM_DESTROY => {
                PostQuitMessage(0);
                LRESULT(0)
            }
            WM_CLOSE => {
                let _ = DestroyWindow(window);
                LRESULT(0)
            }
            WM_KEYDOWN => DefWindowProcW(window, message, wparam, lparam),
            WM_SYSKEYDOWN => {
                // Handle system key combinations like Alt+F4
                if wparam.0 == VK_F4.0 as usize {
                    let _ = PostMessageW(window, WM_CLOSE, WPARAM(0), LPARAM(0));
                    return LRESULT(0);
                }
                DefWindowProcW(window, message, wparam, lparam)
            }
            WM_PAINT => {
                // Use Direct3D for rendering instead of GDI
                let d3d_ptr = std::ptr::addr_of!(D3D_CONTEXT);
                if let Some(d3d_context) = (*d3d_ptr).as_ref() {
                    d3d_context.render();
                }

                // Still need to validate the window for Windows
                let mut ps = PAINTSTRUCT::default();
                let _hdc = BeginPaint(window, &mut ps);
                let _ = EndPaint(window, &ps);

                LRESULT(0)
            }
            WM_SIZE => {
                // Handle window resize
                let d3d_ptr = std::ptr::addr_of_mut!(D3D_CONTEXT);
                if let Some(d3d_context) = (*d3d_ptr).as_mut() {
                    let width = (lparam.0 & 0xFFFF) as u32;
                    let height = ((lparam.0 >> 16) & 0xFFFF) as u32;

                    if width > 0 && height > 0 {
                        let _ = d3d_context.resize(width, height);
                    }
                }
                DefWindowProcW(window, message, wparam, lparam)
            }
            _ => DefWindowProcW(window, message, wparam, lparam),
        }
    }
}

// HLSL Vertex Shader - transforms vertex positions from object space to screen space
// Runs once per vertex (4 times per sprite quad) on the GPU in parallel
const VERTEX_SHADER: &str = r#"
// Constant buffer: data shared across all vertices for this draw call
cbuffer TransformBuffer : register(b0)
{
    matrix transform;  // 4x4 matrix for coordinate space conversion
};

// Input: data from vertex buffer for each vertex
struct VS_INPUT
{
    float3 position : POSITION;    // Vertex position (x, y, z) in NDC space
    float2 texCoord : TEXCOORD0;   // Texture coordinates (u, v) in [0,1] range
};

// Output: data passed to rasterizer and pixel shader
struct VS_OUTPUT
{
    float4 position : SV_POSITION; // Final screen position (required by GPU)
    float2 texCoord : TEXCOORD0;   // UV coordinates for texture sampling
};

VS_OUTPUT main(VS_INPUT input)
{
    VS_OUTPUT output;

    // Transform vertex position by sprite transform matrix
    // Since vertices are pre-transformed to NDC, this is typically identity
    output.position = mul(float4(input.position, 1.0), transform);

    // Pass texture coordinates unchanged to pixel shader
    // These will be interpolated across the triangle surface
    output.texCoord = input.texCoord;

    return output;
}
"#;

// HLSL Pixel Shader - determines final color for each pixel covered by triangles
// Runs massively in parallel (once per pixel) on GPU shader cores
const PIXEL_SHADER: &str = r#"
// Resources: texture and sampler state bound from CPU
Texture2D spriteTexture : register(t0);    // Sprite PNG texture in slot 0
SamplerState spriteSampler : register(s0); // Point filtering sampler in slot 0

// Input: interpolated values from vertex shader output
struct PS_INPUT
{
    float4 position : SV_POSITION; // Screen position (not used in this shader)
    float2 texCoord : TEXCOORD0;   // Interpolated UV coordinates [0,1]
};

float4 main(PS_INPUT input) : SV_TARGET
{
    // Sample the sprite texture at the interpolated UV coordinates
    // Point filtering (from sampler state) ensures crisp pixel art rendering
    // Returns RGBA color: RGB from texture + Alpha for transparency
    return spriteTexture.Sample(spriteSampler, input.texCoord);
}
"#;
