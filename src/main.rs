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

// SpriteBatch for efficient sprite rendering
// Collects multiple sprites into a single draw call
struct SpriteBatch {
    vertices: Vec<Vertex>,
    indices: Vec<u16>,
    max_sprites: usize,
}

impl SpriteBatch {
    fn new(max_sprites: usize) -> Self {
        Self {
            vertices: Vec::with_capacity(max_sprites * 4), // 4 vertices per sprite
            indices: Vec::with_capacity(max_sprites * 6),  // 6 indices per sprite (2 triangles)
            max_sprites,
        }
    }

    fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
    }

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
            return; // Batch is full
        }

        let current_sprite = (self.vertices.len() / 4) as u16;
        let base_index = current_sprite * 4;

        // Convert pixel coordinates to NDC coordinates [-1, 1]
        let left_ndc = (x / window_width) * 2.0 - 1.0;
        let right_ndc = ((x + width) / window_width) * 2.0 - 1.0;
        let top_ndc = 1.0 - (y / window_height) * 2.0;
        let bottom_ndc = 1.0 - ((y + height) / window_height) * 2.0;

        // Add 4 vertices for the sprite quad
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

        // Add 6 indices for 2 triangles (quad)
        self.indices.extend_from_slice(&[
            base_index + 0,
            base_index + 1,
            base_index + 3, // first triangle
            base_index + 1,
            base_index + 2,
            base_index + 3, // second triangle
        ]);
    }

    fn sprite_count(&self) -> usize {
        self.vertices.len() / 4
    }

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

        // Swap chain description
        let swap_chain_desc = DXGI_SWAP_CHAIN_DESC {
            BufferDesc: DXGI_MODE_DESC {
                Width: width,
                Height: height,
                RefreshRate: DXGI_RATIONAL {
                    Numerator: 60,
                    Denominator: 1,
                },
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                ScanlineOrdering: DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED,
                Scaling: DXGI_MODE_SCALING_UNSPECIFIED,
            },
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
            BufferCount: 1,
            OutputWindow: hwnd,
            Windowed: TRUE,
            SwapEffect: DXGI_SWAP_EFFECT_DISCARD,
            Flags: 0,
        };

        // Create device, device context, and swap chain
        // Enable debug layer in debug builds or when d3d11-debug feature is enabled
        #[cfg(any(debug_assertions, feature = "d3d11-debug"))]
        let create_device_flags = D3D11_CREATE_DEVICE_DEBUG;
        #[cfg(not(any(debug_assertions, feature = "d3d11-debug")))]
        let create_device_flags = D3D11_CREATE_DEVICE_FLAG(0);

        unsafe {
            D3D11CreateDeviceAndSwapChain(
                None,
                D3D_DRIVER_TYPE_HARDWARE,
                None,
                create_device_flags,
                None,
                D3D11_SDK_VERSION,
                Some(&swap_chain_desc),
                Some(&mut swap_chain),
                Some(&mut device),
                Some(ptr::null_mut()),
                Some(&mut device_context),
            )?;
        }

        let device = device.unwrap();
        let device_context = device_context.unwrap();
        let swap_chain = swap_chain.unwrap();

        // Create blend state for alpha transparency
        let blend_desc = D3D11_BLEND_DESC {
            AlphaToCoverageEnable: FALSE,
            IndependentBlendEnable: FALSE,
            RenderTarget: [
                D3D11_RENDER_TARGET_BLEND_DESC {
                    BlendEnable: TRUE,
                    SrcBlend: D3D11_BLEND_SRC_ALPHA,
                    DestBlend: D3D11_BLEND_INV_SRC_ALPHA,
                    BlendOp: D3D11_BLEND_OP_ADD,
                    SrcBlendAlpha: D3D11_BLEND_ONE,
                    DestBlendAlpha: D3D11_BLEND_ZERO,
                    BlendOpAlpha: D3D11_BLEND_OP_ADD,
                    RenderTargetWriteMask: D3D11_COLOR_WRITE_ENABLE_ALL.0 as u8,
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
            sprite_batch: SpriteBatch::new(10000), // Support up to 10,000 sprites
            last_time: now,
            frame_count: 0,
            last_log_time: now,
        };

        // Create quad resources (includes texture loading)
        unsafe {
            context.create_quad_resources()?;
        }

        // Initialize sprites with count from command line or default (after texture loading)
        let sprite_count = get_sprite_count();
        context.init_sprites(sprite_count);
        println!("Initialized {sprite_count} sprites");
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
            ByteWidth: (std::mem::size_of::<u16>() * max_indices) as u32,
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
        let texture_desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,                       // No mipmaps (detail levels)
            ArraySize: 1,                       // Single texture, not array
            Format: DXGI_FORMAT_R8G8B8A8_UNORM, // RGBA, 8 bits per channel
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1, // No multisampling
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT, // Standard GPU-only memory
            BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32, // Used as texture in shaders
            CPUAccessFlags: 0,          // CPU cannot access after creation
            MiscFlags: 0,
        };

        // Initial texture data - provides the pixel data to copy to GPU
        let texture_data = D3D11_SUBRESOURCE_DATA {
            pSysMem: pixels.as_ptr() as *const _,
            SysMemPitch: width * 4, // 4 bytes per pixel (RGBA), row stride
            SysMemSlicePitch: 0,    // Not used for 2D textures
        };

        // Create texture in GPU memory
        unsafe {
            self.device.CreateTexture2D(
                &texture_desc,
                Some(&texture_data),
                Some(&mut self.texture),
            )?;
        }

        // Create shader resource view - interface for shaders to read texture
        // Think of this as a "view" or "window" into the texture data
        if let Some(texture) = &self.texture {
            unsafe {
                self.device.CreateShaderResourceView(
                    texture,
                    None, // Use default view (entire texture)
                    Some(&mut self.texture_view),
                )?;
            }
        }

        Ok(())
    }

    unsafe fn compile_shader(
        &self,
        source: &str,
        entry_point: &str,
        target: &str,
    ) -> Result<ID3DBlob> {
        let mut blob = None; // Will contain compiled shader bytecode
        let mut error_blob = None; // Will contain error messages if compilation fails

        // Convert Rust strings to C strings for DirectX API
        let source_cstr = std::ffi::CString::new(source).unwrap();
        let entry_cstr = std::ffi::CString::new(entry_point).unwrap();
        let target_cstr = std::ffi::CString::new(target).unwrap();

        // Compile HLSL shader source code to GPU bytecode
        let result = unsafe {
            D3DCompile(
                source_cstr.as_ptr() as *const _,         // HLSL source code
                source.len(),                             // Source length
                None,                                     // Source file name (optional)
                None,                                     // Macro definitions (optional)
                None,                                     // Include handler (optional)
                PCSTR(entry_cstr.as_ptr() as *const u8),  // Entry point function name
                PCSTR(target_cstr.as_ptr() as *const u8), // Shader model (vs_5_0, ps_5_0)
                0,                                        // Compile flags
                0,                                        // Effect flags
                &mut blob,                                // Compiled bytecode output
                Some(&mut error_blob),                    // Error messages output
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

    // Update the sprite batch with current sprite data
    unsafe fn update_sprite_batch(&mut self) -> Result<()> {
        // Clear the batch and rebuild it
        self.sprite_batch.clear();

        // Get physical sprite dimensions (adjusted for DPI)
        let physical_sprite_width = self.sprite_width / self.dpi_scale;
        let physical_sprite_height = self.sprite_height / self.dpi_scale;

        // Add each sprite to the batch
        for sprite in &self.sprites {
            // Convert sprite center position to top-left corner for batch
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

        // Update vertex buffer with batch data
        if !self.sprite_batch.is_empty() {
            if let Some(vertex_buffer) = &self.vertex_buffer {
                let mut mapped_resource = D3D11_MAPPED_SUBRESOURCE::default();
                self.device_context.Map(
                    vertex_buffer,
                    0,
                    D3D11_MAP_WRITE_DISCARD,
                    0,
                    Some(&mut mapped_resource),
                )?;

                // Copy vertex data
                let vertex_data = self.sprite_batch.vertices.as_ptr() as *const u8;
                let vertex_size = std::mem::size_of::<Vertex>() * self.sprite_batch.vertices.len();
                std::ptr::copy_nonoverlapping(
                    vertex_data,
                    mapped_resource.pData as *mut u8,
                    vertex_size,
                );

                self.device_context.Unmap(vertex_buffer, 0);
            }
        }

        // Update index buffer with batch data
        if !self.sprite_batch.is_empty() {
            if let Some(index_buffer) = &self.index_buffer {
                let mut mapped_resource = D3D11_MAPPED_SUBRESOURCE::default();
                self.device_context.Map(
                    index_buffer,
                    0,
                    D3D11_MAP_WRITE_DISCARD,
                    0,
                    Some(&mut mapped_resource),
                )?;

                // Copy index data
                let index_data = self.sprite_batch.indices.as_ptr() as *const u8;
                let index_size = std::mem::size_of::<u16>() * self.sprite_batch.indices.len();
                std::ptr::copy_nonoverlapping(
                    index_data,
                    mapped_resource.pData as *mut u8,
                    index_size,
                );

                self.device_context.Unmap(index_buffer, 0);
            }
        }

        Ok(())
    }

    unsafe fn render(&self) {
        if let Some(rtv) = &self.render_target_view {
            // Clear the render target to a solid color (cornflower blue)
            let clear_color = [0.39, 0.58, 0.93, 1.0]; // RGBA values [0.0, 1.0]
            unsafe {
                self.device_context.ClearRenderTargetView(rtv, &clear_color);

                // Set up the rendering pipeline - Output Merger stage
                // This tells DirectX where to draw pixels (render target)
                self.device_context
                    .OMSetRenderTargets(Some(&[Some(rtv.clone())]), None);

                // Set blend state for alpha transparency
                if let Some(blend_state) = &self.blend_state {
                    let blend_factor = [0.0, 0.0, 0.0, 0.0]; // Not used for our blend mode
                    self.device_context.OMSetBlendState(
                        blend_state,
                        Some(&blend_factor),
                        0xffffffff,
                    );
                }

                // Set vertex buffer - Input Assembler stage
                // Tells GPU where to find vertex data and how to interpret it
                if let Some(vertex_buffer) = &self.vertex_buffer {
                    let stride = std::mem::size_of::<Vertex>() as u32; // Size of each vertex
                    let offset = 0; // Start at beginning of buffer
                    self.device_context.IASetVertexBuffers(
                        0, // Input slot 0
                        1, // One buffer
                        Some(&Some(vertex_buffer.clone())),
                        Some(&stride),
                        Some(&offset),
                    );
                }

                // Set index buffer - defines triangle connectivity
                // Each index refers to a vertex in the vertex buffer
                if let Some(index_buffer) = &self.index_buffer {
                    self.device_context
                        .IASetIndexBuffer(index_buffer, DXGI_FORMAT_R16_UINT, 0);
                }

                // Set input layout - describes vertex structure to vertex shader
                // Maps vertex buffer data to shader input parameters
                if let Some(input_layout) = &self.input_layout {
                    self.device_context.IASetInputLayout(input_layout);
                }

                // Set primitive topology - how vertices form shapes
                // TRIANGLELIST: Every 3 vertices form a triangle
                self.device_context
                    .IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

                // Set shaders - programs that run on GPU
                // Vertex shader: transforms vertex positions
                if let Some(vertex_shader) = &self.vertex_shader {
                    self.device_context.VSSetShader(vertex_shader, None);
                }
                // Pixel shader: determines final pixel colors
                if let Some(pixel_shader) = &self.pixel_shader {
                    self.device_context.PSSetShader(pixel_shader, None);
                }

                // Set texture and sampler for pixel shader
                // Texture: the sprite image data
                if let Some(texture_view) = &self.texture_view {
                    self.device_context
                        .PSSetShaderResources(0, Some(&[Some(texture_view.clone())]));
                }
                // Sampler: controls how texture is filtered when sampled
                if let Some(sampler_state) = &self.sampler_state {
                    self.device_context
                        .PSSetSamplers(0, Some(&[Some(sampler_state.clone())]));
                }

                // Set viewport - defines screen area for rendering
                // Maps NDC coordinates [-1,1] to pixel coordinates [0,width/height]
                let viewport = D3D11_VIEWPORT {
                    TopLeftX: 0.0,              // Left edge of viewport
                    TopLeftY: 0.0,              // Top edge of viewport
                    Width: self.window_width,   // Viewport width in pixels
                    Height: self.window_height, // Viewport height in pixels
                    MinDepth: 0.0,              // Near depth (0.0 = closest)
                    MaxDepth: 1.0,              // Far depth (1.0 = farthest)
                };
                self.device_context.RSSetViewports(Some(&[viewport]));

                // Draw all sprites in a single batch
                // Use identity transform since vertices are already in NDC space
                if !self.sprite_batch.is_empty() {
                    // Set identity matrix for transform (no transformation needed)
                    let identity_matrix = [
                        1.0, 0.0, 0.0, 0.0, // Row 1: Identity
                        0.0, 1.0, 0.0, 0.0, // Row 2: Identity
                        0.0, 0.0, 1.0, 0.0, // Row 3: Identity
                        0.0, 0.0, 0.0, 1.0, // Row 4: Identity
                    ];
                    let transform_buffer = TransformBuffer {
                        transform: identity_matrix,
                    };

                    // Update constant buffer with screen-space transform
                    if let Some(constant_buffer) = &self.constant_buffer {
                        let mut mapped_resource = D3D11_MAPPED_SUBRESOURCE::default();
                        if self
                            .device_context
                            .Map(
                                constant_buffer,
                                0,
                                D3D11_MAP_WRITE_DISCARD,
                                0,
                                Some(&mut mapped_resource),
                            )
                            .is_ok()
                        {
                            let dst = mapped_resource.pData as *mut TransformBuffer;
                            ptr::copy_nonoverlapping(&transform_buffer, dst, 1);
                            self.device_context.Unmap(constant_buffer, 0);
                        }

                        self.device_context
                            .VSSetConstantBuffers(0, Some(&[Some(constant_buffer.clone())]));
                    }

                    // Draw all sprites with single draw call
                    let index_count = (self.sprite_batch.sprite_count() * 6) as u32;
                    self.device_context.DrawIndexed(index_count, 0, 0);
                }

                // Present the frame with VSync enabled
                // Present(sync_interval, flags): 1 = VSync on, 0 = VSync off
                // VSync caps framerate to display refresh rate (usually 60fps)
                let _ = self.swap_chain.Present(1, 0);
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

fn get_sprite_count() -> usize {
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 {
        match args[1].parse::<usize>() {
            Ok(count) => {
                if count > 0 && count <= 100000 {
                    count
                } else {
                    println!(
                        "Warning: Sprite count must be between 1 and 100000. Using default: 100"
                    );
                    100
                }
            }
            Err(_) => {
                println!(
                    "Warning: Invalid sprite count '{}'. Using default: 100",
                    args[1]
                );
                100
            }
        }
    } else {
        100 // Default sprite count for benchmarking
    }
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

// HLSL Vertex Shader - runs once per vertex, transforms positions
// Constant buffer contains per-sprite transform matrix from CPU
const VERTEX_SHADER: &str = r#"
cbuffer TransformBuffer : register(b0)
{
    matrix transform;  // 4x4 transform matrix for sprite positioning
};

struct VS_INPUT
{
    float3 position : POSITION;    // Vertex position from vertex buffer
    float2 texCoord : TEXCOORD0;   // Texture UV coordinates
};

struct VS_OUTPUT
{
    float4 position : SV_POSITION; // Screen position (required output)
    float2 texCoord : TEXCOORD0;   // UV coords passed to pixel shader
};

VS_OUTPUT main(VS_INPUT input)
{
    VS_OUTPUT output;

    // Transform vertex position by sprite transform matrix
    // Converts local quad coordinates to world position (NDC)
    output.position = mul(float4(input.position, 1.0), transform);

    // Pass texture coordinates unchanged to pixel shader
    output.texCoord = input.texCoord;

    return output;
}
"#;

// Pixel shader source
// HLSL Pixel Shader - runs once per pixel, determines final color
// Samples texture at UV coordinates and outputs RGBA color
const PIXEL_SHADER: &str = r#"
Texture2D spriteTexture : register(t0);  // Sprite texture bound to slot 0
SamplerState spriteSampler : register(s0); // Sampler state bound to slot 0

struct PS_INPUT
{
    float4 position : SV_POSITION; // Screen position (not used)
    float2 texCoord : TEXCOORD0;   // UV coordinates from vertex shader
};

float4 main(PS_INPUT input) : SV_TARGET
{
    // Sample texture at UV coordinates, return RGBA color
    // Point filtering in sampler gives sharp, pixelated look
    return spriteTexture.Sample(spriteSampler, input.texCoord);
}
"#;
