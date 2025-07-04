use std::env;
use std::ptr;
use std::time::Instant;
extern crate rand;
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::{
    Win32::Foundation::*, Win32::Graphics::Direct3D::*, Win32::Graphics::Direct3D11::*,
    Win32::Graphics::Dxgi::Common::*, Win32::Graphics::Dxgi::*, Win32::Graphics::Gdi::*,
    Win32::System::LibraryLoader::GetModuleHandleW, Win32::UI::Input::KeyboardAndMouse::*,
    Win32::UI::WindowsAndMessaging::*, core::*,
};

// Vertex structure for our textured quad
#[repr(C)]
#[derive(Clone, Copy)]
struct Vertex {
    position: [f32; 3],
    tex_coord: [f32; 2],
}

// Sprite structure for movement system
#[derive(Clone, Copy)]
struct Sprite {
    position: [f32; 2],
    velocity: [f32; 2],
}

impl Sprite {
    fn new() -> Self {
        use std::f32::consts::PI;

        // Random angle and speed
        let angle = rand::random::<f32>() * 2.0 * PI;
        let speed = 200.0 + rand::random::<f32>() * 100.0; // 200-300 pixels/second

        Self {
            position: [100.0, 100.0], // Start at top-left area
            velocity: [angle.cos() * speed, angle.sin() * speed],
        }
    }

    fn update(&mut self, dt: f32, window_width: f32, window_height: f32) {
        // Update position
        self.position[0] += self.velocity[0] * dt;
        self.position[1] += self.velocity[1] * dt;

        // Sprite dimensions (128x128)
        let sprite_width = 128.0;
        let sprite_height = 128.0;

        // Bounce off edges
        if self.position[0] <= 0.0 || self.position[0] + sprite_width >= window_width {
            self.velocity[0] = -self.velocity[0];
            println!(
                "Bounce X: pos=({:.1}, {:.1}), vel=({:.1}, {:.1})",
                self.position[0], self.position[1], self.velocity[0], self.velocity[1]
            );
        }
        if self.position[1] <= 0.0 || self.position[1] + sprite_height >= window_height {
            self.velocity[1] = -self.velocity[1];
            println!(
                "Bounce Y: pos=({:.1}, {:.1}), vel=({:.1}, {:.1})",
                self.position[0], self.position[1], self.velocity[0], self.velocity[1]
            );
        }

        // Clamp position to screen bounds
        self.position[0] = self.position[0].clamp(0.0, window_width - sprite_width);
        self.position[1] = self.position[1].clamp(0.0, window_height - sprite_height);
    }

    fn get_transform_matrix(&self, window_width: f32, window_height: f32) -> [f32; 16] {
        // Convert pixel coordinates to NDC
        let ndc_x = (self.position[0] / window_width) * 2.0 - 1.0;
        let ndc_y = -((self.position[1] / window_height) * 2.0 - 1.0); // Flip Y for DirectX

        // Create simple translation matrix (row-major for HLSL mul(vector, matrix))
        [
            1.0, 0.0, 0.0, ndc_x, 0.0, 1.0, 0.0, ndc_y, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]
    }
}

// Transform matrix constant buffer
#[repr(C)]
#[derive(Clone, Copy)]
struct TransformBuffer {
    transform: [f32; 16],
}

struct D3D11Context {
    device: ID3D11Device,
    device_context: ID3D11DeviceContext,
    swap_chain: IDXGISwapChain,
    render_target_view: Option<ID3D11RenderTargetView>,
    vertex_buffer: Option<ID3D11Buffer>,
    index_buffer: Option<ID3D11Buffer>,
    vertex_shader: Option<ID3D11VertexShader>,
    pixel_shader: Option<ID3D11PixelShader>,
    input_layout: Option<ID3D11InputLayout>,
    texture: Option<ID3D11Texture2D>,
    texture_view: Option<ID3D11ShaderResourceView>,
    sampler_state: Option<ID3D11SamplerState>,
    constant_buffer: Option<ID3D11Buffer>,
    window_width: f32,
    window_height: f32,
    sprites: Vec<Sprite>,
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
            window_width: 1920.0,
            window_height: 1080.0,
            sprites: Vec::new(),
            last_time: now,
            frame_count: 0,
            last_log_time: now,
        };

        // Create quad resources
        unsafe {
            context.create_quad_resources()?;
        }

        // Initialize sprites with count from command line or default
        let sprite_count = get_sprite_count();
        context.init_sprites(sprite_count);
        println!("Initialized {} sprites", sprite_count);

        Ok(context)
    }

    fn init_sprites(&mut self, count: usize) {
        self.sprites.clear();
        for _ in 0..count {
            self.sprites.push(Sprite::new());
        }
    }

    unsafe fn create_quad_resources(&mut self) -> Result<()> {
        // Define quad vertices (128x128 pixels, will be positioned by transform matrix)
        let half_width_ndc = 128.0 / self.window_width;
        let half_height_ndc = 128.0 / self.window_height;
        let vertices = [
            Vertex {
                position: [-half_width_ndc, half_height_ndc, 0.0], // Top left
                tex_coord: [0.0, 0.0],
            },
            Vertex {
                position: [half_width_ndc, half_height_ndc, 0.0], // Top right
                tex_coord: [1.0, 0.0],
            },
            Vertex {
                position: [half_width_ndc, -half_height_ndc, 0.0], // Bottom right
                tex_coord: [1.0, 1.0],
            },
            Vertex {
                position: [-half_width_ndc, -half_height_ndc, 0.0], // Bottom left
                tex_coord: [0.0, 1.0],
            },
        ];

        // Define indices for two triangles making a quad
        let indices: [u16; 6] = [
            0, 1, 2, // First triangle
            0, 2, 3, // Second triangle
        ];

        // Create vertex buffer
        let vertex_buffer_desc = D3D11_BUFFER_DESC {
            ByteWidth: (std::mem::size_of::<Vertex>() * vertices.len()) as u32,
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_VERTEX_BUFFER.0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
            StructureByteStride: 0,
        };

        let vertex_data = D3D11_SUBRESOURCE_DATA {
            pSysMem: vertices.as_ptr() as *const _,
            SysMemPitch: 0,
            SysMemSlicePitch: 0,
        };

        unsafe {
            self.device.CreateBuffer(
                &vertex_buffer_desc,
                Some(&vertex_data),
                Some(&mut self.vertex_buffer),
            )?;
        }

        // Create index buffer
        let index_buffer_desc = D3D11_BUFFER_DESC {
            ByteWidth: (std::mem::size_of::<u16>() * indices.len()) as u32,
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_INDEX_BUFFER.0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
            StructureByteStride: 0,
        };

        let index_data = D3D11_SUBRESOURCE_DATA {
            pSysMem: indices.as_ptr() as *const _,
            SysMemPitch: 0,
            SysMemSlicePitch: 0,
        };

        unsafe {
            self.device.CreateBuffer(
                &index_buffer_desc,
                Some(&index_data),
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

        // Load texture
        unsafe {
            self.load_texture()?;
        }

        // Create sampler state with point filtering for pixel-perfect rendering
        // Create and set up sampler state
        let sampler_desc = D3D11_SAMPLER_DESC {
            Filter: D3D11_FILTER_MIN_MAG_MIP_POINT, // Point filtering for pixel art
            AddressU: D3D11_TEXTURE_ADDRESS_CLAMP,
            AddressV: D3D11_TEXTURE_ADDRESS_CLAMP,
            AddressW: D3D11_TEXTURE_ADDRESS_CLAMP,
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
        let cb_desc = D3D11_BUFFER_DESC {
            ByteWidth: std::mem::size_of::<TransformBuffer>() as u32,
            Usage: D3D11_USAGE_DYNAMIC,
            BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
            CPUAccessFlags: D3D11_CPU_ACCESS_WRITE.0 as u32,
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
        // Load the PNG image
        let img = image::open("ferris_pixel_128x128.png")
            .map_err(|_e| Error::from_hresult(windows::core::HRESULT(-1)))?;

        let img = img.to_rgba8();
        let (width, height) = img.dimensions();
        let pixels = img.as_raw();

        // Create texture description
        let texture_desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };

        let texture_data = D3D11_SUBRESOURCE_DATA {
            pSysMem: pixels.as_ptr() as *const _,
            SysMemPitch: width * 4, // 4 bytes per pixel (RGBA)
            SysMemSlicePitch: 0,
        };

        // Create texture
        unsafe {
            self.device.CreateTexture2D(
                &texture_desc,
                Some(&texture_data),
                Some(&mut self.texture),
            )?;
        }

        // Create shader resource view
        if let Some(texture) = &self.texture {
            unsafe {
                self.device.CreateShaderResourceView(
                    texture,
                    None,
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
        let mut blob = None;
        let mut error_blob = None;

        let source_cstr = std::ffi::CString::new(source).unwrap();
        let entry_cstr = std::ffi::CString::new(entry_point).unwrap();
        let target_cstr = std::ffi::CString::new(target).unwrap();

        let result = unsafe {
            D3DCompile(
                source_cstr.as_ptr() as *const _,
                source.len(),
                None,
                None,
                None,
                PCSTR(entry_cstr.as_ptr() as *const u8),
                PCSTR(target_cstr.as_ptr() as *const u8),
                0,
                0,
                &mut blob,
                Some(&mut error_blob),
            )
        };

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

        // Log FPS occasionally
        self.frame_count += 1;
        if current_time.duration_since(self.last_log_time).as_secs() >= 1 {
            let fps = self.frame_count as f32
                / current_time
                    .duration_since(self.last_log_time)
                    .as_secs_f32();
            println!(
                "FPS: {:.1} | Sprites: {} | Frame time: {:.2}ms",
                fps,
                self.sprites.len(),
                dt * 1000.0
            );
            self.frame_count = 0;
            self.last_log_time = current_time;
        }
    }

    unsafe fn render(&self) {
        if let Some(rtv) = &self.render_target_view {
            // Clear the render target to a solid color (cornflower blue)
            let clear_color = [0.39, 0.58, 0.93, 1.0]; // RGBA
            unsafe {
                self.device_context.ClearRenderTargetView(rtv, &clear_color);

                // Set up the rendering pipeline
                self.device_context
                    .OMSetRenderTargets(Some(&[Some(rtv.clone())]), None);

                // Set vertex buffer
                if let Some(vertex_buffer) = &self.vertex_buffer {
                    let stride = std::mem::size_of::<Vertex>() as u32;
                    let offset = 0;
                    self.device_context.IASetVertexBuffers(
                        0,
                        1,
                        Some(&Some(vertex_buffer.clone())),
                        Some(&stride),
                        Some(&offset),
                    );
                }

                // Set index buffer
                if let Some(index_buffer) = &self.index_buffer {
                    self.device_context
                        .IASetIndexBuffer(index_buffer, DXGI_FORMAT_R16_UINT, 0);
                }

                // Set input layout
                if let Some(input_layout) = &self.input_layout {
                    self.device_context.IASetInputLayout(input_layout);
                }

                // Set primitive topology
                self.device_context
                    .IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

                // Set shaders
                if let Some(vertex_shader) = &self.vertex_shader {
                    self.device_context.VSSetShader(vertex_shader, None);
                }
                if let Some(pixel_shader) = &self.pixel_shader {
                    self.device_context.PSSetShader(pixel_shader, None);
                }

                // Set texture and sampler
                if let Some(texture_view) = &self.texture_view {
                    self.device_context
                        .PSSetShaderResources(0, Some(&[Some(texture_view.clone())]));
                }
                if let Some(sampler_state) = &self.sampler_state {
                    self.device_context
                        .PSSetSamplers(0, Some(&[Some(sampler_state.clone())]));
                }

                // Set viewport for pixel-perfect rendering using current window size
                let viewport = D3D11_VIEWPORT {
                    TopLeftX: 0.0,
                    TopLeftY: 0.0,
                    Width: self.window_width,
                    Height: self.window_height,
                    MinDepth: 0.0,
                    MaxDepth: 1.0,
                };
                self.device_context.RSSetViewports(Some(&[viewport]));

                // Draw all sprites
                for sprite in &self.sprites {
                    // Update transform matrix
                    let transform_matrix =
                        sprite.get_transform_matrix(self.window_width, self.window_height);
                    let transform_buffer = TransformBuffer {
                        transform: transform_matrix,
                    };

                    // Map and update constant buffer
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

                        // Set constant buffer
                        self.device_context
                            .VSSetConstantBuffers(0, Some(&[Some(constant_buffer.clone())]));
                    }

                    // Draw the sprite
                    self.device_context.DrawIndexed(6, 0, 0);
                }

                // Present the frame
                let _ = self.swap_chain.Present(0, 0); // 0 for unlimited FPS
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
                if count > 0 && count <= 10000 {
                    count
                } else {
                    println!("Warning: Sprite count must be between 1 and 10000. Using default: 1");
                    1
                }
            }
            Err(_) => {
                println!(
                    "Warning: Invalid sprite count '{}'. Using default: 1",
                    args[1]
                );
                1
            }
        }
    } else {
        1 // Default sprite count (reduced for debugging)
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

// Vertex shader source with transform matrix
const VERTEX_SHADER: &str = r#"
cbuffer TransformBuffer : register(b0)
{
    matrix transform;
};

struct VS_INPUT
{
    float3 position : POSITION;
    float2 texCoord : TEXCOORD0;
};

struct VS_OUTPUT
{
    float4 position : SV_POSITION;
    float2 texCoord : TEXCOORD0;
};

VS_OUTPUT main(VS_INPUT input)
{
    VS_OUTPUT output;
    output.position = mul(float4(input.position, 1.0), transform);
    output.texCoord = input.texCoord;
    return output;
}
"#;

// Pixel shader source
const PIXEL_SHADER: &str = r#"
Texture2D mainTexture : register(t0);
SamplerState mainSampler : register(s0);

struct PS_INPUT
{
    float4 position : SV_POSITION;
    float2 texCoord : TEXCOORD0;
};

float4 main(PS_INPUT input) : SV_TARGET
{
    return mainTexture.Sample(mainSampler, input.texCoord);
}
"#;
