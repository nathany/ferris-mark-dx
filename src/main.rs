use std::ptr;
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
    window_width: f32,
    window_height: f32,
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
            window_width: 1920.0,
            window_height: 1080.0,
        };

        // Create quad resources
        unsafe {
            context.create_quad_resources()?;
        }

        Ok(context)
    }

    unsafe fn create_quad_resources(&mut self) -> Result<()> {
        // Define quad vertices (exactly 128x128 pixels, centered on screen)
        // Calculate exact NDC coordinates based on current window size
        let half_width_ndc = 128.0 / self.window_width;
        let half_height_ndc = 128.0 / self.window_height;
        let vertices = [
            Vertex {
                position: [-half_width_ndc, half_height_ndc, 0.0], // Top left
                tex_coord: [0.0, 0.0],                             // UV coordinates
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
        let vs_source = r#"
            struct VS_INPUT {
                float3 pos : POSITION;
                float2 tex : TEXCOORD;
            };

            struct VS_OUTPUT {
                float4 pos : SV_POSITION;
                float2 tex : TEXCOORD;
            };

            VS_OUTPUT main(VS_INPUT input) {
                VS_OUTPUT output;
                // Pass through NDC coordinates directly (already calculated)
                output.pos = float4(input.pos, 1.0);
                output.tex = input.tex;
                return output;
            }
        "#;

        // Pixel shader source
        let ps_source = r#"
            Texture2D tex : register(t0);
            SamplerState samplerState : register(s0);

            struct PS_INPUT {
                float4 pos : SV_POSITION;
                float2 tex : TEXCOORD;
            };

            float4 main(PS_INPUT input) : SV_TARGET {
                return tex.Sample(samplerState, input.tex);
            }
        "#;

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
        let sampler_desc = D3D11_SAMPLER_DESC {
            Filter: D3D11_FILTER_MIN_MAG_MIP_POINT, // Nearest neighbor filtering
            AddressU: D3D11_TEXTURE_ADDRESS_CLAMP,
            AddressV: D3D11_TEXTURE_ADDRESS_CLAMP,
            AddressW: D3D11_TEXTURE_ADDRESS_CLAMP,
            MipLODBias: 0.0,
            MaxAnisotropy: 1,
            ComparisonFunc: D3D11_COMPARISON_ALWAYS,
            BorderColor: [0.0, 0.0, 0.0, 0.0],
            MinLOD: 0.0,
            MaxLOD: f32::MAX,
        };

        unsafe {
            self.device
                .CreateSamplerState(&sampler_desc, Some(&mut self.sampler_state))?;
        }

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

                // Draw the quad
                self.device_context.DrawIndexed(6, 0, 0);

                // Present the frame
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

        // Message loop
        let mut message = MSG::default();
        while GetMessageW(&mut message, None, 0, 0).into() {
            let _ = TranslateMessage(&message);
            DispatchMessageW(&message);
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
