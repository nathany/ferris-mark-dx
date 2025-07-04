use std::ptr;
use windows::{
    Win32::Foundation::*, Win32::Graphics::Direct3D::*, Win32::Graphics::Direct3D11::*,
    Win32::Graphics::Dxgi::Common::*, Win32::Graphics::Dxgi::*, Win32::Graphics::Gdi::*,
    Win32::System::LibraryLoader::GetModuleHandleW, Win32::UI::Input::KeyboardAndMouse::*,
    Win32::UI::WindowsAndMessaging::*, core::*,
};

struct D3D11Context {
    device: ID3D11Device,
    device_context: ID3D11DeviceContext,
    swap_chain: IDXGISwapChain,
    render_target_view: Option<ID3D11RenderTargetView>,
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

        // Log debug layer status
        #[cfg(any(debug_assertions, feature = "d3d11-debug"))]
        println!("DirectX 11 Debug Layer: ENABLED");
        #[cfg(not(any(debug_assertions, feature = "d3d11-debug")))]
        println!("DirectX 11 Debug Layer: DISABLED");

        Ok(D3D11Context {
            device,
            device_context,
            swap_chain,
            render_target_view,
        })
    }

    unsafe fn render(&self) {
        if let Some(rtv) = &self.render_target_view {
            // Clear the render target to a solid color (cornflower blue)
            let clear_color = [0.39, 0.58, 0.93, 1.0]; // RGBA
            unsafe {
                self.device_context.ClearRenderTargetView(rtv, &clear_color);

                // Present the frame
                let _ = self.swap_chain.Present(1, 0);
            }
        }
    }

    unsafe fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        if width == 0 || height == 0 {
            return Ok(());
        }

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
