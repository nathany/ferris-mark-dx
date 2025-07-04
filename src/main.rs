use windows::{
    Win32::Foundation::*, Win32::Graphics::Gdi::*, Win32::System::LibraryLoader::GetModuleHandleW,
    Win32::UI::Input::KeyboardAndMouse::*, Win32::UI::WindowsAndMessaging::*, core::*,
};

fn main() -> Result<()> {
    unsafe {
        // Get the module handle
        let instance = GetModuleHandleW(None)?;

        // Define the window class name
        let window_class = w!("FerrisMarkWindow");

        // Create the window class
        let wc = WNDCLASSW {
            lpfnWndProc: Some(wndproc),
            hInstance: instance.into(),
            lpszClassName: window_class,
            hbrBackground: CreateSolidBrush(COLORREF(0x00F0F0F0)), // Light gray background
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
            w!("Ferris Mark"),
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

        // Show the window
        let _ = ShowWindow(hwnd, SW_SHOW);
        let _ = UpdateWindow(hwnd);

        // Message loop
        let mut message = MSG::default();
        while GetMessageW(&mut message, None, 0, 0).into() {
            let _ = TranslateMessage(&message);
            DispatchMessageW(&message);
        }

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
                let mut ps = PAINTSTRUCT::default();
                let hdc = BeginPaint(window, &mut ps);

                // Fill the background
                let _ = FillRect(hdc, &ps.rcPaint, CreateSolidBrush(COLORREF(0x00F0F0F0)));

                // Draw some text
                let text = w!("Welcome to Ferris Mark!");
                let mut rect = RECT::default();
                let _ = GetClientRect(window, &mut rect);

                let _ = SetBkMode(hdc, TRANSPARENT);
                let _ = SetTextColor(hdc, COLORREF(0x00000000)); // Black text

                // Create a mutable copy of the text for DrawTextW
                let mut text_buffer: Vec<u16> = text.as_wide().to_vec();

                let _ = DrawTextW(
                    hdc,
                    &mut text_buffer,
                    &mut rect,
                    DT_CENTER | DT_VCENTER | DT_SINGLELINE,
                );

                let _ = EndPaint(window, &ps);
                LRESULT(0)
            }
            _ => DefWindowProcW(window, message, wparam, lparam),
        }
    }
}
