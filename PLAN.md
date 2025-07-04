We have a 2d sprite rendering now. The next steps are:

1. To add movement
2. Add some benchmarking

Let's focus on the movement with a single sprite first. What I'm trying to achieve is based on this Go code, but it's for OpenGL. Whereas we're using Rust and DirectX 11.
https://github.com/unitoftime/experiments/blob/master/gophermark/regular/main.go

DirectX Tool Kit (aka DirectXTK)

3. SpriteBatch
https://raw.githubusercontent.com/microsoft/DirectXTK/refs/heads/main/Src/SpriteBatch.cpp

Later:
* big refactors
* safety audit
* PIX for debugging
* finding a solution for debug layer logging
