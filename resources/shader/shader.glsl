#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, rgba8)

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < buf.length()) {
        buf.data[idx] = 255;
    }
}