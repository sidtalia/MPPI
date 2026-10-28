// MPPI rollout of IGHAStar holonomic_6D.cu.
// Controls are body vx, vy, omega (already integrated by the sampler).
// State layout matches the bicycle plant: x y z roll pitch yaw vx vy vz ax ay az wx wy wz.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define x_index 0
#define y_index 1
#define z_index 2
#define roll_index 3
#define pitch_index 4
#define yaw_index 5
#define vx_index 6
#define vy_index 7
#define ax_index 9
#define ay_index 10
#define az_index 11
#define wx_index 12
#define wy_index 13
#define wz_index 14
#define GRAVITY 9.81f

__device__ float holo_nan(float x) {
  return (isnan(x) || isinf(x)) ? 0.0f : x;
}

__device__ float holo_wrap(float x) {
  return fmodf(x + M_PI, 2.0f * M_PI) - M_PI;
}

__device__ float map_to_elev(float x, float y, const float *elev, int map_size_px, float res_inv) {
  int img_X = min(max((int)(x * res_inv + map_size_px / 2), 0), map_size_px - 1);
  int img_Y = min(max((int)(y * res_inv + map_size_px / 2), 0), map_size_px - 1);
  return elev[img_Y * map_size_px + img_X];
}

__device__ void get_footprint_z(float *fl, float *fr, float *bl, float *br, float &z,
                                float x, float y, float cy, float sy, const float *elev,
                                int map_size_px, float res_inv, float car_l2, float car_w2) {
  fl[0] = car_l2 * cy - car_w2 * sy + x;
  fl[1] = car_l2 * sy + car_w2 * cy + y;
  fr[0] = car_l2 * cy + car_w2 * sy + x;
  fr[1] = car_l2 * sy - car_w2 * cy + y;
  bl[0] = -car_l2 * cy - car_w2 * sy + x;
  bl[1] = -car_l2 * sy + car_w2 * cy + y;
  br[0] = -car_l2 * cy + car_w2 * sy + x;
  br[1] = -car_l2 * sy - car_w2 * cy + y;
  float z_cent = map_to_elev(0, 0, elev, map_size_px, res_inv);
  z = map_to_elev(x, y, elev, map_size_px, res_inv) - z_cent;
  fl[2] = map_to_elev(fl[0], fl[1], elev, map_size_px, res_inv) - z_cent;
  fr[2] = map_to_elev(fr[0], fr[1], elev, map_size_px, res_inv) - z_cent;
  bl[2] = map_to_elev(bl[0], bl[1], elev, map_size_px, res_inv) - z_cent;
  br[2] = map_to_elev(br[0], br[1], elev, map_size_px, res_inv) - z_cent;
}

__global__ void holonomic_rollout(float *state, const float *controls, const float *elev,
                                  float dt, int rollouts, int timesteps, int NX, int NC,
                                  int map_size_px, float res_inv, float car_l2, float car_w2) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= rollouts) {
    return;
  }
  int state_index = k * timesteps * NX;
  int control_index = k * timesteps * NC;
  float last_roll = state[state_index + roll_index];
  float last_pitch = state[state_index + pitch_index];

  for (int t = 0; t < timesteps - 1; t++) {
    int curr = state_index + t * NX;
    int next = state_index + (t + 1) * NX;
    int ctrl = control_index + t * NC;

    float x = state[curr + x_index];
    float y = state[curr + y_index];
    float yaw = state[curr + yaw_index];
    float vx = state[curr + vx_index];
    float vy = state[curr + vy_index];
    float omega = state[curr + wz_index];

    float cy = cosf(yaw);
    float sy = sinf(yaw);
    float fl[3], fr[3], bl[3], br[3], z;
    get_footprint_z(fl, fr, bl, br, z, x, y, cy, sy, elev, map_size_px, res_inv, car_l2, car_w2);
    float roll = atan2f((fl[2] + bl[2]) - (fr[2] + br[2]), 4.0f * car_w2);
    float pitch = atan2f((bl[2] + br[2]) - (fl[2] + fr[2]), 4.0f * car_l2);
    float wx = (roll - last_roll) / dt;
    float wy = (pitch - last_pitch) / dt;
    last_roll = roll;
    last_pitch = pitch;

    float cp = cosf(pitch);
    float sp = sinf(pitch);
    float cr = cosf(roll);
    float sr = sinf(roll);
    float ct = holo_nan(sqrtf(1.0f - sp * sp - sr * sr));

    // Sampler already integrated delta v. Reach that command in one step, then
    // add the same terrain-gravity terms as holonomic_6D.cu.
    float vx_cmd = controls[ctrl + 0];
    float vy_cmd = controls[ctrl + 1];
    float om_cmd = controls[ctrl + 2];
    float ax = (vx_cmd - vx) / dt;
    float ay_cmd = (vy_cmd - vy) / dt;
    float alpha = (om_cmd - omega) / dt;
    vx += (ax + sp * GRAVITY) * dt;
    vy += (ay_cmd - sr * GRAVITY) * dt;
    omega += alpha * dt;

    yaw = holo_wrap(yaw + omega * dt);
    cy = cosf(yaw);
    sy = sinf(yaw);
    float vx_world = vx * (cp * cy) + vy * (sr * sp * cy - cr * sy);
    float vy_world = vx * (cp * sy) + vy * (sr * sp * sy + cr * cy);
    x += vx_world * dt;
    y += vy_world * dt;

    float ay = vx * omega - sr * GRAVITY;
    float az = GRAVITY * ct - vx * wy - vy * wx;

    state[next + x_index] = x;
    state[next + y_index] = y;
    state[next + z_index] = z;
    state[next + roll_index] = roll;
    state[next + pitch_index] = pitch;
    state[next + yaw_index] = yaw;
    state[next + vx_index] = vx;
    state[next + vy_index] = vy;
    state[next + ax_index] = ax;
    state[next + ay_index] = ay;
    state[next + az_index] = az;
    state[next + wx_index] = wx;
    state[next + wy_index] = wy;
    state[next + wz_index] = omega;
  }
}

void holonomic_launcher(torch::Tensor state, torch::Tensor controls, torch::Tensor elev,
                        float dt, int rollouts, int timesteps, int NX, int NC,
                        int map_size_px, float map_res, float car_l2, float car_w2) {
  TORCH_CHECK(state.is_cuda() && controls.is_cuda() && elev.is_cuda(), "holonomic rollout needs CUDA");
  TORCH_CHECK(state.is_contiguous() && controls.is_contiguous() && elev.is_contiguous(), "contiguous");
  int threads = 32;
  int blocks = (rollouts + threads - 1) / threads;
  holonomic_rollout<<<blocks, threads>>>(
      state.data_ptr<float>(), controls.data_ptr<float>(), elev.data_ptr<float>(), dt, rollouts,
      timesteps, NX, NC, map_size_px, 1.0f / map_res, car_l2, car_w2);
  cudaDeviceSynchronize();
}
