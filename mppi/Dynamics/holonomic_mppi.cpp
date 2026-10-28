#include <torch/extension.h>

void holonomic_launcher(torch::Tensor state, torch::Tensor controls, torch::Tensor elev,
                        float dt, int rollouts, int timesteps, int NX, int NC,
                        int map_size_px, float map_res, float car_l2, float car_w2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rollout", &holonomic_launcher, "holonomic 6D MPPI rollout");
}
