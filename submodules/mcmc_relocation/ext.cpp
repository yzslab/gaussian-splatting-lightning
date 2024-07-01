// copied from https://github.com/shakibakh/diff-gaussian-rasterization/tree/1761eb93b8b9bfebec9fbf06af4654583c2b1ddb/cuda_rasterizer

#include <torch/extension.h>
#include <stdio.h>

void ComputeRelocation(
        int P,
        float* opacity_old,
        float* scale_old,
        int* N,
        float* binoms,
        int n_max,
        float* opacity_new,
        float* scale_new);

std::tuple<torch::Tensor, torch::Tensor> ComputeRelocationCUDA(
        torch::Tensor& opacity_old,
        torch::Tensor& scale_old,
        torch::Tensor& N,
        torch::Tensor& binoms,
        const int n_max)
{
    const int P = opacity_old.size(0);

    torch::Tensor final_opacity = torch::full({P}, 0, opacity_old.options().dtype(torch::kFloat32));
    torch::Tensor final_scale = torch::full({3 * P}, 0, scale_old.options().dtype(torch::kFloat32));

    if(P != 0)
    {
        ComputeRelocation(P,
                                 opacity_old.contiguous().data<float>(),
                                 scale_old.contiguous().data<float>(),
                                 N.contiguous().data<int>(),
                                 binoms.contiguous().data<float>(),
                                 n_max,
                                 final_opacity.contiguous().data<float>(),
                                 final_scale.contiguous().data<float>());
    }

    return std::make_tuple(final_opacity, final_scale);

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("compute_relocation", &ComputeRelocationCUDA);
}