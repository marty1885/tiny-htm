#include <tiny_htm/tiny_htm.hpp>

#include <vector>
#include <chrono>

#include <omp.h>

float benchmarkSpatialPooler(const std::vector<size_t>& out_shape, const std::vector<xt::xarray<bool>>& x, size_t num_epoch)
{
	SpatialPooler sp(as<std::vector<size_t>>(x[0].shape()), out_shape);

	auto t0 = std::chrono::high_resolution_clock::now();
	for(size_t i=0;i<num_epoch;i++) {
		for(const auto& d : x) {
			sp.compute(d, true);
		}
	}
	auto t1 = std::chrono::high_resolution_clock::now();

	return std::chrono::duration_cast<std::chrono::duration<float>>(t1-t0).count();
}

int main()
{
	std::vector<xt::xarray<bool>> input_data;
	std::mt19937 rng;
	std::uniform_real_distribution<float> dist(0, 1);
	for(int i=0;i<1000;i++)
		input_data.push_back(encodeScalar(dist(rng), 0, 1, 256, 24));
	
	std::cout << benchmarkSpatialPooler({64}, input_data, 100) << std::endl;
}