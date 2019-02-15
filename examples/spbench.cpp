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

	return std::chrono::duration_cast<std::chrono::duration<float>>(t1-t0).count()/num_epoch;
}

std::vector<xt::xarray<bool>> generateRandomData(size_t input_length, size_t num_data)
{
	std::vector<xt::xarray<bool>> res(num_data);
	static std::mt19937 rng;
	std::uniform_real_distribution<float> dist(0, 1);

	#pragma omp parallel for
	for(size_t i=0;i<num_data;i++)
		res[i] = encodeScalar(dist(rng), 0, 1, input_length, input_length*0.15);
	return res;
}

int main()
{
	std::cout << "Benchmarking SpatialPooler algorithm: \n\n";

	std::vector<xt::xarray<bool>> input_data;
	std::vector<int> num_threads = {1, 2, 4, 8};
	std::vector<size_t> input_size = {64, 128, 256, 512, 1024};
	size_t num_data = 1000;

	for(auto input_len : input_size) {
		auto input_data = generateRandomData(input_len, num_data);

		for(int threads : num_threads) {
			omp_set_num_threads(threads);
			float t = benchmarkSpatialPooler({input_len}, input_data, 100);
			std::cout << input_len << " bits per SDR, " << threads << " threads: " << t/num_data*1000 << "ms per forward" << std::endl;
		}
	}
}