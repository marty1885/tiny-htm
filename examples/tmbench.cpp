#include <tiny_htm/tiny_htm.hpp>

#include <vector>
#include <chrono>

#include <omp.h>

float benchmarkTemporalMemory(const std::vector<size_t>& out_shape, const std::vector<xt::xarray<bool>>& x, size_t num_epoch)
{
	float time_used = 0;
	for(size_t i=0;i<num_epoch;i++) {

		TemporalMemory tm(as<std::vector<size_t>>(x[0].shape()), 16);

		auto t0 = std::chrono::high_resolution_clock::now();
		size_t j = 0;
		for(const auto& d : x) {
			tm.compute(d, true);

			//Reorganize the synapses every N times for better perofmance
			//This makes the memory access pattern more linear
			j += 1;
			if(j == 20) {
				//tm.cells_.decaySynapse(0.05);
				tm.organizeSynapse();
				j = 0;
			}
		}

		auto t1 = std::chrono::high_resolution_clock::now();

		time_used += std::chrono::duration_cast<std::chrono::duration<float>>(t1-t0).count();
	}

	return time_used/num_epoch;
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
	std::cout << "Benchmarking TemporalMemory algorithm: \n\n";

	std::vector<xt::xarray<bool>> input_data;
	std::vector<int> num_threads = {1, 2, 4, 8};
	std::vector<size_t> input_size = {64, 128, 256, 512, 1024};
	size_t num_data = 100;

	for(auto input_len : input_size) {
		auto input_data = generateRandomData(input_len, num_data);

		for(int threads : num_threads) {
			omp_set_num_threads(threads);
			float t = benchmarkTemporalMemory({input_len}, input_data, 10);
			std::cout << input_len << " bits per SDR, " << threads << " threads: " << t/num_data*1000 << "ms per forward" << std::endl;
		}
	}
}