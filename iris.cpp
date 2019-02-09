#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xcsv.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xdynamic_view.hpp>
#include <xtensor/xrandom.hpp>

#include "HTMField.hpp"

#include <string>
#include <fstream>
#include <map>

xt::xarray<float> normalizeIris(const xt::xarray<float>& dataset)
{
	auto res = dataset;
	auto v = xt::view(res, xt::all(), xt::range(0, -1));
	for(size_t i=0;i<v.shape()[1];i++) {
		auto r = xt::view(v, xt::all(), i);
		float min = xt::amin(r)[0];
		float max = xt::amax(r)[0];
		r = (r-min)/(max-min);
	}
	return res;
}

template <typename T>
xt::xarray<bool> makeSDR(const T& v)
{
	return xt::concatenate(xt::xtuple(
		encodeScalar(v[0], 0,1,8,16),
		encodeScalar(v[1], 0,1,8,16),
		encodeScalar(v[2], 0,1,8,16),
		encodeScalar(v[3], 0,1,8,16)
	));
}

int main()
{
	std::ifstream in("Iris.csv");
	if(in.good() == false) {
		std::cout << "Iris.csv file not found. Please download the iris dataset and put it in your working directory" << std::endl;
		return 0;
	}

	xt::xarray<std::string> csv_content = xt::load_csv<std::string>(in);
	in.close();

	//std::cout << csv_content << std::endl;

	//Some crazy C++ data processing
	xt::xarray<float> dataset = xt::zeros<float>({csv_content.shape()[0]-1, csv_content.shape()[1]-1});
	std::map<std::string, int> category_map;
	category_map["Iris-setosa"] = 0;
	category_map["Iris-versicolor"] = 1;
	category_map["Iris-virginica"] = 2;

	for(size_t i=1;i<csv_content.shape()[0];i++) { // Ignore the first col. It contains the description
		for(size_t j=1;j<csv_content.shape()[1];j++) {
			if(j != csv_content.shape()[1]-1)
				xt::view(dataset, i-1, j-1) = std::stof(xt::view(csv_content, i,j)[0]);
			else
				xt::view(dataset, i-1, j-1) = category_map[xt::view(csv_content, i,j)[0]];
		}
	}

	dataset = normalizeIris(dataset);
	xt::random::shuffle(dataset);

	//Split traning and testing set
	auto traning_set = xt::view(dataset, xt::range(0, dataset.shape()[0]/2));
	auto testing_set = xt::view(dataset, xt::range(dataset.shape()[0]/2, -1));

	SpatialPooler1D sp(64, 32);

	//Train SpatialPooler to reconize the patterns
	for(size_t i=0;i<traning_set.shape()[0];i++) {
		auto v = xt::view(traning_set, i);
		auto sdr = makeSDR(v);
		sp.compute(sdr, true);
	}

	//Construct a classifer to classify the results
	SDRClassifer classifer(3, {32});
	for(size_t i=0;i<traning_set.shape()[0];i++) {
		auto v = xt::view(traning_set, i);
		auto sdr = makeSDR(v);
		auto out = sp.compute(sdr, false);

		classifer.add((size_t)v[4], out);
	}

	//Test HTM and classifer with the testing set
	xt::xarray<int> confustion_matrix = xt::zeros<int>({3,3});
	for(size_t i=0;i<testing_set.shape()[0];i++) {
		auto v = xt::view(testing_set, i);
		auto sdr = makeSDR(v);
		auto out = sp.compute(sdr, false);
		size_t predict = classifer.compute(out, 0.5);

		size_t real_category = (size_t)v[4];
		xt::view(confustion_matrix, real_category, predict) += 1;
	}

	//Print the results
	size_t correct = 0;
	size_t total = xt::sum(confustion_matrix)[0];
	for(size_t i=0;i<3;i++)
		correct += xt::view(confustion_matrix, i,i)[0];
	std::cout << "accuracy: " << 100.f*correct/total << ", " << correct << "/" << total << '\n';
	std::cout << "*\t0\t1\t2\n";
	for(size_t i=0;i<confustion_matrix.shape()[0];i++) {
		std::cout << i << '\t';
		for(size_t j=0;j<confustion_matrix.shape()[1];j++) {
			std::cout << xt::view(confustion_matrix, i, j)[0]
				<< (j==confustion_matrix.shape()[1]-1?'\n':'\t');
		}
	}
}
