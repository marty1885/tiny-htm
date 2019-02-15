#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xcsv.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xdynamic_view.hpp>
#include <xtensor/xrandom.hpp>

#include <tiny_htm/tiny_htm.hpp>

#include <string>

int main()
{
	TemporalMemory tm({8}, 6);
	CategoryEncoder encoder(2, 4);
	std::cout << "Learning the sequence 01010101...." << std::endl << std::endl
		<< "Algorithm prediction (N = no prediction): " << std::endl;
	for(int i=0;i<40 ;i++) {
		auto res = tm.compute(encoder.encode(i%2), true);
		auto cat = encoder.decode(res);
		if(cat.size() == 0)
			std::cout << "\033[90mN\033[0m";
		else
			std::cout << cat[0];
	}
	std::cout << std::endl;
}
