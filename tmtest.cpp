#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xcsv.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xdynamic_view.hpp>
#include <xtensor/xrandom.hpp>

#include "HTMField.hpp"

#include <string>

int main()
{
	TemporalMemory tm({8}, 6);
	CategoryEncoder encoder(2, 4);
	for(int i=0;i<120;i++)
		std::cout << xt::cast<int>(tm.compute(encoder.encode(i%2), true)) << std::endl;
}