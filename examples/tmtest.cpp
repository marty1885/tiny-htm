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
	//In this program, TM is going to learn the pattern of 01010101.. So we have 2 possible inputs, 1 and 0,
	//Trough arbitrary decition, each input has 4 bits.
	CategoryEncoder encoder(2, 4);
	TemporalMemory tm({8}, 6); //So the TM has to accept 8 bits of input data. A column size of 6 is arbitrary.

	std::cout << "Learning the sequence 01010101...." << std::endl << std::endl
		<< "Ground truth and algorithm prediction (N = no prediction): " << std::endl;

	//Print the ground truth vales
	for(int i=0;i<40 ;i++)
		std::cout << (i+1)%2;
	std::cout << std::endl;

	//Let the TM learn the patten on the go and make predictions
	for(int i=0;i<40 ;i++) {
		//Learn the pattern and make prediction (true = enable learning)
		auto res = tm.compute(encoder.encode(i%2), true);

		//Decode the results. Print 'N' in gray if no prediction
		auto cat = encoder.decode(res);
		if(cat.size() == 0)
			std::cout << "\033[90mN\033[0m";
		else
			std::cout << cat[0];
	}
	std::cout << std::endl;
}
