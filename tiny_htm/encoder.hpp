#pragma once

#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

namespace th
{

//Your standard ScalarEncoder.
struct ScalarEncoder
{
	ScalarEncoder() = default;
	ScalarEncoder(float minval, float maxval, size_t result_sdr_length, size_t num_active_bits)
		: min_val(minval), max_val(maxval), active_bits(num_active_bits), sdr_length(result_sdr_length)
	{
		if(min_val > max_val)
			throw std::runtime_error("ScalarEncoder error: min_val > max_val");
		if(result_sdr_length < num_active_bits)
			throw std::runtime_error("ScalarEncoder error: result_sdr_length < num_active_bits");
	}

	xt::xarray<bool> operator() (float value) const
	{
		return encode(value);
	}

	xt::xarray<bool> encode(float value) const
	{
		float encode_space = (sdr_length - active_bits)/(max_val - min_val);
		int start = encode_space*(value-min_val);
		int end = start + active_bits;
		xt::xarray<bool> res = xt::zeros<bool>({sdr_length});
		xt::view(res, xt::range(start, end))  = true;
		return res;
	}

	void setMiniumValue(float val) {min_val = val;}
	void setMaximumValue(float val) {max_val = val;}
	void setEncodeLengt(size_t val) {active_bits = val;}
	void setSDRLength(size_t val) {sdr_length = val;}

	float miniumValue() const {return min_val;}
	float maximumValue() const {return max_val;}
	size_t encodeLength() const {return active_bits;}
	size_t sdrLength() const {return sdr_length;}


protected:
	float min_val = 0;
	float max_val = 1;
	size_t active_bits = 8;
	size_t sdr_length = 32;
};

//Unlike in NuPIC. The CategoryEncoder in tinyhtm does NOT include space for
//an Unknown category. And the encoding is done by passing a size_t representing
//the category instread of a string.
struct CategoryEncoder
{
	CategoryEncoder(size_t num_cat, size_t encode_len)
		: num_category(num_cat), encode_length(encode_len)
	{}

	xt::xarray<bool> operator() (size_t category) const
	{
		return encode(category);
	}

	xt::xarray<bool> encode(size_t category) const
	{
		if(category > num_category)
			throw std::runtime_error("CategoryEncoder: category > num_category");
		xt::xarray<bool> res = xt::zeros<bool>({num_category, encode_length});
		xt::view(res, category) = true;
		return xt::flatten(res);
	}

	std::vector<size_t> decode(const xt::xarray<bool>& t)
	{
		std::vector<size_t> possible_category;
		for(size_t i=0;i<num_category;i++) {
			if(xt::sum(xt::view(t, xt::range(i*encode_length, (i+1)*encode_length)))[0] > 0)
				possible_category.push_back(i);
		}
		//if(possible_category.size() == 0)
		//	possible_category.push_back(0);
		return possible_category;
	}

	void setNumCategorise(size_t num_cat) {num_category = num_cat;}
	void setEncodeLengt(size_t val) {encode_length = val;}

	size_t numCategories() const {return num_category;}
	size_t encodeLength() const {return encode_length;} 
	size_t sdrLength() const {return num_category*encode_length;}
protected:
	size_t num_category;
	size_t encode_length;
};

float random(float min=0, float max=1)
{
	static std::random_device rd;
	static std::mt19937 eng(rd());
	std::uniform_real_distribution<float> dist;
	float diff = max-min;
	return diff*dist(eng) + min;
}

int roundCoord(float x)
{
	int v = (int)x + ((x-(int)x) > 0.5 ? 1 : -1);
	if(v < 0)
		v += 4;
	return v;
}

template <typename T>
inline decltype(auto) matmul2D(const T& a, const T& b)
{
	return xt::sum(a*b, -1);
}

//TODO: remove global RNG to make object predictable
class GridCellUnit2D
{
public:
	GridCellUnit2D(xt::xtensor<size_t, 1> module_shape={4,4}, float scale_min=6, float scale_max=25)
		: border_len_(module_shape)
	{
		float theta = random(0,2*xt::numeric_constants<float>::PI);
		scale_ = random(scale_min, scale_max);
		bias_ = {random(0, border_len_[0]), random(0, border_len_[1])};
		transform_matrix_ = {{cosf(theta), -sinf(theta)}, {sinf(theta), cosf(theta)}};
	}

	xt::xarray<bool> encode(const xt::xtensor<float, 2>& pos) const
	{
		xt::xarray<bool> res = xt::zeros<bool>({border_len_[0], border_len_[1]});
		
		//Wrap the position
		auto grid_cord = xt::fmod(matmul2D(transform_matrix_, pos)/scale_+bias_, border_len_);
		assert(grid_coors.size() == 2);

		//Set the nearest cell to active
		xt::view(res, (int)grid_cord[1], (int)grid_cord[0]) = 1;

		//Set the 2nd nearest cell to active
		int cx = roundCoord(grid_cord[1])%4;
		int cy = roundCoord(grid_cord[0])%4;
		xt::view(res, cx, cy) = 1;

		res.reshape({res.size()});

		return res;
	}

	size_t encodeSize() const
	{
		return border_len_[0] * border_len_[1];
	}

	xt::xtensor<float, 2> transform_matrix_;
	xt::xtensor<size_t, 1> border_len_;
	xt::xtensor<float, 1> bias_;
	float scale_;
};

class GridCellEncoder2D
{
public:
	GridCellEncoder2D(int num_modules_ = 32, xt::xtensor<size_t, 1> module_shape={4,4}, float scale_min=6, float scale_max=25)
	{
		for(int i=0;i<num_modules_;i++)
			units.push_back(GridCellUnit2D(module_shape, scale_min, scale_max));
	}

	xt::xarray<bool> encode(const xt::xtensor<float, 2>& pos) const
	{
		size_t num_cells = 0;
		for(const auto& u : units)
			num_cells += u.encodeSize();
		xt::xarray<bool> res = xt::zeros<bool>({num_cells});
		size_t start = 0;
		for(const auto& u : units) {
			size_t l = u.encodeSize();
			xt::view(res, xt::range((int)start, l)) = u.encode(pos);
			start += l;
		}
		return res;
	}
	
	std::vector<GridCellUnit2D> units;
};


//Handy encode functions
inline xt::xarray<bool> encodeScalar(float value, float minval, float maxval, size_t result_sdr_length, size_t num_active_bits)
{
	ScalarEncoder e(minval, maxval, result_sdr_length, num_active_bits);
	return e.encode(value);
}

inline xt::xarray<bool> encodeCategory(size_t category, size_t num_cat, size_t encode_len)
{
	CategoryEncoder e(num_cat, encode_len);
	return e.encode(category);
}

}