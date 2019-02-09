#pragma once

#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xdynamic_view.hpp>

#include <vector>
#include <algorithm>
#include <random>
#include <queue>

#include <assert.h>

xt::xdynamic_slice_vector sliceVec(const std::vector<size_t>& v)
{
	xt::xdynamic_slice_vector sv;
	for(auto i : v)
		sv.push_back(i);
	return sv;
}

//Your standard ScalarEncoder.
struct ScalarEncoder
{
	ScalarEncoder() = default;
	ScalarEncoder(float minval, float maxval, size_t encode_len, size_t width)
		: min_val(minval), max_val(maxval), encode_length(encode_len), sdr_length(width)
	{
		if(min_val > max_val)
			throw std::runtime_error("ScalarEncoder error: min_val > max_val");
	}

	xt::xarray<bool> operator() (float value) const
	{
		return encode(value);
	}

	xt::xarray<bool> encode(float value) const
	{
		float encode_space = sdr_length - encode_length;
		int start = encode_space*value;
		int end = start + encode_length;
		xt::xarray<bool> res = xt::zeros<bool>({sdr_length});
		xt::view(res, xt::range(start, end))  = true;
		return res;
	}

	void setMiniumValue(float val) {min_val = val;}
	void setMaximumValue(float val) {max_val = val;}
	void setEncodeLengt(size_t val) {encode_length = val;}
	void setSDRLength(size_t val) {sdr_length = val;}

	float miniumValue() const {return min_val;}
	float maximumValue() const {return max_val;}
	size_t encodeLength() const {return encode_length;}
	size_t sdrLength() const {return sdr_length;}


protected:
	float min_val = 0;
	float max_val = 1;
	size_t encode_length = 8;
	size_t sdr_length = 32;
};

//Unlike in NuPIC. The CategoryEncoder in HTMHelper does NOT include space for
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


//Handy encode functions
inline xt::xarray<bool> encodeScalar(float value, float minval, float maxval, size_t encode_len, size_t width)
{
	ScalarEncoder e(minval, maxval, encode_len, width);
	return e.encode(value);
}

inline xt::xarray<bool> encodeCategory(size_t category, size_t num_cat, size_t encode_len)
{
	CategoryEncoder e(num_cat, encode_len);
	return e.encode(category);
}

inline std::vector<size_t> foldIndex(size_t index, const std::vector<size_t>& shape)
{
	std::vector<size_t> v(shape.size());
	size_t acc = 1;
	for(size_t i=shape.size()-1;i>=0;i--) {
		acc *= v[i];
		v[i] = acc;
	}
	std::vector<size_t> res(shape.size());
	for(size_t i=0;i<v.size();i++) {
		res[i] = index/v[i];
		index = index%v[i];
	}
	return res;
}

template <typename T>
inline size_t unfoldIndex(const std::vector<size_t>& index, const T& shape)
{
	size_t s = 0;
	size_t v = 1;
	assert(index.size() == shape.size());
	for(size_t i=0;i<index.size();i++) {
		size_t n = v * index[i];
		s += n;
		v *= shape[i];
	}

	return s;
}

//Flexable framework for HTM implementations

struct CellSheet
{
	CellSheet(size_t num_cells, size_t max_connection_per_cell)
	{
		connection_target = xt::xarray<std::vector<size_t>>::from_shape({num_cells, max_connection_per_cell});
		connection_permence = xt::zeros<float>({num_cells, max_connection_per_cell});
	}


	xt::xarray<uint32_t> calculateOverlapScore(const xt::xarray<bool>& x)
	{
		xt::xarray<uint32_t> res = xt::zeros<uint32_t>({connection_permence.shape()[0]});
		for(size_t i=0;i<connection_permence.shape()[0];i++) {
			uint32_t sum = 0;
			for(size_t j=0;j<connection_permence.shape()[1];j++) {
				if(xt::dynamic_view(x, sliceVec(xt::view(connection_target, i, j)[0]))[0] == true
					&& xt::view(connection_permence, i, j)[0] > 0.21)
					sum += 1;
			}
			xt::view(res, i) = sum;
		}
		return res;
	}

	void learnCorrelation(const xt::xarray<bool>& x, const xt::xarray<bool>& y, float perm_incerment, float perm_decarment)
	{
		assert(y.shape()[0] == connection_permence.shape()[0]);
		for(size_t i=0;i<connection_permence.shape()[0];i++) {
			if(xt::view(y, i)[0] != true)
				continue;
			
			for(size_t j=0;j<connection_permence.shape()[1];j++) {
				auto v = xt::view(connection_permence, i, j);
				if(xt::dynamic_view(x, sliceVec(xt::view(connection_target, i, j)[0]))[0] == true)
					v += perm_incerment;
				else
					v -= perm_decarment;
			}
		}
	}

	void growSynasps(const xt::xarray<bool>& x, const xt::xarray<bool>& learn)
	{
		//D only now
		std::vector<size_t> active_input;
		//TODO: This is slow
		for(size_t i=0;i<x.size();i++) {
			if(x[i] == true)
				active_input.push_back(i);
		}
		
		for(size_t i=0;i<learn.size();i++) {
			if(learn[i] == false)
				continue;
			
			std::vector<size_t> possible_new_connection;
			for(size_t i=0;i<connection_target.shape()[1];i++) {
				size_t ci = unfoldIndex(connection_target[i], x.shape());
				//auto v = 
				//if(std::find())
			}
		}
	}

	xt::xarray<float> connection_permence;
	xt::xarray<std::vector<size_t>> connection_target;
};

void allInputCellInternal(int depth, const std::vector<size_t>& shape, std::vector<size_t> curr_iter, std::vector<std::vector<size_t>>& res) {
	if(depth == shape.size()) {
		res.push_back(curr_iter);
		return;
	}
	for(size_t i=0;i<shape[i];i++) {
		auto n = curr_iter;
		n.push_back(i);
		allInputCellInternal(depth+1, shape, std::move(n), res);
	}

};

std::vector<std::vector<size_t>> allInputCell(const std::vector<size_t>& input_shape)
{
	std::vector<std::vector<size_t>> res;
	size_t vol = 1;
	for(auto l : input_shape)
		vol *= l;
	res.resize(vol);
	
	allInputCellInternal(0, input_shape, {}, res);
	return res;
}

xt::xarray<bool> globalInhibition(const xt::xarray<uint32_t>& x, float density)
{
	std::vector<std::pair<int32_t, size_t>> v(x.size());
	for(size_t i=0;i<x.size();i++)
		v[i] = {x[i], i};
	std::sort(v.begin(), v.end(), [](auto& a, auto&b){return a.first > b.first;});

	xt::xarray<bool> res = xt::zeros<bool>(x.shape());
	uint32_t min_accept_val = v[x.size()*density].first;
	for(size_t i=0;v[i].first >= min_accept_val;i++)
		res[v[i].second] = true;
	return res;
}

//TODO: Implement Boosting
class SpatialPooler1D
{
public:
	SpatialPooler1D(size_t input_size, size_t output_size, float potential_pool_pct=0.75f)
		: cells_(output_size, input_size*potential_pool_pct), output_size_(output_size)
	{
		std::vector<std::vector<size_t>> connection_cells = allInputCell({input_size});
		int idx = 0; std::generate(connection_cells.begin(), connection_cells.end(), [&idx](){return std::vector<size_t>{(size_t)idx++};});

		std::random_device rd;
		std::mt19937 rng(rd());
		auto genPermence = [&rng](){std::normal_distribution<float> perm_dist(0.15, 1); return std::max(std::min(perm_dist(rng), 1.f), 0.f);};

		size_t num_max_connection = input_size*potential_pool_pct;
		for(size_t i=0;i<output_size;i++) {
			std::random_shuffle(connection_cells.begin(), connection_cells.end());
			for(size_t j=0;j<num_max_connection;j++) {
				xt::view(cells_.connection_permence, i, j) = genPermence();
				xt::view(cells_.connection_target, i, j) = connection_cells[j];
			}
		}
	}

	xt::xarray<bool> compute(const xt::xarray<bool>& x, bool learn) 
	{
		xt::xarray<uint32_t> overlap_score = cells_.calculateOverlapScore(x);

		//Global inhibition
		xt::xarray<bool> res = globalInhibition(overlap_score, global_density);
		//End of global inhibition

		if(learn == true)
			 cells_.learnCorrelation(x, res, 0.045, 0.045);

		return res;
	}

	CellSheet cells_;
	size_t output_size_;
	bool global_inhibition = true;
	float global_density = 0.1f;
};

class TemporalMemory1D
{
public:
	TemporalMemory1D(size_t num_input, size_t cells_per_column)
		: cells_(num_input*cells_per_column, 64)
	{
		active_cells_ = xt::zeros<bool>({num_input, cells_per_column});
	}

	xt::xarray<bool> compute(const xt::xarray<bool>& x, bool learn)
	{
		xt::xarray<bool> active_cells_post_burst = applyBurst(x, active_cells_);
		xt::xarray<uint32_t> overlap = cells_.calculateOverlapScore(active_cells_post_burst);
		
		xt::xarray<bool> predictive = globalInhibition(overlap, 0.1f);
		xt::xarray<bool> pred = xt::cast<bool>(xt::sum(predictive, 0));

		if(learn == true) {
			static std::mt19937 rng;
			//Learn sequence memory
			xt::xarray<bool> learn_cells = active_cells_post_burst;
			std::uniform_int_distribution<size_t> dist(0, learn_cells.shape()[1]);
			for(auto i=0;i<learn_cells.shape()[0];i++) {
				auto v = xt::view(learn_cells, i);
				if(xt::sum(v)[0] == v.size()) {
					v = false;
					v[dist(rng)] = true;
				}
			}
			cells_.growSynasps(active_cells_, learn_cells);
			cells_.learnCorrelation(active_cells_, learn_cells, 0.1, 0.1);
		}
		active_cells_ = pred;
		return pred;
	}

	static xt::xarray<bool> applyBurst(const xt::xarray<bool>& x, const xt::xarray<bool>& s)
	{
		if(x.shape().size() != 1)
			throw std::runtime_error("applyBurst: input data must be 1D");
		if(x.shape()[0] != s.shape()[0])
			throw std::runtime_error("applyBurst: input data shape mismach with internal state");
		xt::xarray<bool> res = s;
		for(size_t i=0;i<x.shape()[0];i++) {
			auto v = xt::view(s, i);
			if(xt::view(x, i)[0] == true && xt::sum(v)[0] == 0)
				xt::view(res, i) = true;
		}
		return res;
	}
	CellSheet cells_;
	xt::xarray<bool> active_cells_;
};

template<typename ResType, typename InType>
inline ResType as(const InType& shape)
{
	return ResType(shape.begin(), shape.end());
}

//Classifers
struct SDRClassifer
{
	SDRClassifer(size_t num_classes, std::vector<size_t> shape)
		: stored_patterns(num_classes, xt::zeros<int>(as<xt::xarray<int>::shape_type>(shape)))
		, pattern_sotre_num(num_classes)
	{}

	void add(size_t category, const xt::xarray<bool>& t)
	{
		stored_patterns[category] += t;
		pattern_sotre_num[category] += 1;
	}

	size_t compute(const xt::xarray<bool>& t, float bit_common_threhold = 0.5) const
	{
		assert(bit_common_threhold >= 0.f && bit_common_threhold <= 1.f);
		size_t best_pattern = 0;
		size_t best_score = 0;
		for(size_t i=0;i<numPatterns();i++) {
			auto overlap = t & xt::cast<bool>(stored_patterns[i] >= (int)(pattern_sotre_num[i]*bit_common_threhold));
			size_t overlap_score = xt::sum(overlap)[0];
			if(overlap_score > best_score) {
				best_score = overlap_score;
				best_pattern = i;
			}
		}

		return best_pattern;
	}

	size_t numPatterns() const
	{
		return stored_patterns.size();
	}

	void reset()
	{
		for(size_t i=0;i<numPatterns();i++) {
			stored_patterns[i] = 0;
			pattern_sotre_num[i] = 0;
		}
	}

protected:
	std::vector<xt::xarray<int>> stored_patterns;
	std::vector<size_t> pattern_sotre_num;
};