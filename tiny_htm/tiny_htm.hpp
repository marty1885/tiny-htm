#pragma once

#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xindex_view.hpp>

#include <vector>
#include <algorithm>
#include <random>

#include <assert.h>

template<typename ResType, typename InType>
inline ResType as(const InType& shape)
{
	return ResType(shape.begin(), shape.end());
}

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
		float encode_space = sdr_length - active_bits;
		int start = encode_space*value;
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

template <typename ShapeType>
std::vector<size_t> foldIndex(size_t index, const ShapeType& shape)
{
	assert(shape.size() != 0);
	std::vector<size_t> v(shape.size());
	size_t acc = 1;
	for(int i=(int)shape.size()-1;i>=0;i--) {
		acc *= shape[i];
		v[i] = acc;
	}
	std::vector<size_t> res(v.size());
	for(size_t i=1;i<v.size();i++) {
		res[i-1] = index/v[i];
		index = index%v[i];
	}
	res.back() = index;
	return res;
}

template <typename IdxType, typename ShapeType>
inline size_t unfoldIndex(const IdxType& index, const ShapeType& shape)
{
	size_t s = 0;
	size_t v = 1;
	assert(index.size() == shape.size());
	for(int i=(int)index.size()-1;i>=0;i--) {
		v *= (i==(int)index.size()-1?1:shape[i+1]);
		s += index[i] * v;
	}

	return s;
}

void _allPosition(size_t depth, const std::vector<size_t>& shape, std::vector<size_t> curr_iter, std::vector<std::vector<size_t>>& res)
{
	if(depth == shape.size()) {
		res.push_back(curr_iter);
		return;
	}
	for(size_t i=0;i<shape[depth];i++) {
		auto n = curr_iter;
		n.push_back(i);
		_allPosition(depth+1, shape, std::move(n), res);
	}

};

std::vector<std::vector<size_t>> allPosition(const std::vector<size_t>& input_shape)
{
	std::vector<std::vector<size_t>> res;
	size_t vol = 1;
	for(auto l : input_shape)
		vol *= l;
	res.reserve(vol);
	
	_allPosition(0, input_shape, {}, res);
	return res;
}

template <typename T, typename Compare>
inline std::vector<std::size_t> sort_permutation(
    const std::vector<T>& vec,
    Compare compare)
{
	std::vector<std::size_t> p(vec.size());
	std::iota(p.begin(), p.end(), 0);
	std::sort(p.begin(), p.end(),
		[&](std::size_t i, std::size_t j){ return compare(vec[i], vec[j]); });
	return p;
}

template <typename T>
std::vector<T> apply_permutation(
    const std::vector<T>& vec,
    const std::vector<std::size_t>& p)
{
	std::vector<T> sorted_vec(vec.size());
	std::transform(p.begin(), p.end(), sorted_vec.begin(),
		[&](std::size_t i){ return vec[i]; });
	return sorted_vec;
}

template <typename T>
const T& ndIndexing(const xt::xarray<T>& arr,const std::vector<size_t>& idx)
{
	return arr.storage()[unfoldIndex(idx, arr.shape())];
}


template <typename T>
T& ndIndexing(xt::xarray<T>& arr,const std::vector<size_t>& idx)
{
	return arr.storage()[unfoldIndex(idx, arr.shape())];
}

struct Cells
{
	Cells() = default;
	Cells(std::vector<size_t> cell_shape, size_t max_connection_per_cell)
		: max_connection_per_cell_(max_connection_per_cell)
	{
		connections_ = decltype(connections_)::from_shape(as<decltype(connections_)::shape_type>(cell_shape));
		permence_ = decltype(permence_)::from_shape(as<decltype(permence_)::shape_type>(cell_shape));
	}

	size_t size() const
	{
		return connections_.size();
	}

	std::vector<size_t> shape() const
	{
		return as<std::vector<size_t>>(connections_.shape());
	}

	void connect(size_t input_pos, size_t cell_pos, float initial_permence)
	{
		auto& connection_list = connections_[cell_pos];
		auto& permence_list = permence_[cell_pos];

		if(connection_list.size() == max_connection_per_cell_) return;//throw std::runtime_error("Synapes are full in cells");
		
		connection_list.push_back(input_pos);
		permence_list.push_back(initial_permence);
	}

	xt::xarray<uint32_t> calcOverlap(const xt::xarray<bool>& x, float connected_thr) const
	{
		xt::xarray<uint32_t> res = xt::xarray<uint32_t>::from_shape(shape());

		#pragma omp parallel for schedule(guided)
		for(size_t i=0;i<size();i++) {
			const auto& connections = connections_[i];
			const auto& permence = permence_[i];

			assert(connections.size() == permence.size());
			uint32_t score = 0;
			for(size_t j=0;j<connections.size();j++) {
				if(permence[j] < connected_thr)
					continue;
				bool bit = x[connections[j]];
				score += bit;
			}
			res[i] = score;
		}
		return res;
	}

	void learnCorrilation(const xt::xarray<bool>& x, const xt::xarray<bool>& learn, float perm_inc, float perm_dec)
	{
		assert(connections_.size() == learn.size()); // A loose check for the same shape
		auto clamp = [](float x) {return std::min(1.f, std::max(x, 0.f));};

		#pragma omp parallel for
		for(size_t i=0;i<connections_.size();i++) {
			if(learn[i] == false)
				continue;

			const auto& connections = connections_[i];
			auto& permence = permence_[i];
			for(size_t j=0;j<connections.size();j++) {
				bool bit = x[connections[j]];
				if(bit == true)
					permence[j] += perm_inc;
				else
					permence[j] -= perm_dec;
				permence[j] = clamp(permence[j]);
			}
		}
	}

	void growSynapse(const xt::xarray<bool>& x, const xt::xarray<bool> learn, float perm_init)
	{
		assert(learn.size() == size()); //A loose check
		std::vector<size_t> all_on_bits;
		for(size_t i=0;i<x.size();i++) {
			if(x[i] == true)
				all_on_bits.push_back(i);
		}
		if(all_on_bits.size() == 0)
			return;

		#pragma omp parallel for schedule(guided) //TODO: guided might not be the best for cases with large amount of cells
		for(size_t i=0;i<learn.size();i++) {
			if(learn[i] == false)
				continue;

			auto& connections = connections_[i];
			if(connections.size() == max_connection_per_cell_)
				continue;

			std::vector<bool> connection_list(learn.size());
			for(size_t i=0;i<connections.size();i++)
				connection_list[connections[i]] = true;

			for(const auto& input : all_on_bits) {
				if(connections.size() == max_connection_per_cell_) //Don't make new connections if full
					break;
				if(connection_list[input] == true)
					continue;

				//#pragma omp critical //No need to make it a critial session as we are modifing 
				//                     //Different list of synapses every tiem. No rac condition will occur
				connect(input, i, perm_init);
			}
		}
	}

	void sortSynapse()
	{
		assert(connections_.size() == permence_.size());

		#pragma omp parallel for
		for(size_t i=0;i<connections_.size();i++) {
			auto& connections = connections_[i];
			auto& permence = permence_[i];
			auto p = sort_permutation(connections, [](auto a, auto b){return a<b;});
			connections = apply_permutation(connections, p);
			permence = apply_permutation(permence, p);

		}
	}

	void decaySynapse(float thr)
	{
		assert(connections_.size() == permence_.size());
		#pragma omp parallel for
		for(size_t i=0;i<connections_.size();i++) {
			auto& connections = connections_[i];
			auto& permence = permence_[i];
			std::vector<size_t> remove_list(connections.size());

			for(size_t i=0;i<connections.size();i++)
				remove_list[i] = permence[i] < thr;

			connections.erase(std::remove_if(connections.begin(), connections.end()
				, [&](const auto& a){return remove_list[&a - connections.data()];}), connections.end());
			permence.erase(std::remove_if(permence.begin(), permence.end()
				, [&](const auto& a){return remove_list[&a - permence.data()];}), permence.end());
		}
	}

	xt::xarray<std::vector<size_t>> connections_;
	xt::xarray<std::vector<float>> permence_;
	size_t max_connection_per_cell_;
};

xt::xarray<bool> globalInhibition(const xt::xarray<uint32_t>& x, float density)
{
	std::vector<std::pair<uint32_t, size_t>> v;
	size_t target_size = x.size()*density;
	v.reserve(target_size);//Some sane value
	for(size_t i=0;i<x.size();i++) {
		if(x[i] != 0)
			v.push_back({x[i], i});
	}
	std::sort(v.begin(), v.end(), [](const auto& a, const auto&b){return a.first > b.first;});

	xt::xarray<bool> res = xt::xarray<bool>::from_shape(x.shape());
	#pragma omp parallel for
	for(size_t i=0;i<res.size();i++)
		res[i] = false;
	uint32_t min_accept_val = v[std::min(target_size, v.size())].first;
	auto it = std::upper_bound(v.begin(), v.end(), min_accept_val, [](const auto& a, const auto& b){return a > b.first;});
	size_t stop_index = std::distance(v.begin(), it);

	for(size_t i=0;i<stop_index;i++)
		res[v[i].second] = true;
	return res;
}

std::vector<size_t> vector_range(size_t start, size_t end)
{
	std::vector<size_t> v(end-start);
	for(size_t i=0;i<end-start;i++)
		v[i] = i;
	return v;
}

//TODO: implement boosting, topology. The SP breaks spatial infomation tho
struct SpatialPooler
{
	SpatialPooler(std::vector<size_t> input_shape, std::vector<size_t> output_shape, float potential_pool_pct=0.75, size_t seed=42)
		: cells_(output_shape, std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>())*potential_pool_pct)
	{
		if(potential_pool_pct > 1 or potential_pool_pct < 0)
			throw std::runtime_error("potential_pool_pct must be between 0~1, but get" + std::to_string(potential_pool_pct));
		
		//Initalize potential pool
		std::mt19937 rng(seed);
		size_t input_cell_num = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());
		size_t potential_pool_connections = input_cell_num*potential_pool_pct;
		std::vector<size_t> all_input_cell = vector_range(0, input_cell_num);
		for(size_t i=0;i<cells_.size();i++) {
			auto& connections = cells_.connections_[i];
			auto& permence = cells_.permence_[i];
			connections.resize(potential_pool_connections);
			permence.resize(potential_pool_connections);

			std::shuffle(all_input_cell.begin(), all_input_cell.end(), rng);
			std::normal_distribution<float> dist(0.5, 1);
			auto clamp =[](float x) {return std::min(1.f, std::max(x, 0.f));};
			for(size_t j=0;j<potential_pool_connections;j++) {
				connections[j] = all_input_cell[j];
				permence[j] = clamp(dist(rng));
			}
		}
		cells_.sortSynapse();
	}

	xt::xarray<bool> compute(const xt::xarray<bool>& x, bool learn)
	{
		xt::xarray<uint32_t> overlap_score = cells_.calcOverlap(x, 0.21);
		xt::xarray<bool> res = globalInhibition(overlap_score, global_density_);

		if(learn == true)
			cells_.learnCorrilation(x, res, permanence_incerment_, permanence_decerment_);
		return res;
	}

	Cells cells_;

	//All the getter and seters
	void setPermanenceIncerment(float v) {permanence_incerment_ = v;}
	void setPermanenceDecerment(float v) {permanence_decerment_ = v;}
	void setConnectedPermanence(float v) {connected_permanence_ = v;}

	float permanenceIncerment() const {return permanence_incerment_;}
	float permanenceDecerment() const {return permanence_decerment_;}
	float connectedPermanence() const {return permanence_decerment_;}

	float permanence_incerment_ = 0.1f;
	float permanence_decerment_ = 0.1f;
	float connected_permanence_ = 0.15;

	float global_density_ = 0.15;
};

xt::xarray<bool> applyBurst(const xt::xarray<bool>& s, const xt::xarray<bool>& x)
{
	assert(s.dimension() == x.dimension()+1);
	assert(s.size()/s.shape().back() == x.size());
	xt::xarray<bool> res = xt::xarray<bool>::from_shape(s.shape());
	size_t column_size = res.shape().back();

	#pragma omp parallel for
	for(size_t i=0;i<res.size()/column_size;i++) {
		for(size_t j=0;j<column_size;j++)
			res[i*column_size+j] = s[i*column_size+j];

		if(x[i] == false)
			continue;

		size_t sum = 0;
		for(size_t j=0;j<column_size;j++)
			sum += s[i*column_size+j];
		if(sum == 0) {
			for(size_t j=0;j<column_size;j++)
				res[i*column_size+j] = true;
		}
	}
	return res;
}

xt::xarray<bool> selectLearningCell(const xt::xarray<bool>& x)
{
	assert(x.dimension() >= 2);
	static std::mt19937 rng;
	size_t column_size = x.shape().back();
	std::uniform_int_distribution<size_t> dist(0, column_size-1);
	xt::xarray<bool> res = xt::xarray<bool>::from_shape(x.shape());

	#pragma omp parallel for
	for(size_t i=0;i<x.size()/column_size;i++) {
		size_t sum = 0;
		for(size_t j=0;j<column_size;j++)
			sum += x[i*column_size+j];

		if(sum == column_size) {
			for(size_t j=0;j<column_size;j++)
				res[i*column_size+j] = false;
			res[i*column_size+dist(rng)] = true;
		}
		else {
			for(size_t j=0;j<column_size;j++)
				res[i*column_size+j] = x[i*column_size+j];
		}
	}
	return res;
}

struct TemporalMemory
{
	TemporalMemory(const std::vector<size_t>& data_shape, size_t cells_per_column=16, size_t segments_per_cell = 1024)
	{
		std::vector<size_t> cell_shape = data_shape;
		cell_shape.push_back(cells_per_column);
		cells_ = Cells(cell_shape, segments_per_cell);

		predictive_cells_ = xt::zeros<bool>(cell_shape);
		active_cells_ = xt::zeros<bool>(cell_shape);
	}

	xt::xarray<bool> compute(const xt::xarray<bool>& x, bool learn)
	{
		xt::xarray<bool> active_cells = applyBurst(predictive_cells_, x);
		xt::xarray<uint32_t> overlap = cells_.calcOverlap(active_cells, connected_permanence_);
		predictive_cells_ = (overlap > 2); //TODO: Arbitrary value
		if(learn == true) {
			xt::xarray<bool> apply_learning = selectLearningCell(active_cells);
			xt::xarray<bool> last_active = selectLearningCell(active_cells_);
			cells_.learnCorrilation(last_active, apply_learning, permanence_incerment_, permanence_decerment_);
			cells_.growSynapse(last_active, apply_learning, initial_permanence_);
		}
		active_cells_ = active_cells;
		return xt::sum(predictive_cells_, -1);
	}

	void reset()
	{
		predictive_cells_ = xt::zeros<bool>(cells_.shape());
		active_cells_ = xt::zeros<bool>(cells_.shape());
	}

	void organizeSynapse()
	{
		cells_.sortSynapse();
	}

	//All the getter and seters
	void setPermanenceIncerment(float v) {permanence_incerment_ = v;}
	void setPermanenceDecerment(float v) {permanence_decerment_ = v;}
	void setInitialPermanence(float v) {initial_permanence_ = v;}
	void setConnectedPermanence(float v) {connected_permanence_ = v;}

	float permanenceIncerment() const {return permanence_incerment_;}
	float permanenceDecerment() const {return permanence_decerment_;}
	float initialPermanence() const {return initial_permanence_;}
	float connectedPermanence() const {return permanence_decerment_;}

	float permanence_incerment_ = 0.1f;
	float permanence_decerment_ = 0.1f;
	float initial_permanence_ = 0.21f;
	float connected_permanence_ = 0.15;

	Cells cells_;
	xt::xarray<bool> predictive_cells_;
	xt::xarray<bool> active_cells_;
};

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
