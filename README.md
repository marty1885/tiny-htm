# tiny-htm
tiny-htm is a flexable HTM framework for research and all other purpose you can think of. In pure C++.

This project is in it's really early ages. Archicture breaking changes will happen.

## Features
* Both major HTM algorithms
  * Spatial Pooler
  * Temporal Memory
* Easily workable with Pyhon. It's implemented using xtensor!


## Dependency
* [xtensor](https://github.com/QuantStack/xtensor/)
* A C++17 capable compiler
* OpenMP capable compiler (optional)

## Build and install
```shell
mkdir build
cmake ..
make -j8
```

To install
```
make install
```

## The TODO list

### Encoders
* [ ] Grid Cells
* [ ] Date time
* [ ] RDSE

### Spatial Pooler
* [ ] Boosting

### Misc
* [ ] Performance optimization
  * [x] Overhead reduction.
  * [x] benchmark tools
* [ ] Parallel algorithm execution
  * [x] Add OpenMP support.
* [ ] Saving the models
* [ ] Make hyper parameter changable


### Documentation
* [ ] add Anything
