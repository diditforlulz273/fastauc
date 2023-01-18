#include <vector>
#include <iostream>
#include <algorithm>
#include <tuple>
#include <type_traits>


// Fill the zipped vector with pairs consisting of the
// corresponding elements of a and b. (This assumes 
// that the vectors have equal length)
template<typename tuple_type> void zip(
    const bool* a, 
    const float* b,
    const float* sample_weight,
    const size_t len, 
    std::vector<tuple_type> &zipped)
{
    for(size_t i=0; i<len; ++i)
    {
        if constexpr(std::is_same<tuple_type,std::tuple<bool,float,float>>::value){
            zipped.push_back(std::make_tuple(a[i], b[i], sample_weight[i]));
        }
        else{
            zipped.push_back(std::make_tuple(a[i], b[i]));
        }
    }
}

double trapezoid_area(double x1, double x2, double y1, double y2) {
  double dx = x2 - x1;
  double dy = y2 - y1;
  return dx * y1 + dy * dx / 2.0;
}

template<typename tuple_type> float auc_kernel(float* ts, bool* st, size_t len,float* sample_weight) {
  // sort the data
  // Zip the vectors together
  std::vector<tuple_type> zipped;
  zipped.reserve(len);
  zip<tuple_type>(st, ts, sample_weight, len, zipped);

  // Sort the vector of pairs
  std::sort(std::begin(zipped), std::end(zipped), 
    [&](const auto& a, const auto& b)
    {
        return std::get<1>(a) > std::get<1>(b);
    });

  double fps = 0;
  double tps = 0;
  double last_counted_fps = 0;
  double last_counted_tps = 0;
  double auc = 0.0;
  for(size_t i=0; i<zipped.size(); ++i) {
    if constexpr(std::is_same<tuple_type,std::tuple<bool,float,float>>::value){
        tps += std::get<0>(zipped[i]) * std::get<2>(zipped[i]);
        fps += (1 - std::get<0>(zipped[i])) * std::get<2>(zipped[i]);
    }
    else{
        tps += std::get<0>(zipped[i]);
        fps += (1 - std::get<0>(zipped[i]));
    }
    if( (i == zipped.size() - 1) || (std::get<1>(zipped[i+1]) != std::get<1>(zipped[i])) ) {
        auc += trapezoid_area(last_counted_fps, fps, last_counted_tps, tps);
        last_counted_fps = fps;
        last_counted_tps = tps;
    }
  }
  return auc / (tps * fps);
}

extern "C" {
    float cpp_auc_ext(float* ts, bool* st, size_t len, float* sample_weight,size_t n_sample_weights) {
        if(n_sample_weights>0){
            return auc_kernel<std::tuple<bool,float,float>>(ts, st, len, sample_weight);
        }
        else{
            return auc_kernel<std::tuple<bool,float>>(ts, st, len, sample_weight);
        }
    }
}
