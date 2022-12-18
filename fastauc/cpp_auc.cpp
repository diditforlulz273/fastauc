
#include <vector>
#include <iostream>
#include <algorithm>
#include <iterator>


// Fill the zipped vector with pairs consisting of the
// corresponding elements of a and b. (This assumes 
// that the vectors have equal length)
void zip(
    const bool* a, 
    const float* b,
    const size_t len, 
    std::vector<std::pair<bool,float>> &zipped)
{
    for(size_t i=0; i<len; ++i)
    {
        zipped.push_back(std::make_pair(a[i], b[i]));
    }
}

double trapezoid_area(double x1,double x2,double y1,double y2){
  double dx = x2-x1;
  double dy = y2-y1;
  return dx*y1+dy*dx/2.0;
}

float auc_kernel(float* ts, bool* st, size_t len) {
  // sort the data
  // Zip the vectors together
  std::vector<std::pair<bool,float>> zipped;
  zipped.reserve(len);
  zip(st, ts, len, zipped);

  // Sort the vector of pairs
  std::sort(std::begin(zipped), std::end(zipped), 
    [&](const auto& a, const auto& b)
    {
        return a.second > b.second;
    });

  double prev_fps = 0;
  double prev_tps = 0;
  double last_counted_fps = 0;
  double last_counted_tps = 0;
  double auc = 0.0;
  for(size_t i=0;i<zipped.size();++i){
    const double tps = prev_tps + zipped[i].first;
    const double fps = prev_fps + (1-zipped[i].first);
    if( (i==zipped.size()-1) || (zipped[i+1].second!=zipped[i].second) ){
        auc += trapezoid_area(last_counted_fps,fps,last_counted_tps,tps);
        last_counted_fps = fps;
        last_counted_tps = tps;
    }
    prev_tps = tps;
    prev_fps = fps;
  }
  return auc/(prev_tps*prev_fps);
}

extern "C" {
    float cpp_auc_ext(float* ts, bool* st, size_t len) {
        return auc_kernel(ts, st, len);
    }

}
