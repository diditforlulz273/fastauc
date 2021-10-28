
#include <vector>
#include <iostream>
#include <algorithm>
#include <iterator>


// Fill the zipped vector with pairs consisting of the
// corresponding elements of a and b. (This assumes 
// that the vectors have equal length)
void zip(
    const std::vector<bool> &a, 
    const std::vector<float> &b, 
    std::vector<std::pair<bool,float>> &zipped)
{
    for(size_t i=0; i<a.size(); ++i)
    {
        zipped.push_back(std::make_pair(a[i], b[i]));
    }
}

// Write the first and second element of the pairs in 
// the given zipped vector into a and b. (This assumes 
// that the vectors have equal length)
void unzip(
    const std::vector<std::pair<bool, float>> &zipped, 
    std::vector<bool> &a, 
    std::vector<float> &b)
{
    for(size_t i=0; i<a.size(); i++)
    {
        a[i] = zipped[i].first;
        b[i] = zipped[i].second;
    }
}

 float auc_kernel(float* ts, bool* st, size_t len) {

  std::vector <float> test;   
  std::vector <bool> status;                  
  std::vector <float> out;
  std::vector <float> neg_dup_cnt;
  std::vector <float> pos_dup_cnt;
  std::vector <float> neg_all_cnt;
  std::vector <float> pos_all_cnt;
  std::size_t next_row;

  int ndup_pos = 0, ndup_neg = 0, neg_cnt = 0, pos_cnt = 0;
  float total_pos = 0, pos_sum = 0, auc = 0, prev_val;

  // convert C-style arrays to C++ vectors
  test.assign(ts, ts + len);
  status.assign(st, st + len);

  // sort the data
  // Zip the vectors together
  std::vector<std::pair<bool,float>> zipped;
  zip(status, test, zipped);

  // Sort the vector of pairs
  std::sort(std::begin(zipped), std::end(zipped), 
    [&](const auto& a, const auto& b)
    {
        return a.second < b.second;
    });

  // Write the sorted pairs back to the original vectors
  unzip(zipped, status, test);

  // iterate over the data to determine the number of duplicates at each
  // unique value as well as the number of total true positive observations

  for (std::size_t row_i = 0; row_i < status.size(); ++row_i){

    // determine the total number of true positives
    if(status[row_i]){
      total_pos++;
    }

    // identify if the current record has a new test value
    if(row_i == 0 || prev_val != test[row_i]){

      next_row = row_i + 1;
      ndup_pos = 0;
      ndup_neg = 0;

      while(true){

        // if you're at the last row stop
        if(row_i == (test.size() - 1)){
          break;
        }

        // count duplicates in the next rows
        if(test[row_i] == test[next_row]){

          if(status[next_row]){
            ndup_pos++;
            pos_cnt++;
          } else{
            ndup_neg++;
            neg_cnt++;
          }

          next_row++;

          // see if you just looked at the last value
          if(next_row == test.size()) break;
        } else break;
      }
      if(status[row_i]){
        ndup_pos++;
        pos_cnt++;
        }
      else{
        ndup_neg++;
        neg_cnt++;
      }
      prev_val = test[row_i];
    }

  neg_dup_cnt.push_back(ndup_neg);
  pos_dup_cnt.push_back(ndup_pos);

  neg_all_cnt.push_back(neg_cnt);
  pos_all_cnt.push_back(pos_cnt);

  }

  // iterate through the data to determine X and Y values per DeLong
  for (std::size_t row_i = 0; row_i < status.size(); ++row_i){

    if(status[row_i]){
      out.push_back(neg_all_cnt[row_i] - 0.5 * neg_dup_cnt[row_i]);
    } else {
      out.push_back(total_pos - pos_all_cnt[row_i] + 0.5 * pos_dup_cnt[row_i]);
    }

  }

  // get sum of all positive examples' predictions
  for (int i=0; i<out.size(); ++i) {
    if (status[i])
      pos_sum += out[i];
  }

  // calc auc finally
  auc = pos_sum / total_pos / (status.size() - total_pos);
  return(auc);
}

extern "C" {
    float cpp_auc_ext(float* ts, bool* st, size_t len) {
        return auc_kernel(ts, st, len);
    }

}
