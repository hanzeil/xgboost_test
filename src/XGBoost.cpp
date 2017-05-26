//
// Created by tangjinghao on 2017/5/26.
//

#include <sstream>
#include <cmath>
#include "XGBoost.h"

XGBoost::XGBoost() : cur_training_set_num_(0), training_set_num_(0),
                     h_train_(NULL), h_test_(NULL) {
}

void XGBoost::set_training_set_num(std::size_t num) {
    training_set_num_ = num;
    h_train_ = new DMatrixHandle[training_set_num_];
}

bool XGBoost::add_training_set_from_csv(
        const string &in_file_path,
        const std::size_t rows,
        const std::size_t cols) {
    std::ifstream in_file(in_file_path);
    std::string item;

    float *train_datas = NULL;
    float *train_labels = NULL;
    train_datas = (float *) malloc(rows * (cols - 1) * sizeof(float));
    train_labels = (float *) malloc(rows * sizeof(float));
    for (int i = 0; std::getline(in_file, item) && i < rows; i++) {
        std::size_t begin = 0;
        for (int j = 0; j < cols; j++) {
            std::size_t pos = item.find_first_of(',', begin);
            if (pos == string::npos) {
                if (j != cols - 1) {
                    return false;
                }
                std::string tmp_string = string(item.begin() + begin, item.end());
                train_labels[i] = (float) std::atof(tmp_string.c_str());
                break;
            }
            std::string tmp_string = string(item.begin() + begin, item.begin() + pos);
            train_datas[i * (cols - 1) + j] = (float) std::atof(tmp_string.c_str());
            // cout << train_datas[i][j] << " ";
            begin = pos + 1;
        }
        // cout << train_labels[i] << endl;
    }
    XGDMatrixCreateFromMat(train_datas, rows, cols - 1, -1, &h_train_[cur_training_set_num_]);

    XGDMatrixSetFloatInfo(h_train_[cur_training_set_num_], "label", train_labels, rows);

    /*
    //read back the labels, just a sanity check
    bst_ulong bst_result;
    const float *out_floats;
    XGDMatrixGetFloatInfo(h_train_, "label", &bst_result, &out_floats);
    for (unsigned int i = 0; i < bst_result; i++)
        std::cout << "label[" << i << "]=" << out_floats[i] << std::endl;
    */
    cur_training_set_num_++;
    return true;
}

bool XGBoost::train() {
// create the booster and load some parameters
    XGBoosterCreate(&h_train_, 1, &h_booster_);
    XGBoosterSetParam(h_booster_, "booster", "gbtree");
    XGBoosterSetParam(h_booster_, "objective", "reg:linear");
    XGBoosterSetParam(h_booster_, "max_depth", "6");
    XGBoosterSetParam(h_booster_, "eta", "0.1");
    XGBoosterSetParam(h_booster_, "min_child_weight", "1");
    XGBoosterSetParam(h_booster_, "subsample", "0.9");
    XGBoosterSetParam(h_booster_, "colsample_bytree", "1");
    XGBoosterSetParam(h_booster_, "num_parallel_tree", "1");
// perform 200 learning iterations
    for (int iter = 0; iter < 200; iter++)
        XGBoosterUpdateOneIter(h_booster_, iter, h_train_);
}

bool XGBoost::set_test_set_from_csv(const string &in_file_path, const std::size_t rows, const std::size_t cols) {
    if (h_test_) {
        XGDMatrixFree(h_test_);
        h_test_ = NULL;
    }

}

bool XGBoost::create_matrix_from_csv(const string &in_file_path, const std::size_t rows, const std::size_t cols,
                                     DMatrixHandle &h_train_) {


}