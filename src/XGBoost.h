//
// Created by tangjinghao on 2017/5/26.
//

#ifndef XGBOOST_TEST_XGBOOST_H
#define XGBOOST_TEST_XGBOOST_H


#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include "xgboost/c_api.h"

using std::string;
using std::cout;
using std::endl;

class XGBoost {
public:
    XGBoost();

    ~XGBoost();

    bool add_training_set_from_csv(const string &in_file_path, const std::size_t rows, const std::size_t cols);

    bool set_test_set_from_csv(const string &in_file_path, const std::size_t rows, const std::size_t cols);

    void set_training_set_num(std::size_t num);

    bool train();

    bool predict();

    void precision_and_recall();

private:
    DMatrixHandle *h_train_;
    DMatrixHandle *h_test_;
    BoosterHandle h_booster_;
    const float *predict_labels;
    std::size_t training_set_num_;
    std::size_t cur_training_set_num_;

    bool create_matrix_from_csv(const string &in_file_path, const std::size_t rows, const std::size_t cols,
                                DMatrixHandle &h_train);


    int round(float num);

};

#endif //XGBOOST_TEST_XGBOOST_H
