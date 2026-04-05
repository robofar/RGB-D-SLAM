#pragma once

#include "Dataset.hpp"
#include <string>

class ReplicaDataset : public Dataset {
public:
    ReplicaDataset(const Config& cfg);

    Frame getFrame(int idx) override;

private:
    std::string results_dir_;
};
