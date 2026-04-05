#pragma once

#include "Types.hpp"
#include "Config.hpp"
#include <vector>
#include <string>

class Dataset {
public:
    virtual ~Dataset() = default;

    virtual Frame getFrame(int idx) = 0;

    const Intrinsic& getIntrinsic() const { return intrinsic_; }
    size_t size() const { return associations_.size(); }

protected:
    Intrinsic intrinsic_;
    std::vector<Association> associations_;
};
