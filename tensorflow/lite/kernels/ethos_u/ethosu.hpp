/*
 * Copyright (c) 2020-2021 Arm Limited. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

/*
 *The following undef are necessary to avoid clash with macros in GNU C Library
 * if removed the following warning/error are produced:
 *
 *  In the GNU C Library, "major" ("minor") is defined
 *  by <sys/sysmacros.h>. For historical compatibility, it is
 *  currently defined by <sys/types.h> as well, but we plan to
 *  remove this soon. To use "major" ("minor"), include <sys/sysmacros.h>
 *  directly. If you did not intend to use a system-defined macro
 *  "major" ("minor"), you should undefine it after including <sys/types.h>.
 */
#undef major
#undef minor

namespace EthosU {

class Exception : public std::exception {
public:
    Exception(const char *msg);
    virtual ~Exception() throw();
    virtual const char *what() const throw();

private:
    std::string msg;
};

/**
 * Sematic Version : major.minor.patch
 */
class SemanticVersion {
public:
    SemanticVersion(uint32_t _major = 0, uint32_t _minor = 0, uint32_t _patch = 0) :
        major(_major), minor(_minor), patch(_patch){};

    bool operator==(const SemanticVersion &other);
    bool operator<(const SemanticVersion &other);
    bool operator<=(const SemanticVersion &other);
    bool operator!=(const SemanticVersion &other);
    bool operator>(const SemanticVersion &other);
    bool operator>=(const SemanticVersion &other);

    uint32_t major;
    uint32_t minor;
    uint32_t patch;
};

std::ostream &operator<<(std::ostream &out, const SemanticVersion &v);

/*
 * Hardware Identifier
 * @versionStatus:             Version status
 * @version:                   Version revision
 * @product:                   Product revision
 * @architecture:              Architecture revison
 */
struct HardwareId {
public:
    HardwareId(uint32_t _versionStatus,
               const SemanticVersion &_version,
               const SemanticVersion &_product,
               const SemanticVersion &_architecture) :
        versionStatus(_versionStatus),
        version(_version), product(_product), architecture(_architecture) {}

    uint32_t versionStatus;
    SemanticVersion version;
    SemanticVersion product;
    SemanticVersion architecture;
};

/*
 * Hardware Configuration
 * @macsPerClockCycle:         MACs per clock cycle
 * @cmdStreamVersion:          NPU command stream version
 * @customDma:                 Custom DMA enabled
 */
struct HardwareConfiguration {
public:
    HardwareConfiguration(uint32_t _macsPerClockCycle, uint32_t _cmdStreamVersion, bool _customDma) :
        macsPerClockCycle(_macsPerClockCycle), cmdStreamVersion(_cmdStreamVersion), customDma(_customDma) {}

    uint32_t macsPerClockCycle;
    uint32_t cmdStreamVersion;
    bool customDma;
};

/**
 * Device capabilities
 * @hwId:                      Hardware
 * @driver:                    Driver revision
 * @hwCfg                      Hardware configuration
 */
class Capabilities {
public:
    Capabilities(const HardwareId &_hwId, const HardwareConfiguration &_hwCfg, const SemanticVersion &_driver) :
        hwId(_hwId), hwCfg(_hwCfg), driver(_driver) {}

    HardwareId hwId;
    HardwareConfiguration hwCfg;
    SemanticVersion driver;
};

class Device {
public:
    Device(const char *device = "/dev/ethosu0");
    virtual ~Device();

    int ioctl(unsigned long cmd, void *data = nullptr) const;
    Capabilities capabilities() const;

private:
    int fd;
};

class Buffer {
public:
    Buffer(const Device &device, const size_t capacity);
    virtual ~Buffer();

    size_t capacity() const;
    void clear() const;
    char *data() const;
    void resize(size_t size, size_t offset = 0) const;
    size_t offset() const;
    size_t size() const;

    int getFd() const;

private:
    int fd;
    char *dataPtr;
    const size_t dataCapacity;
};

class Network {
public:
    Network(const Device &device, std::shared_ptr<Buffer> &buffer);
    virtual ~Network();

    int ioctl(unsigned long cmd, void *data = nullptr);
    std::shared_ptr<Buffer> getBuffer();

private:
    int fd;
    std::shared_ptr<Buffer> buffer;
};

class Inference {
public:
    template <typename T>
    Inference(const std::shared_ptr<Network> &network,
              const T &ifmBegin,
              const T &ifmEnd,
              const T &ofmBegin,
              const T &ofmEnd) :
        network(network) {
        std::copy(ifmBegin, ifmEnd, std::back_inserter(ifmBuffers));
        std::copy(ofmBegin, ofmEnd, std::back_inserter(ofmBuffers));
        std::vector<uint32_t> counterConfigs = initializeCounterConfig();

        create(counterConfigs, false);
    }
    template <typename T, typename U>
    Inference(const std::shared_ptr<Network> &network,
              const T &ifmBegin,
              const T &ifmEnd,
              const T &ofmBegin,
              const T &ofmEnd,
              const U &counters,
              bool enableCycleCounter) :
        network(network) {
        std::copy(ifmBegin, ifmEnd, std::back_inserter(ifmBuffers));
        std::copy(ofmBegin, ofmEnd, std::back_inserter(ofmBuffers));
        std::vector<uint32_t> counterConfigs = initializeCounterConfig();

        if (counters.size() > counterConfigs.size())
            throw EthosU::Exception("PMU Counters argument to large.");

        std::copy(counters.begin(), counters.end(), counterConfigs.begin());
        create(counterConfigs, enableCycleCounter);
    }

    virtual ~Inference();

    int wait(int64_t timeoutNanos = -1) const;
    const std::vector<uint32_t> getPmuCounters() const;
    uint64_t getCycleCounter() const;
    bool failed() const;
    int getFd() const;
    const std::shared_ptr<Network> getNetwork() const;
    std::vector<std::shared_ptr<Buffer>> &getIfmBuffers();
    std::vector<std::shared_ptr<Buffer>> &getOfmBuffers();

    static uint32_t getMaxPmuEventCounters();

private:
    void create(std::vector<uint32_t> &counterConfigs, bool enableCycleCounter);
    std::vector<uint32_t> initializeCounterConfig();

    int fd;
    const std::shared_ptr<Network> network;
    std::vector<std::shared_ptr<Buffer>> ifmBuffers;
    std::vector<std::shared_ptr<Buffer>> ofmBuffers;
};

} // namespace EthosU
