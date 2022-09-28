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

#include <tensorflow/lite/kernels/ethos_u/ethosu.hpp>
#include <ethosu.h>

#include <algorithm>
#include <queue>
#include <exception>
#include <iostream>

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

using namespace std;

namespace EthosU {
__attribute__((weak)) int eioctl(int fd, unsigned long cmd, void *data = nullptr) {
    int ret = ::ioctl(fd, cmd, data);
    if (ret < 0) {
        throw Exception("IOCTL failed");
    }

    return ret;
}

__attribute__((weak)) int eopen(const char *pathname, int flags) {
    int fd = ::open(pathname, flags);
    if (fd < 0) {
        throw Exception("Failed to open device");
    }

    return fd;
}

__attribute__((weak)) int
eppoll(struct pollfd *fds, nfds_t nfds, const struct timespec *tmo_p, const sigset_t *sigmask) {
    int result = ::ppoll(fds, nfds, tmo_p, sigmask);
    if (result < 0) {
        throw Exception("Failed to wait for ppoll event or signal");
    }

    return result;
}

__attribute__((weak)) int eclose(int fd) {
    int result = ::close(fd);
    if (result < 0) {
        throw Exception("Failed to close file");
    }

    return result;
}
__attribute((weak)) void *emmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
    void *ptr = ::mmap(addr, length, prot, flags, fd, offset);
    if (ptr == MAP_FAILED) {
        throw Exception("Failed to mmap file");
    }

    return ptr;
}

__attribute__((weak)) int emunmap(void *addr, size_t length) {
    int result = ::munmap(addr, length);
    if (result < 0) {
        throw Exception("Failed to munmap file");
    }

    return result;
}

} // namespace EthosU

namespace EthosU {

/****************************************************************************
 * Exception
 ****************************************************************************/

Exception::Exception(const char *msg) : msg(msg) {}

Exception::~Exception() throw() {}

const char *Exception::what() const throw() {
    return msg.c_str();
}

/****************************************************************************
 * Semantic Version
 ****************************************************************************/

bool SemanticVersion::operator==(const SemanticVersion &other) {
    return other.major == major && other.minor == minor && other.patch == patch;
}

bool SemanticVersion::operator<(const SemanticVersion &other) {
    if (other.major > major)
        return true;
    if (other.minor > minor)
        return true;
    return other.patch > patch;
}

bool SemanticVersion::operator<=(const SemanticVersion &other) {
    return *this < other || *this == other;
}

bool SemanticVersion::operator!=(const SemanticVersion &other) {
    return !(*this == other);
}

bool SemanticVersion::operator>(const SemanticVersion &other) {
    return !(*this <= other);
}

bool SemanticVersion::operator>=(const SemanticVersion &other) {
    return !(*this < other);
}

ostream &operator<<(ostream &out, const SemanticVersion &v) {
    return out << "{ major=" << unsigned(v.major) << ", minor=" << unsigned(v.minor) << ", patch=" << unsigned(v.patch)
               << " }";
}

/****************************************************************************
 * Device
 ****************************************************************************/
Device::Device(const char *device) {
    fd = eopen(device, O_RDWR | O_NONBLOCK);
    //workaround for kernel call trace when startup
    usleep(10 * 1000);
    this->ioctl(ETHOSU_IOCTL_VERSION_REQ);
}

Device::~Device() {
    eclose(fd);
}

Device* Device::GetSingleton(const char *device){
    static Device dev(device);
    return &dev;
}

int Device::ioctl(unsigned long cmd, void *data) const {
    return eioctl(fd, cmd, data);
}

Capabilities Device::capabilities() const {
    ethosu_uapi_device_capabilities uapi;
    (void)eioctl(fd, ETHOSU_IOCTL_CAPABILITIES_REQ, static_cast<void *>(&uapi));

    Capabilities capabilities(
        HardwareId(uapi.hw_id.version_status,
                   SemanticVersion(uapi.hw_id.version_major, uapi.hw_id.version_minor),
                   SemanticVersion(uapi.hw_id.product_major),
                   SemanticVersion(uapi.hw_id.arch_major_rev, uapi.hw_id.arch_minor_rev, uapi.hw_id.arch_patch_rev)),
        HardwareConfiguration(uapi.hw_cfg.macs_per_cc, uapi.hw_cfg.cmd_stream_version, bool(uapi.hw_cfg.custom_dma)),
        SemanticVersion(uapi.driver_major_rev, uapi.driver_minor_rev, uapi.driver_patch_rev));
    return capabilities;
}

/****************************************************************************
 * Buffer
 ****************************************************************************/

Buffer::Buffer(const Device &device, const size_t capacity) : fd(-1), dataPtr(nullptr), dataCapacity(capacity) {
    ethosu_uapi_buffer_create uapi = {static_cast<uint32_t>(dataCapacity)};
    fd                             = device.ioctl(ETHOSU_IOCTL_BUFFER_CREATE, static_cast<void *>(&uapi));

    void *d;
    try {
        d = emmap(nullptr, dataCapacity, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    } catch (std::exception &e) {
        try {
            eclose(fd);
        } catch (...) { std::throw_with_nested(e); }
    }
    dataPtr = reinterpret_cast<char *>(d);
    this->resize(dataCapacity);
}

Buffer::~Buffer() {
    emunmap(dataPtr, dataCapacity);
    eclose(fd);
}

size_t Buffer::capacity() const {
    return dataCapacity;
}

void Buffer::clear() const {
    resize(0, 0);
}

char *Buffer::data() const {
    return dataPtr + offset();
}

void Buffer::resize(size_t size, size_t offset) const {
    ethosu_uapi_buffer uapi;
    uapi.offset = offset;
    uapi.size   = size;
    eioctl(fd, ETHOSU_IOCTL_BUFFER_SET, static_cast<void *>(&uapi));
}

size_t Buffer::offset() const {
    ethosu_uapi_buffer uapi;
    eioctl(fd, ETHOSU_IOCTL_BUFFER_GET, static_cast<void *>(&uapi));
    return uapi.offset;
}

size_t Buffer::size() const {
    ethosu_uapi_buffer uapi;
    eioctl(fd, ETHOSU_IOCTL_BUFFER_GET, static_cast<void *>(&uapi));
    return uapi.size;
}

int Buffer::getFd() const {
    return fd;
}

/****************************************************************************
 * Network
 ****************************************************************************/

Network::Network(const Device &device, shared_ptr<Buffer> &buffer) : fd(-1), buffer(buffer) {
    // Create buffer handle
    ethosu_uapi_network_create uapi;
    uapi.fd = buffer->getFd();
    fd      = device.ioctl(ETHOSU_IOCTL_NETWORK_CREATE, static_cast<void *>(&uapi));
}

Network::~Network() {
    eclose(fd);
}

int Network::ioctl(unsigned long cmd, void *data) {
    return eioctl(fd, cmd, data);
}

shared_ptr<Buffer> Network::getBuffer() {
    return buffer;
}

/****************************************************************************
 * Inference
 ****************************************************************************/

Inference::~Inference() {
    eclose(fd);
}

void Inference::create(std::vector<uint32_t> &counterConfigs, bool cycleCounterEnable = false) {
    ethosu_uapi_inference_create uapi;

    if (ifmBuffers.size() > ETHOSU_FD_MAX) {
        throw Exception("IFM buffer overflow");
    }

    if (ofmBuffers.size() > ETHOSU_FD_MAX) {
        throw Exception("OFM buffer overflow");
    }

    if (counterConfigs.size() != ETHOSU_PMU_EVENT_MAX) {
        throw Exception("Wrong size of counter configurations");
    }

    uapi.ifm_count = 0;
    for (auto it : ifmBuffers) {
        uapi.ifm_fd[uapi.ifm_count++] = it->getFd();
    }

    uapi.ofm_count = 0;
    for (auto it : ofmBuffers) {
        uapi.ofm_fd[uapi.ofm_count++] = it->getFd();
    }

    for (int i = 0; i < ETHOSU_PMU_EVENT_MAX; i++) {
        uapi.pmu_config.events[i] = counterConfigs[i];
    }

    uapi.pmu_config.cycle_count = cycleCounterEnable;
    uapi.inference_type = ETHOSU_UAPI_INFERENCE_OP;

    fd = network->ioctl(ETHOSU_IOCTL_INFERENCE_CREATE, static_cast<void *>(&uapi));
}

std::vector<uint32_t> Inference::initializeCounterConfig() {
    return std::vector<uint32_t>(ETHOSU_PMU_EVENT_MAX, 0);
}

uint32_t Inference::getMaxPmuEventCounters() {
    return ETHOSU_PMU_EVENT_MAX;
}

int Inference::wait(int64_t timeoutNanos) const {
    struct pollfd pfd;
    pfd.fd      = fd;
    pfd.events  = POLLIN | POLLERR;
    pfd.revents = 0;

    // if timeout negative wait forever
    if (timeoutNanos < 0) {
        return eppoll(&pfd, 1, NULL, NULL);
    }

    struct timespec tmo_p;
    int64_t nanosec = 1000000000;
    tmo_p.tv_sec    = timeoutNanos / nanosec;
    tmo_p.tv_nsec   = timeoutNanos % nanosec;

    return eppoll(&pfd, 1, &tmo_p, NULL);
}

bool Inference::failed() const {
    ethosu_uapi_result_status uapi;

    eioctl(fd, ETHOSU_IOCTL_INFERENCE_STATUS, static_cast<void *>(&uapi));

    return uapi.status != ETHOSU_UAPI_STATUS_OK;
}

const std::vector<uint32_t> Inference::getPmuCounters() const {
    ethosu_uapi_result_status uapi;
    std::vector<uint32_t> counterValues = std::vector<uint32_t>(ETHOSU_PMU_EVENT_MAX, 0);

    eioctl(fd, ETHOSU_IOCTL_INFERENCE_STATUS, static_cast<void *>(&uapi));

    for (int i = 0; i < ETHOSU_PMU_EVENT_MAX; i++) {
        if (uapi.pmu_config.events[i]) {
            counterValues.at(i) = uapi.pmu_count.events[i];
        }
    }

    return counterValues;
}

uint64_t Inference::getCycleCounter() const {
    ethosu_uapi_result_status uapi;

    eioctl(fd, ETHOSU_IOCTL_INFERENCE_STATUS, static_cast<void *>(&uapi));

    return uapi.pmu_count.cycle_count;
}

int Inference::getFd() const {
    return fd;
}

const shared_ptr<Network> Inference::getNetwork() const {
    return network;
}

vector<shared_ptr<Buffer>> &Inference::getIfmBuffers() {
    return ifmBuffers;
}

vector<shared_ptr<Buffer>> &Inference::getOfmBuffers() {
    return ofmBuffers;
}

} // namespace EthosU
