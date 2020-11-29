#include "NvInfer.h"
#include "half.h"
#include <cassert>
#include <cuda_runtime_api.h>
#include <iostream>
#include <iterator>
#include <memory>
#include <new>
#include <numeric>
#include <string>
#include <vector>

inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kINT8: return 1;
    }
    throw std::runtime_error("Invalid DataType.");
}

template <typename A, typename B>
inline A divUp(A x, B n)
{
    return (x + n - 1) / n;
}

inline int64_t volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

template <typename AllocFunc, typename FreeFunc>
class GenericBuffer
{
public:
    GenericBuffer(nvinfer1::DataType type = nvinfer1::DataType::kFLOAT)
        : mSize(0)
        , mCapacity(0)
        , mType(type)
        , mBuffer(nullptr)
    {
    }

    GenericBuffer(size_t size, nvinfer1::DataType type)
        : mSize(size)
        , mCapacity(size)
        , mType(type)
    {
        if (!allocFn(&mBuffer, this->nbBytes()))
        {
            throw std::bad_alloc();
        }
    }

    GenericBuffer(GenericBuffer&& buf)
        : mSize(buf.mSize)
        , mCapacity(buf.mCapacity)
        , mType(buf.mType)
        , mBuffer(buf.mBuffer)
    {
        buf.mSize = 0;
        buf.mCapacity = 0;
        buf.mType = nvinfer1::DataType::kFLOAT;
        buf.mBuffer = nullptr;
    }

    GenericBuffer& operator=(GenericBuffer&& buf)
    {
        if (this != &buf)
        {
            freeFn(mBuffer);
            mSize = buf.mSize;
            mCapacity = buf.mCapacity;
            mType = buf.mType;
            mBuffer = buf.mBuffer;
            // Reset buf.
            buf.mSize = 0;
            buf.mCapacity = 0;
            buf.mBuffer = nullptr;
        }
        return *this;
    }

    void* data()
    {
        return mBuffer;
    }

    const void* data() const
    {
        return mBuffer;
    }

    size_t size() const
    {
        return mSize;
    }

    size_t nbBytes() const
    {
        return this->size() * getElementSize(mType);
    }


    void resize(size_t newSize)
    {
        mSize = newSize;
        if (mCapacity < newSize)
        {
            freeFn(mBuffer);
            if (!allocFn(&mBuffer, this->nbBytes()))
            {
                throw std::bad_alloc{};
            }
            mCapacity = newSize;
        }
    }


    void resize(const nvinfer1::Dims& dims)
    {
        return this->resize(volume(dims));
    }

    ~GenericBuffer()
    {
        freeFn(mBuffer);
    }

private:
    size_t mSize{0}, mCapacity{0};
    nvinfer1::DataType mType;
    void* mBuffer;
    AllocFunc allocFn;
    FreeFunc freeFn;
};

class DeviceAllocator
{
public:
    bool operator()(void** ptr, size_t size) const
    {
        return cudaMalloc(ptr, size) == cudaSuccess;
    }
};

class DeviceFree
{
public:
    void operator()(void* ptr) const
    {
        cudaFree(ptr);
    }
};

class HostAllocator
{
public:
    bool operator()(void** ptr, size_t size) const
    {
        *ptr = malloc(size);
        return *ptr != nullptr;
    }
};

class HostFree
{
public:
    void operator()(void* ptr) const
    {
        free(ptr);
    }
};

using DeviceBuffer = GenericBuffer<DeviceAllocator, DeviceFree>;
using HostBuffer = GenericBuffer<HostAllocator, HostFree>;

class ManagedBuffer
{
public:
    DeviceBuffer deviceBuffer;
    HostBuffer hostBuffer;
};

class BufferManager
{
public:
    static const size_t kINVALID_SIZE_VALUE = ~size_t(0);

    BufferManager(std::shared_ptr<nvinfer1::ICudaEngine> engine, const int& batchSize,
        const nvinfer1::IExecutionContext* context = nullptr)
        : mEngine(engine)
        , mBatchSize(batchSize)
    {
        for (int i = 0; i < mEngine->getNbBindings(); i++)
        {
            auto dims = context ? context->getBindingDimensions(i) : mEngine->getBindingDimensions(i);
            size_t vol = context ? 1 : static_cast<size_t>(mBatchSize);
            nvinfer1::DataType type = mEngine->getBindingDataType(i);
            int vecDim = mEngine->getBindingVectorizedDim(i);
            if (-1 != vecDim) // i.e., 0 != lgScalarsPerVector
            {
                int scalarsPerVec = mEngine->getBindingComponentsPerElement(i);
                dims.d[vecDim] = divUp(dims.d[vecDim], scalarsPerVec);
                vol *= scalarsPerVec;
            }
            vol *= volume(dims);
            std::unique_ptr<ManagedBuffer> manBuf{new ManagedBuffer()};
            manBuf->deviceBuffer = DeviceBuffer(vol, type);
            manBuf->hostBuffer = HostBuffer(vol, type);
            mDeviceBindings.emplace_back(manBuf->deviceBuffer.data());
            mManagedBuffers.emplace_back(std::move(manBuf));
        }
    }

    std::vector<void*>& getDeviceBindings()
    {
        return mDeviceBindings;
    }

    const std::vector<void*>& getDeviceBindings() const
    {
        return mDeviceBindings;
    }

    void* getDeviceBuffer(const std::string& tensorName) const
    {
        return getBuffer(false, tensorName);
    }


    void* getHostBuffer(const std::string& tensorName) const
    {
        return getBuffer(true, tensorName);
    }

    size_t size(const std::string& tensorName) const
    {
        int index = mEngine->getBindingIndex(tensorName.c_str());
        if (index == -1)
            return kINVALID_SIZE_VALUE;
        return mManagedBuffers[index]->hostBuffer.nbBytes();
    }

    void dumpBuffer(std::ostream& os, const std::string& tensorName)
    {
        int index = mEngine->getBindingIndex(tensorName.c_str());
        if (index == -1)
        {
            os << "Invalid tensor name" << std::endl;
            return;
        }
        void* buf = mManagedBuffers[index]->hostBuffer.data();
        size_t bufSize = mManagedBuffers[index]->hostBuffer.nbBytes();
        nvinfer1::Dims bufDims = mEngine->getBindingDimensions(index);
        size_t rowCount = static_cast<size_t>(bufDims.nbDims >= 1 ? bufDims.d[bufDims.nbDims - 1] : mBatchSize);

        os << "[" << mBatchSize;
        for (int i = 0; i < bufDims.nbDims; i++)
            os << ", " << bufDims.d[i];
        os << "]" << std::endl;
        switch (mEngine->getBindingDataType(index))
        {
        case nvinfer1::DataType::kINT32: print<int32_t>(os, buf, bufSize, rowCount); break;
        case nvinfer1::DataType::kFLOAT: print<float>(os, buf, bufSize, rowCount); break;
        case nvinfer1::DataType::kHALF: print<half_float::half>(os, buf, bufSize, rowCount); break;
        case nvinfer1::DataType::kINT8: assert(0 && "Int8 network-level input and output is not supported"); break;
        }
    }

    template <typename T>
    void print(std::ostream& os, void* buf, size_t bufSize, size_t rowCount)
    {
        assert(rowCount != 0);
        assert(bufSize % sizeof(T) == 0);
        T* typedBuf = static_cast<T*>(buf);
        size_t numItems = bufSize / sizeof(T);
        for (int i = 0; i < static_cast<int>(numItems); i++)
        {
            if (rowCount == 1 && i != static_cast<int>(numItems) - 1)
                os << typedBuf[i] << std::endl;
            else if (rowCount == 1)
                os << typedBuf[i];
            else if (i % rowCount == 0)
                os << typedBuf[i];
            else if (i % rowCount == rowCount - 1)
                os << " " << typedBuf[i] << std::endl;
            else
                os << " " << typedBuf[i];
        }
    }

    void copyInputToDevice()
    {
        memcpyBuffers(true, false, false);
    }

    void copyOutputToHost()
    {
        memcpyBuffers(false, true, false);
    }

    void copyInputToDeviceAsync(const cudaStream_t& stream = 0)
    {
        memcpyBuffers(true, false, true, stream);
    }

    void copyOutputToHostAsync(const cudaStream_t& stream = 0)
    {
        memcpyBuffers(false, true, true, stream);
    }

    ~BufferManager() = default;

private:
    void* getBuffer(const bool isHost, const std::string& tensorName) const
    {
        int index = mEngine->getBindingIndex(tensorName.c_str());
        if (index == -1)
            return nullptr;
        return (isHost ? mManagedBuffers[index]->hostBuffer.data() : mManagedBuffers[index]->deviceBuffer.data());
    }

    void memcpyBuffers(const bool copyInput, const bool deviceToHost, const bool async, const cudaStream_t& stream = 0)
    {
        for (int i = 0; i < mEngine->getNbBindings(); i++)
        {
            void* dstPtr
                = deviceToHost ? mManagedBuffers[i]->hostBuffer.data() : mManagedBuffers[i]->deviceBuffer.data();
            const void* srcPtr
                = deviceToHost ? mManagedBuffers[i]->deviceBuffer.data() : mManagedBuffers[i]->hostBuffer.data();
            const size_t byteSize = mManagedBuffers[i]->hostBuffer.nbBytes();
            const cudaMemcpyKind memcpyType = deviceToHost ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
            if ((copyInput && mEngine->bindingIsInput(i)) || (!copyInput && !mEngine->bindingIsInput(i)))
            {
                if (async)
                    cudaMemcpyAsync(dstPtr, srcPtr, byteSize, memcpyType, stream);
                else
                    cudaMemcpy(dstPtr, srcPtr, byteSize, memcpyType);
            }
        }
    }

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    int mBatchSize;
    std::vector<std::unique_ptr<ManagedBuffer>> mManagedBuffers;
    std::vector<void*> mDeviceBindings;
};
