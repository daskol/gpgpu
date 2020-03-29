/**
 * \file memory.cuh
 */

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <utility>

#include <cuda.h>

namespace my {

enum MemoryType : uint8_t {
    kHost = 0,
    kDevice,
};

enum MemoryKind : uint8_t {
    kDefault = 0,
    kPinned,
};

enum SyncDir {
    kToDevice,
    kToHost,
};

template <typename T>
class host_ptr;

/**
 * Provide partial specialization only for array types.
 */
template <typename T>
class host_ptr<T[]> {
public:
    using cnt_type = uint32_t;
    using ptr_type = T*;
    using val_type = T;

private:
    class counted_ptr {
    private:
    public:
        ptr_type ptr = ptr_type();
        cnt_type cnt = cnt_type();

    public:
        constexpr counted_ptr(void) = default;

        constexpr counted_ptr(ptr_type ptr) noexcept
            : ptr{ptr}
            , cnt{1}
        {}

        ~counted_ptr(void) {
            release();
        }

        explicit operator bool(void) noexcept {
            return !cnt;
        }

        void acquire(void) noexcept {
            ++cnt;
        }

        ptr_type get(void) const noexcept {
            return ptr;
        }

        void release(void) noexcept {
            if (cnt > 0) {
                --cnt;
                if (cnt == 0 && ptr != ptr_type()) {
                    delete[] ptr;
                }
            }
        }
    };

public:
    host_ptr(void)
        : ptr{new counted_ptr}
    {}

    host_ptr(std::nullptr_t)
        : host_ptr()
    {}

    explicit host_ptr(ptr_type ptr)
    {
        try {
            this->ptr = new counted_ptr(ptr);
        } catch (...) {
            delete this->ptr;
            throw;
        }
    }

    host_ptr(host_ptr<val_type[]> const &that) noexcept
        : ptr{that.ptr}
    {
        ptr->acquire();
    }

    host_ptr(host_ptr &&that) noexcept
        : ptr{std::move(that.ptr)}
    {}

    ~host_ptr(void) {
        ptr->release();
    }

    explicit operator bool(void) const noexcept {
        return static_cast<bool>(*ptr);
    }

    host_ptr &operator=(host_ptr const &rhs) noexcept {
        ptr->release();
        ptr = rhs.ptr;
        ptr->acquire();
        return *this;
    }

    host_ptr &operator=(host_ptr &&rhs) noexcept {
        ptr = std::move(rhs.ptr);
        return *this;
    }

    ptr_type get(void) const noexcept {
        return ptr->get();
    }

    void release(void) noexcept {
        ptr->release();
    }

    std::add_lvalue_reference_t<val_type> operator*(void) const {
        return *ptr->get();
    }

    std::add_lvalue_reference_t<val_type> operator[](size_t index) const {
        return ptr->get()[index];
    }

    ptr_type operator->(void) const {
        return get();
    }

    cnt_type count(void) const noexcept {
        return ptr->cnt;
    }

private:
    counted_ptr *ptr;
};

template <typename T>
inline host_ptr<T> make_host(size_t size) {
    return host_ptr<T>(new std::remove_extent_t<T>[size]());
}

template <typename T>
class cuda_ptr;

/**
 * Template specialization of CUDA-pointer for arrays.
 */
template <typename T>
class cuda_ptr<T[]> {
public:
    using cnt_type = uint32_t;
    using ptr_type = T*;
    using val_type = T;
    using mem_type = MemoryKind;

private:
    class counted_ptr {
    private:
    public:
        ptr_type host = ptr_type();
        ptr_type dev = ptr_type();
        cnt_type cnt = cnt_type();
        mem_type mem = MemoryKind::kDefault;

    public:
        constexpr counted_ptr(void) = default;

        constexpr counted_ptr(ptr_type host, ptr_type dev, MemoryKind kind) noexcept
            : host{host}
            , dev{dev}
            , cnt{1}
            , mem{kind}
        {}

        ~counted_ptr(void) {
            release();
        }

        explicit operator bool(void) noexcept {
            return !cnt;
        }

        void acquire(void) noexcept {
            ++cnt;
        }

        cnt_type count(void) const noexcept {
            return cnt;
        }

        ptr_type get(MemoryType type) const noexcept {
            switch (type) {
            case MemoryType::kHost:
                return host;
            case MemoryType::kDevice:
                return dev;
            default:
                return nullptr;
            }
        }

        void release(void) noexcept {
            if (cnt > 0) {
                --cnt;
            }

            if (cnt != 0) {
                return;
            }

            if (dev != ptr_type()) {
                cudaFree(reinterpret_cast<void *>(dev));
            }

            if (host != ptr_type()) {
                switch (mem) {
                case MemoryKind::kDefault:
                    delete[] host;
                    break;
                case MemoryKind::kPinned:
                    cudaFreeHost(host);
                    break;
                }
            }

            dev = nullptr;
            host = nullptr;
        }
    };

public:
    cuda_ptr(void)
        : ptr{new counted_ptr}
    {}

    cuda_ptr(std::nullptr_t)
        : cuda_ptr()
    {}

    explicit cuda_ptr(
        ptr_type host,
        ptr_type dev,
        MemoryKind mem = MemoryKind::kDefault
    ) {
        try {
            ptr = new counted_ptr(host, dev, mem);
        } catch (...) {
            // TODO(@daskol): remove allocated memory.
            throw;
        }
    }

    cuda_ptr(cuda_ptr<val_type[]> const &that) noexcept
        : ptr{that.ptr}
    {
        ptr->acquire();
    }

    cuda_ptr(cuda_ptr &&that) noexcept
        : ptr{std::move(that.ptr)}
    {}

    ~cuda_ptr(void) {
        release();
    }

    explicit operator bool(void) const noexcept {
        return static_cast<bool>(*ptr);
    }

    cuda_ptr &operator=(cuda_ptr const &rhs) noexcept {
        ptr->release();
        ptr = rhs.ptr;
        ptr->acquire();
        return *this;
    }

    cuda_ptr &operator=(cuda_ptr &&rhs) noexcept {
        std::swap(ptr, rhs.ptr);
        return *this;
    }

    ptr_type get(void) const noexcept {
        return get(MemoryType::kHost);
    }

    ptr_type get(MemoryType type) const noexcept {
        return ptr->get(type);
    }

    void release(void) noexcept {
        if (ptr) {
            ptr->release();
            if (!ptr->count()) {
                delete ptr;
                ptr = nullptr;
            }
        }
    }

    std::add_lvalue_reference_t<val_type> operator*(void) const {
        return *ptr->get(MemoryType::kHost);
    }

    std::add_lvalue_reference_t<val_type> operator[](size_t index) const {
        return ptr->get(MemoryType::kHost)[index];
    }

    ptr_type operator->(void) const {
        return get(MemoryType::kHost);
    }

    cnt_type count(void) const noexcept {
        return ptr->cnt;
    }

    cuda_ptr &sync(
        size_t size,
        SyncDir dir, cudaStream_t stream = 0
    ) noexcept {
        return sync(0, size, dir, stream);
    }

    cuda_ptr &sync(
        size_t offset, size_t size,
        SyncDir dir, cudaStream_t stream = 0
    ) noexcept {
        ptr_type src, dst;
        size_t length = size * sizeof(val_type);
        switch (dir) {
        case SyncDir::kToDevice:
            src = get(MemoryType::kHost) + offset;
            dst = get(MemoryType::kDevice) + offset;
            cudaMemcpyAsync(dst, src, length, cudaMemcpyHostToDevice, stream);
        case SyncDir::kToHost:
            src = get(MemoryType::kDevice) + offset;
            dst = get(MemoryType::kHost) + offset;
            cudaMemcpyAsync(dst, src, length, cudaMemcpyDeviceToHost, stream);
        }
        return *this;
    }

private:
    counted_ptr *ptr;
};

template <typename T>
inline cuda_ptr<T> make_cuda(
    size_t size,
    MemoryKind kind = MemoryKind::kDefault
) {
    using val_type = std::remove_extent_t<T>;
    using ptr_type = val_type *;

    cudaError_t err;
    size_t size_elems = size, size_bytes = size * sizeof(val_type);
    ptr_type dev = nullptr, host = nullptr;

    err = cudaMalloc(reinterpret_cast<void **>(&dev), size_bytes);
    if (err != cudaSuccess) {
        return {};
    }

    switch (kind) {
    case MemoryKind::kDefault:
        host = new val_type[size_elems]();
        break;
    case MemoryKind::kPinned:
        err = cudaMallocHost(reinterpret_cast<void **>(&host), size_bytes);
        if (err != cudaSuccess) {
            cudaFree(reinterpret_cast<void *>(dev));
            return {};
        }
        break;
    default:
        return {};
    }

    return cuda_ptr<T>(host, dev, kind);
}

} // namespace my
