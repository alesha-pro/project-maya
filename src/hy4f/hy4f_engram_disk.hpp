// Exact Engram row reads from a GGUF on a direct-I/O filesystem.
// CUDA host nodes reuse these allocations during graph replay. All CUDA calls
// are made by the submitting thread, never by the I/O callback.
#pragma once
#include <linux/aio_abi.h>
#include <sys/syscall.h>
#include <fcntl.h>
#include <unistd.h>
#include <array>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <numeric>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <vector>
#include <cuda_runtime.h>

namespace strata::hy4f {
class EngramDisk {
    static constexpr unsigned Depth = 64;
    static constexpr unsigned Sector = 512;
    static constexpr unsigned Width = 96;
    static constexpr unsigned CacheRows = 1u << 20;
    struct Miss { int64_t row; unsigned dst; };
    struct Group { unsigned first, end, length; uint64_t offset; };
    struct HostCall { EngramDisk* disk; unsigned lane, tokens; };
    struct Lane {
        int64_t* ids = nullptr;
        uint8_t* packed = nullptr;
        uint8_t* device = nullptr;
        std::vector<HostCall> calls;
    };
    int fd_ = -1;
    aio_context_t ctx_ = 0;
    uint8_t* io_ = nullptr;
    int device_ = -1;
    uint64_t offset_, rows_;
    unsigned cols_;
    std::array<Lane,2> lanes_;
    int64_t* index_ = nullptr;
    std::vector<int64_t> keys_;
    std::vector<std::array<uint8_t,Width>> cache_;
    std::vector<Miss> missing_;
    std::mutex lock_;
    std::atomic<bool> failed_{false};
    char error_[256]{};
    uint64_t queries_ = 0, misses_ = 0, requests_ = 0, bytes_ = 0;

    static void ck(cudaError_t e) {
        if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
    }
    static void syserr(const char* name) {
        throw std::system_error(errno,std::generic_category(),name);
    }
    static void CUDART_CB callback(void* data) noexcept {
        auto& c = *static_cast<HostCall*>(data);
        auto& d = *c.disk;
        std::lock_guard<std::mutex> lock(d.lock_);
        if (d.failed_.load()) return;
        try { d.fetch(d.lanes_[c.lane], c.tokens*d.cols_); }
        catch (const std::exception& e) {
            std::snprintf(d.error_,sizeof(d.error_),"%s",e.what());
            d.failed_.store(true);
        }
    }
    void read_groups(const Group* groups, unsigned n) {
        std::array<iocb,Depth> ops{};
        std::array<iocb*,Depth> ptrs{};
        std::array<io_event,Depth> events{};
        for (unsigned i=0; i<n; ++i) {
            auto& op=ops[i];
            op.aio_data=i; op.aio_fildes=fd_; op.aio_lio_opcode=IOCB_CMD_PREAD;
            op.aio_offset=groups[i].offset; op.aio_nbytes=groups[i].length;
            op.aio_buf=reinterpret_cast<uintptr_t>(io_+i*2*Sector);
            ptrs[i]=&op;
        }
        unsigned done=0;
        while (done<n) {
            long submitted;
            do { submitted=syscall(SYS_io_submit,ctx_,n-done,ptrs.data()+done); } while (submitted<0 && errno==EINTR);
            if (submitted<0) syserr("Engram io_submit");
            if (!submitted) throw std::runtime_error("Engram io_submit returned zero");
            long collected=0;
            while (collected<submitted) {
                long got=syscall(SYS_io_getevents,ctx_,1,submitted-collected,events.data()+collected,nullptr);
                if (got<0 && errno==EINTR) continue;
                if (got<0) syserr("Engram io_getevents");
                collected+=got;
            }
            for (long j=0; j<submitted; ++j) {
                const auto& ev=events[j];
                if (ev.res<0) throw std::system_error(-ev.res,std::generic_category(),"Engram direct read");
                const auto& g=groups[ev.data];
                const uint64_t needed=offset_+uint64_t(missing_[g.end-1].row)*Width+Width-g.offset;
                if (ev.res2 || uint64_t(ev.res)<needed) throw std::runtime_error("Engram short direct read");
                bytes_+=ev.res;
            }
            done+=submitted;
        }
        requests_+=n;
    }
    void fetch(Lane& lane, unsigned count) {
        missing_.clear();
        queries_+=count;
        for (unsigned i=0; i<count; ++i) {
            const int64_t row=lane.ids[i];
            if (uint64_t(row)>=rows_) throw std::runtime_error("Engram row outside GGUF tensor");
            const auto slot=uint64_t(row)&(CacheRows-1);
            if (keys_[slot]==row) std::memcpy(lane.packed+i*Width,cache_[slot].data(),Width);
            else missing_.push_back({row,i});
        }
        if (missing_.empty()) return;
        std::sort(missing_.begin(),missing_.end(),[](const Miss& a,const Miss& b){return a.row<b.row;});
        unsigned pos=0;
        while (pos<missing_.size()) {
            std::array<Group,Depth> groups;
            unsigned n=0;
            while (n<Depth && pos<missing_.size()) {
                const unsigned first=pos++;
                const uint64_t off=(offset_+uint64_t(missing_[first].row)*Width) & ~uint64_t(Sector-1);
                while (pos<missing_.size() && offset_+uint64_t(missing_[pos].row)*Width<off+Sector) ++pos;
                const uint64_t end=offset_+uint64_t(missing_[pos-1].row)*Width+Width;
                groups[n++]={first,pos,unsigned((end-off+Sector-1)&~uint64_t(Sector-1)),off};
            }
            read_groups(groups.data(),n);
            for (unsigned i=0; i<n; ++i) {
                const auto& g=groups[i];
                for (unsigned j=g.first; j<g.end; ++j) {
                    const auto m=missing_[j]; const size_t slot=uint64_t(m.row)&(CacheRows-1);
                    const uint8_t* src=io_+i*2*Sector+(offset_+uint64_t(m.row)*Width-g.offset);
                    std::memcpy(lane.packed+m.dst*Width,src,Width);
                    if (j==g.first || m.row!=missing_[j-1].row) {
                        std::memcpy(cache_[slot].data(),src,Width); keys_[slot]=m.row; ++misses_;
                    }
                }
            }
        }
    }
    void release() noexcept {
        if (device_>=0) cudaSetDevice(device_);
        for (auto& l:lanes_) {
            if (l.device) cudaFree(l.device);
            if (l.ids) cudaFreeHost(l.ids);
            if (l.packed) cudaFreeHost(l.packed);
        }
        if (index_) cudaFree(index_);
        if (ctx_) syscall(SYS_io_destroy,ctx_);
        std::free(io_);
        if (fd_>=0) close(fd_);
    }
public:
    static constexpr unsigned Batch = 2048;
    EngramDisk(const std::string& path,uint64_t offset,uint64_t rows,unsigned width,unsigned cols,int dev)
        : offset_(offset),rows_(rows),cols_(cols),keys_(CacheRows,-1),cache_(CacheRows) {
        if (width!=Width || !cols || cols>32) throw std::runtime_error("unsupported Engram row geometry");
        try {
            fd_=open(path.c_str(),O_RDONLY|O_CLOEXEC|O_DIRECT);
            if (fd_<0) syserr("Engram open O_DIRECT");
            if (syscall(SYS_io_setup,Depth,&ctx_)<0) syserr("Engram io_setup");
            if (posix_memalign(reinterpret_cast<void**>(&io_),4096,Depth*2*Sector)!=0) throw std::bad_alloc();
            ck(cudaSetDevice(dev)); device_=dev;
            const size_t cap=Batch*cols;
            missing_.reserve(cap);
            std::vector<int64_t> seq(cap); std::iota(seq.begin(),seq.end(),int64_t(0));
            ck(cudaMalloc(&index_,cap*sizeof(int64_t)));
            ck(cudaMemcpy(index_,seq.data(),cap*sizeof(int64_t),cudaMemcpyHostToDevice));
            for (unsigned i=0;i<lanes_.size();++i) {
                auto& l=lanes_[i];
                const unsigned limit=i?8:Batch;
                const size_t maxrows=limit*cols;
                ck(cudaMallocHost(&l.ids,maxrows*sizeof(int64_t)));
                ck(cudaMallocHost(&l.packed,maxrows*Width));
                std::memset(l.packed,0,maxrows*Width);
                ck(cudaMalloc(&l.device,maxrows*Width));
                for (unsigned t=0;t<=limit;++t) l.calls.push_back({this,i,t});
            }
        } catch (...) { release(); throw; }
        std::fprintf(stderr,"hy4f: Engram O_DIRECT %llu rows, %u bytes/row, %u cache rows, QD %u\n",
                     (unsigned long long)rows_,Width,CacheRows,Depth);
    }
    ~EngramDisk() {
        if (device_>=0) { cudaSetDevice(device_); cudaDeviceSynchronize(); }
        std::fprintf(stderr,"hy4f: Engram O_DIRECT stats queries=%llu unique_misses=%llu ios=%llu bytes=%llu\n",
                     (unsigned long long)queries_,(unsigned long long)misses_,(unsigned long long)requests_,(unsigned long long)bytes_);
        release();
    }
    void check() const { if (failed_.load()) throw std::runtime_error(error_); }
    const int64_t* indices() const { return index_; }
    const uint8_t* stage(const int64_t* ids,unsigned t,unsigned lane,cudaStream_t st) {
        check();
        auto& l=lanes_.at(lane);
        if (t>=l.calls.size()) throw std::runtime_error("Engram chunk exceeds lane capacity");
        ck(cudaMemcpyAsync(l.ids,ids,t*cols_*sizeof(int64_t),cudaMemcpyDeviceToHost,st));
        ck(cudaLaunchHostFunc(st,callback,&l.calls[t]));
        ck(cudaMemcpyAsync(l.device,l.packed,t*cols_*Width,cudaMemcpyHostToDevice,st));
        return l.device;
    }
};
}  // namespace strata::hy4f
