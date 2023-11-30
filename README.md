# fym1

# Weekly Plan(11.21)

1.看完powerbev（Approach部分） 已完成
2.跑通源代码

报错： raise RuntimeError("No rendezvous handler for {}://".format(result.scheme)) RuntimeError: No rendezvous handler for env:/ 已解决 改torch版本

报错：Distributed package doesn‘t have NCCL built in 已解决 Windows系统改成gloo

报错：OSError: [WinError 123] 文件名、目录名或卷标语法不正确。: 'tensorboard_logs\\26November2023at12:42:59ÖÐ¹ú±ê×¼Ê±¼ä_LAPTOP-3D1NFAER_powerbev' 已解决！ 修改命名格式

报错：AttributeError: 'OSError' object has no attribute 'message' 已解决

报错：BrokenPipeError: [Errno 32] Broken pipe  已解决 

报错：RuntimeError: CUDA error: operation not supported when calling `cusparseCreate(handle)` 已解决 重装网上建议版本cuda11.7 torch1.30


报错：torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 28.00 MiB (GPU 0; 8.00 GiB total capacity; 7.12 GiB already allocated; 0 bytes free; 7.23 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF  改小batchsize

# Weekly Plan(11.30)

看懂powerbev代码
