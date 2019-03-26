NAME="Red Hat Enterprise Linux Server"
VERSION="7.6 (Maipo)"
ID="rhel"
ID_LIKE="fedora"
VARIANT="Server"
VARIANT_ID="server"
VERSION_ID="7.6"
PRETTY_NAME="Red Hat Enterprise Linux"
ANSI_COLOR="0;31"
CPE_NAME="cpe:/o:redhat:enterprise_linux:7.6:GA:server"
HOME_URL="https://www.redhat.com/"
BUG_REPORT_URL="https://bugzilla.redhat.com/"

REDHAT_BUGZILLA_PRODUCT="Red Hat Enterprise Linux 7"
REDHAT_BUGZILLA_PRODUCT_VERSION=7.6
REDHAT_SUPPORT_PRODUCT="Red Hat Enterprise Linux"
REDHAT_SUPPORT_PRODUCT_VERSION="7.6"
Red Hat Enterprise Linux Server release 7.6 (Maipo)
/fslhome/tarch/anaconda3/envs/munit/bin/python
{'image_save_iter': 2000, 'image_display_iter': 500, 'display_size': 16, 'snapshot_save_iter': 2000, 'log_iter': 10, 'max_iter': 1000000, 'batch_size': 2, 'weight_decay': 0.0001, 'beta1': 0.5, 'beta2': 0.999, 'init': 'kaiming', 'lr': 0.0001, 'lr_policy': 'step', 'step_size': 100000, 'gamma': 0.5, 'gan_w': 1, 'recon_x_w': 10, 'recon_s_w': 1, 'recon_c_w': 1, 'recon_x_cyc_w': 10, 'vgg_w': 0, 'gen': {'dim': 64, 'mlp_dim': 256, 'style_dim': 8, 'activ': 'relu', 'n_downsample': 2, 'n_res': 4, 'pad_type': 'reflect'}, 'dis': {'dim': 64, 'norm': 'none', 'activ': 'lrelu', 'n_layer': 4, 'gan_type': 'lsgan', 'num_scales': 3, 'pad_type': 'reflect'}, 'input_dim_a': 1, 'input_dim_b': 1, 'num_workers': 8, 'new_size': 64, 'crop_image_height': 64, 'crop_image_width': 1280, 'horizontal_flip': False, 'data_folder_train_a': './datasets/handwriting/train_online', 'data_list_train_a': './datasets/handwriting/train_online.txt', 'data_folder_test_a': './datasets/handwriting/test_online', 'data_list_test_a': './datasets/handwriting/test_online.txt', 'data_folder_train_b': './datasets/handwriting/train_offline', 'data_list_train_b': './datasets/handwriting/train_offline.txt', 'data_folder_test_b': './datasets/handwriting/test_offline', 'data_list_test_b': './datasets/handwriting/test_offline.txt'}
tensor([[[ 1.0000,  1.0000,  1.0000,  ...,  1.0000,  1.0000,  1.0000],
         [ 1.0000,  1.0000,  1.0000,  ...,  1.0000,  1.0000,  1.0000],
         [ 1.0000,  1.0000,  1.0000,  ...,  1.0000,  1.0000,  1.0000],
         ...,
         [ 1.0000,  1.0000,  1.0000,  ...,  1.0000,  1.0000,  1.0000],
         [ 1.0000,  1.0000,  1.0000,  ...,  1.0000,  1.0000,  1.0000],
         [ 1.0000,  1.0000,  1.0000,  ...,  1.0000,  1.0000,  1.0000]]])
Creating directory: ./outputs/handwriting_online/01_Run/images
Creating directory: ./outputs/handwriting_online/01_Run/checkpoints
/fslhome/tarch/anaconda3/envs/munit/lib/python3.7/site-packages/torch/nn/modules/upsampling.py:122: UserWarning: nn.Upsampling is deprecated. Use nn.functional.interpolate instead.
  warnings.warn("nn.Upsampling is deprecated. Use nn.functional.interpolate instead.")
Elapsed time in update: 5.260142
Elapsed time in update: 3.085456
Elapsed time in update: 3.084378
Elapsed time in update: 3.084677
Elapsed time in update: 3.084694
Elapsed time in update: 3.083670
Elapsed time in update: 3.084130
Elapsed time in update: 3.083961
Elapsed time in update: 3.083966
Elapsed time in update: 3.084125
Iteration: 00000010/01000000
Elapsed time in update: 3.083761
Elapsed time in update: 3.084118
Elapsed time in update: 3.084364
Elapsed time in update: 3.083751
Elapsed time in update: 3.085404
Elapsed time in update: 3.083903
Elapsed time in update: 3.083774
Exception ignored in: <function _DataLoaderIter.__del__ at 0x7f966c2b22f0>
Traceback (most recent call last):
  File "/fslhome/tarch/anaconda3/envs/munit/lib/python3.7/site-packages/torch/utils/data/dataloader.py", line 399, in __del__
    self._shutdown_workers()
  File "/fslhome/tarch/anaconda3/envs/munit/lib/python3.7/site-packages/torch/utils/data/dataloader.py", line 378, in _shutdown_workers
    self.worker_result_queue.get()
  File "/fslhome/tarch/anaconda3/envs/munit/lib/python3.7/multiprocessing/queues.py", line 354, in get
    return _ForkingPickler.loads(res)
  File "/fslhome/tarch/anaconda3/envs/munit/lib/python3.7/site-packages/torch/multiprocessing/reductions.py", line 151, in rebuild_storage_fd
    fd = df.detach()
  File "/fslhome/tarch/anaconda3/envs/munit/lib/python3.7/multiprocessing/resource_sharer.py", line 57, in detach
    with _resource_sharer.get_connection(self._id) as conn:
  File "/fslhome/tarch/anaconda3/envs/munit/lib/python3.7/multiprocessing/resource_sharer.py", line 87, in get_connection
    c = Client(address, authkey=process.current_process().authkey)
  File "/fslhome/tarch/anaconda3/envs/munit/lib/python3.7/multiprocessing/connection.py", line 498, in Client
    answer_challenge(c, authkey)
  File "/fslhome/tarch/anaconda3/envs/munit/lib/python3.7/multiprocessing/connection.py", line 746, in answer_challenge
    response = connection.recv_bytes(256)        # reject large message
  File "/fslhome/tarch/anaconda3/envs/munit/lib/python3.7/multiprocessing/connection.py", line 216, in recv_bytes
    buf = self._recv_bytes(maxlength)
  File "/fslhome/tarch/anaconda3/envs/munit/lib/python3.7/multiprocessing/connection.py", line 407, in _recv_bytes
    buf = self._recv(4)
  File "/fslhome/tarch/anaconda3/envs/munit/lib/python3.7/multiprocessing/connection.py", line 379, in _recv
    chunk = read(handle, remaining)
ConnectionResetError: [Errno 104] Connection reset by peer
Traceback (most recent call last):
  File "train.py", line 82, in <module>
    for it, (images_a, images_b) in enumerate(zip(train_loader_a, train_loader_b)):
  File "/fslhome/tarch/anaconda3/envs/munit/lib/python3.7/site-packages/torch/utils/data/dataloader.py", line 336, in __next__
    return self._process_next_batch(batch)
  File "/fslhome/tarch/anaconda3/envs/munit/lib/python3.7/site-packages/torch/utils/data/dataloader.py", line 357, in _process_next_batch
    raise batch.exc_type(batch.exc_msg)
ValueError: Traceback (most recent call last):
  File "/fslhome/tarch/anaconda3/envs/munit/lib/python3.7/site-packages/torch/utils/data/dataloader.py", line 106, in _worker_loop
    samples = collate_fn([dataset[i] for i in batch_indices])
  File "/fslhome/tarch/anaconda3/envs/munit/lib/python3.7/site-packages/torch/utils/data/dataloader.py", line 106, in <listcomp>
    samples = collate_fn([dataset[i] for i in batch_indices])
  File "/panfs/pan.fsl.byu.edu/scr/usr/65/tarch/research/handwriting/MUNIT/data.py", line 47, in __getitem__
    img = self.transform(img)
  File "/fslhome/tarch/anaconda3/envs/munit/lib/python3.7/site-packages/torchvision/transforms/transforms.py", line 49, in __call__
    img = t(img)
  File "/fslhome/tarch/anaconda3/envs/munit/lib/python3.7/site-packages/torchvision/transforms/transforms.py", line 421, in __call__
    i, j, h, w = self.get_params(img, self.size)
  File "/fslhome/tarch/anaconda3/envs/munit/lib/python3.7/site-packages/torchvision/transforms/transforms.py", line 400, in get_params
    j = random.randint(0, w - tw)
  File "/fslhome/tarch/anaconda3/envs/munit/lib/python3.7/random.py", line 222, in randint
    return self.randrange(a, b+1)
  File "/fslhome/tarch/anaconda3/envs/munit/lib/python3.7/random.py", line 200, in randrange
    raise ValueError("empty range for randrange() (%d,%d, %d)" % (istart, istop, width))
ValueError: empty range for randrange() (0,-1215, -1215)

Exception ignored in: <function _DataLoaderIter.__del__ at 0x7f966c2b22f0>
Traceback (most recent call last):
  File "/fslhome/tarch/anaconda3/envs/munit/lib/python3.7/site-packages/torch/utils/data/dataloader.py", line 399, in __del__
    self._shutdown_workers()
  File "/fslhome/tarch/anaconda3/envs/munit/lib/python3.7/site-packages/torch/utils/data/dataloader.py", line 378, in _shutdown_workers
    self.worker_result_queue.get()
  File "/fslhome/tarch/anaconda3/envs/munit/lib/python3.7/multiprocessing/queues.py", line 354, in get
    return _ForkingPickler.loads(res)
  File "/fslhome/tarch/anaconda3/envs/munit/lib/python3.7/site-packages/torch/multiprocessing/reductions.py", line 151, in rebuild_storage_fd
    fd = df.detach()
  File "/fslhome/tarch/anaconda3/envs/munit/lib/python3.7/multiprocessing/resource_sharer.py", line 57, in detach
    with _resource_sharer.get_connection(self._id) as conn:
  File "/fslhome/tarch/anaconda3/envs/munit/lib/python3.7/multiprocessing/resource_sharer.py", line 87, in get_connection
    c = Client(address, authkey=process.current_process().authkey)
  File "/fslhome/tarch/anaconda3/envs/munit/lib/python3.7/multiprocessing/connection.py", line 492, in Client
    c = SocketClient(address)
  File "/fslhome/tarch/anaconda3/envs/munit/lib/python3.7/multiprocessing/connection.py", line 619, in SocketClient
    s.connect(address)
ConnectionRefusedError: [Errno 111] Connection refused
