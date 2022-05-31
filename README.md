This is my solution of the test task:
<details>
Обработка данных:

Исходные данные: последовательность из 1 миллиона чисел (int, от 1 до 100) (14, 45, 77, 1, ...)
Сжать данные, чтобы они занимали как можно меньше места
Разжать данные в оригинальную последовательность
Условия:

Должна быть многопоточность
Вычисления должны делаться на GPU
</details>
Task: To compress 1000.000 numbers (from 1 to 100) as much as possible and then to decompress it.
Condition:

- Multithreading
- GPU calculation

Build the project:
```bash
git clone --branch v2.2.0/task https://github.com/ser-mk/nvcomp task
mkdir build_task
cd build_task
cmake -DTEST_TASK=ON -DBUILD_STATIC=ON ../task && pwd && \
VERBOSE=1 make -j && ls -h bin/ 
# Run the app
bin/task
```
You can use the docker image
https://hub.docker.com/layers/cuda/nvidia/cuda/11.1.1-devel-ubi8/images/sha256-c226bbca81bf493891124e3a575af96179734229cc91d9fead6b52dbf113144c?context=explore

I tested it using by Tesla T4.

The solution using the NVCOMP Library with my patch:
- Fix the Bitpacking scheme, Issue(https://github.com/NVIDIA/nvcomp/issues/59) ,patch(https://github.com/NVIDIA/nvcomp/commit/c779e47ea9396669e4c2e7eaca5aa5f66e60dfc3)
- New Delta encoding scheme, Issue(https://github.com/NVIDIA/nvcomp/issues/61) ,patch(https://github.com/NVIDIA/nvcomp/commit/fb6b66bb8cc855ec6fc2d31e6e04a15210e25509)
- No exception in NVCOMP lib, patch(https://github.com/NVIDIA/nvcomp/commit/1ea14f26b6abfba373c04aeba4e8759f9fe803fa). It's a dirty hack to avoid an exception, yet it is the simplest one. 

A workaround of https://github.com/NVIDIA/nvcomp/issues/63 issue fix in the task code(https://github.com/ser-mk/nvcomp/blob/v2.2.0/task2/task/task.cpp).
