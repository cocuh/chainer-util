import queue
import threading

import multiprocessing as mp
import numpy as np

from chainer import cuda

try:
    import cupy
except:
    cupy = None


class HorizontalParallelEvaluator(object):
    def __init__(self, devices):
        self.devices = devices
        self.device_num = len(self.devices.keys())
        self._input_queue = mp.Queue(self.device_num)
        self._output_queue = mp.Queue()

        self._iter_worker = None
        self._eval_workers = []

    def run(self, model, iterator):
        self._iter_worker = mp.Process(target=_iter_worker, args=(iterator, self._input_queue, self.device_num))
        self._iter_worker.start()

        self._eval_workers = [
            mp.Process(
                target=EvalWorker(worker_id, device_id, self._input_queue, self._output_queue, model),
            )
            for worker_id, device_id in enumerate(self.devices.values())
            ]

        for workers in self._eval_workers:
            workers.start()

        try:
            self._join_workers()
        except KeyboardInterrupt:
            self._terminate_workers()
        return self._collect_result()

    def _join_workers(self):
        for workers in self._eval_workers:
            workers.join()
        self._iter_worker.join()

    def _collect_result(self):
        res = []
        while not self._output_queue.empty():
            res.append(self._output_queue.get())
        return res

    def _terminate_workers(self):
        for workers in self._eval_workers:
            workers.terminate()
        self._iter_worker.terminate()


def _iter_worker(iterator, input_queue, device_num):
    """
    disk -> cpu memory
    
    :param iterator: 
    :return: 
    """
    for data in iterator:
        input_queue.put(data)
    for i in range(device_num):
        input_queue.put(None)


class EvalWorker(object):
    def __init__(self, worker_id, device_id, input_queue, output_queue, model):
        self.worker_id = worker_id
        self.device_id = device_id
        self.input_queue = input_queue
        self.output_queue = output_queue

        self.prefetch_queue = queue.Queue(1)
        self.prefetch_thread = threading.Thread(target=self._prefetch)

        if self.device_id < 0:
            if hasattr(model, 'to_cpu'):
                model.to_cpu()
        else:
            if hasattr(model, 'to_gpu'):
                model.to_gpu(self.device_id)
        self.model = model

    def _prefetch(self):
        while True:
            data = self.input_queue.get()
            if data is None:
                self.prefetch_queue.put(None)
                break
            else:
                self.prefetch_queue.join()
                try:
                    data = self._to_xxx(data)
                except:  # TODO memory overflow error of cuda
                    pass
                self.prefetch_queue.put(data)

    def _to_xxx(self, arr):
        if self.device_id < 0:
            if isinstance(arr, np.ndarray):
                return cuda.to_cpu(arr)
            else:
                return arr
        else:
            if cuda is not None and isinstance(arr, cupy.ndarray):
                return cuda.to_gpu(arr, self.device_id)
            else:
                return arr

    def _to_cpu(self, arr):
        if isinstance(arr, np.ndarray):
            arr = cuda.to_cpu(arr)
        return arr

    def __call__(self):
        self.prefetch_thread.start()
        for data in iter(self.prefetch_queue.get, None):
            data = self._to_xxx(data)
            res = self.model(*data)
            self.output_queue.put((self._to_cpu(res),))
            del data
            self.prefetch_queue.task_done()

    def _join_processes(self):
        self.prefetch_thread.join()


def main():
    ev = HorizontalParallelEvaluator({
        'first': -1,
        'second': -1,
    })
    model = lambda x, y: \
        print('start', x[-1] + 1) or \
        __import__('time').sleep(3) or \
        print('end', x[-1] + 1) or \
        x.sum() + y.sum()

    def iterator():
        import numpy as np
        for i, j in zip(range(1, 101), range(10, 110)):
            print('start iter', i)
            __import__('time').sleep(1)
            print('end iter', i)
            yield np.arange(i), np.arange(j)

    ev.run(model, iterator())


if __name__ == '__main__':
    main()
