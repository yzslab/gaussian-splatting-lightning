import os
import queue
import threading
import concurrent.futures
from concurrent.futures.thread import ThreadPoolExecutor
import torch
import cv2


class AsyncTensorSaver:
    def __init__(self, maxsize: int = 16):
        self.queue = queue.Queue(maxsize=maxsize)
        self.thread = threading.Thread(target=self._save_tensor_from_queue)
        self.thread.start()

    def _save_tensor_from_queue(self):
        while True:
            t = self.queue.get()
            if t is None:
                break
            tensor, path = t
            self._sync_save_tensor_to_file(tensor, path)

    def save(self, tensor, path):
        self.queue.put((tensor.cpu(), path))

    def stop(self):
        self.queue.put(None)
        self.thread.join()

    @staticmethod
    def _sync_save_tensor_to_file(tensor, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(tensor, path + ".tmp")
        os.rename(path + ".tmp", path)


class AsyncImageSaver:
    def __init__(self, maxsize=16):
        self.queue = queue.Queue(maxsize=maxsize)
        self.thread = threading.Thread(target=self._save_image_from_queue)
        self.thread.start()

    def save(self, image, path):
        self.queue.put((image, path))

    def _save_image_from_queue(self):
        while True:
            i = self.queue.get()
            if i is None:
                break
            image, path = i
            self._sync_save_image(image, path)

    def stop(self):
        self.queue.put(None)
        self.thread.join()

    @staticmethod
    def _sync_save_image(image, path):
        dot_index = path.rfind(".")
        ext = path[dot_index:]

        tmp_path = f"{path[:dot_index]}.tmp{ext}"

        cv2.imwrite(tmp_path, image)
        os.rename(tmp_path, path)


class AsyncImageReader:
    def __init__(self, image_list: list, maxsize=16):
        self.image_list = image_list.copy()
        self.queue = queue.Queue(maxsize=maxsize)

        self.tpe = ThreadPoolExecutor(max_workers=4)

        self.thread = threading.Thread(target=self._read_images)
        self.thread.start()

        self.is_finished = False

    def get(self):
        return self.queue.get()

    def _read_images(self):
        results = self.tpe.map(self._read_image_to_queue, self.image_list)
        try:
            for _ in results:
                pass
        except concurrent.futures.CancelledError:
            pass
        self.tpe.shutdown()
        self.is_finished = True

    def _read_image_to_queue(self, image_path):
        self.queue.put((image_path, cv2.imread(image_path)))

    def stop(self):
        while self.is_finished is False:
            try:
                self.queue.get(block=False)
            except:
                pass
            self.tpe.shutdown(wait=False, cancel_futures=True)
