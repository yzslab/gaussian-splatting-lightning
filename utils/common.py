import os
import queue
import threading
from concurrent.futures.thread import ThreadPoolExecutor
import torch
import cv2


class AsyncTensorSaving:
    def __init__(self, maxsize: int = 16):
        self.queue = queue.Queue(maxsize=maxsize)
        self.thread = threading.Thread(target=self.save_tensor_from_queue)
        self.thread.start()

    def save_tensor_from_queue(self):
        while True:
            t = self.queue.get()
            if t is None:
                break
            tensor, path = t
            self.sync_save_tensor_to_file(tensor, path)

    def save_tensor(self, tensor, path):
        self.queue.put((tensor.cpu(), path))

    def stop(self):
        self.queue.put(None)
        self.thread.join()

    @staticmethod
    def sync_save_tensor_to_file(tensor, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(tensor, path + ".tmp")
        os.rename(path + ".tmp", path)


class AsyncImageSaving:
    def __init__(self, maxsize=16):
        self.queue = queue.Queue(maxsize=maxsize)
        self.thread = threading.Thread(target=self.save_image_from_queue)
        self.thread.start()

    def save_image(self, image, path):
        self.queue.put((image, path))

    def save_image_from_queue(self):
        while True:
            i = self.queue.get()
            if i is None:
                break
            image, path = i
            self.sync_save_image(image, path)

    def stop(self):
        self.queue.put(None)
        self.thread.join()

    @staticmethod
    def sync_save_image(image, path):
        cv2.imwrite(path, image)


class AsyncImageReading:
    def __init__(self, image_list: list, maxsize=16):
        self.image_list = image_list.copy()
        self.queue = queue.Queue(maxsize=maxsize)

        self.tpe = ThreadPoolExecutor(max_workers=4)

        self.thread = threading.Thread(target=self.read_images)
        self.thread.start()

    def read_images(self):
        futures = self.tpe.map(self.read_image_to_queue, self.image_list)
        for _ in futures:
            pass
        self.tpe.shutdown()

    def read_image_to_queue(self, image_path):
        self.queue.put((image_path, cv2.imread(image_path)))

