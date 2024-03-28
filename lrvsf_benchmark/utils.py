from io import BytesIO
from time import perf_counter

import pyarrow as pa
from PIL import Image


def stream_to_PIL(img_bytes):
    return Image.open(BytesIO(img_bytes))


class catchtime:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.start
        self.readout = f"{self.name} - Time: {self.time:.3f} seconds"
        print(self.readout)


class FusedRecordBatch:
    def __init__(self, record_batch_iterator, target_batch_size):
        self.record_batch_iterator = record_batch_iterator
        self.target_batch_size = target_batch_size

    def __iter__(self):
        aggregator = []
        aggregator_size = 0

        for record_batch in self.record_batch_iterator:
            aggregator.append(record_batch)
            aggregator_size += len(record_batch)

            if aggregator_size >= self.target_batch_size:
                yield pa.Table.from_batches(aggregator)
                aggregator = []
                aggregator_size = 0

        if aggregator:
            yield pa.Table.from_batches(aggregator)


def get_number_of_params(model):
    return sum(p.numel() for p in model.parameters())


def human_format(num):
    num = float("{:.3g}".format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format(
        "{:f}".format(num).rstrip("0").rstrip("."), ["", "K", "M", "B", "T"][magnitude]
    )
