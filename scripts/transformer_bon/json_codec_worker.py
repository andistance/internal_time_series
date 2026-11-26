"""Private stdlib-only codec worker; executed as a file, never imports CUDA."""
import json
import io
import pickle
import struct
import sys


class _BuiltinUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Canonical trajectory records contain only builtin JSON types. Do not
        # let custom records import model/runtime modules into this CPU helper.
        raise ModuleNotFoundError("Custom object requires the local JSON serializer")


def read_exact(stream, size):
    data = bytearray()
    while len(data) < size:
        part = stream.read(size - len(data))
        if not part:
            raise EOFError("JSON codec pipe closed")
        data.extend(part)
    return bytes(data)


def main():
    source, target = sys.stdin.buffer, sys.stdout.buffer
    while True:
        header = source.read(8)
        if not header:
            return
        try:
            size = struct.unpack("!Q", header)[0]
            if size > 4 * 1024 * 1024:
                raise ValueError("JSON codec request exceeds its byte budget")
            operation, value = _BuiltinUnpickler(io.BytesIO(read_exact(source, size))).load()
            if operation == "encode":
                result = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
            elif operation == "decode":
                result = pickle.dumps(json.loads(value), protocol=4)
            else:
                raise ValueError("Unknown JSON codec operation")
            payload = b"\x00" + result
        except Exception as error:
            payload = b"\x01" + pickle.dumps(error, protocol=4)
        target.write(struct.pack("!Q", len(payload)))
        target.write(payload)
        target.flush()


if __name__ == "__main__":
    main()
